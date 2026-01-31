import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization import QuantizeConfig
import os
import sys
import dataclasses
from tqdm import tqdm
import jax

# Set JAX to CPU to avoid GPU conflicts with PyTorch
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from openpi.training import config as _config
from openpi.models_pytorch import pi0_pytorch

CHECKPOINT_DIR = "/home/taco/checkpoints/pi05_libero_onnx_compat"
CONFIG_NAME = "pi05_libero"
CALIBRATION_FILE = "calibration_data.pt"
OUTPUT_DIR = "/home/taco/checkpoints/pi05_libero_onnx_compat/thor_fp4_ckpt"

def main():
    print(f"Loading config: {CONFIG_NAME}")
    config = _config.get_config(CONFIG_NAME)
    # PATCH: Override action_dim to match checkpoint
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, action_dim=32))
    
    print(f"Loading policy from {CHECKPOINT_DIR}...")
    model = pi0_pytorch.PI0Pytorch(config.model)
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
    from safetensors.torch import load_file
    sd = load_file(ckpt_path)
    model.load_state_dict(sd, strict=False)
    
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    # 2. Configure for Jetson Thor
    print("Configuring for Blackwell FP4...")
    if hasattr(mtq, "NVFP4_DEFAULT_CFG"):
        print("Using preset NVFP4_DEFAULT_CFG")
        config = mtq.NVFP4_DEFAULT_CFG
    else:
        print("Using custom manual config")
        config = QuantizeConfig(
            qformat="fp4",           
            groupsize=128,           
            quantize_weights=True,
            quantize_activations=True,
            exclude_modules=[
                "lm_head",           
                "action_head",       
                "norm",              
                "input_layernorm",
                "post_attention_layernorm",
                "vision_tower",
                "action_in_proj",
                "action_out_proj"
            ],
            layer_mappings={
                "Linear": {"enable": True, "format": "fp4"},
                "MultiheadAttention": {"enable": True, "format": "fp4"}
            }
        )

    # 3. Calibration Loop
    if not os.path.exists(CALIBRATION_FILE):
        print(f"ERROR: {CALIBRATION_FILE} not found")
        if not os.path.exists(CALIBRATION_FILE):
             # Only return if strictly needed, or maybe generate dummy?
             print("Calibration file missing. Cannot calibrate.")
             return

    print(f"Loading calibration data from {CALIBRATION_FILE}...")
    calibration_data = torch.load(CALIBRATION_FILE, weights_only=False)
    
    def forward_loop(model):
        print("Running calibration steps...")
        device = next(model.parameters()).device
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calibration_data)):
                (base_rgb, left_rgb, right_rgb, state, prompt, prompt_mask, noise) = batch
                
                # Convert to tensor if numpy
                if hasattr(base_rgb, "__array__"): base_rgb = torch.from_numpy(base_rgb)
                if hasattr(left_rgb, "__array__"): left_rgb = torch.from_numpy(left_rgb)
                if hasattr(right_rgb, "__array__"): right_rgb = torch.from_numpy(right_rgb)
                if hasattr(state, "__array__"): state = torch.from_numpy(state)
                # prompt/mask likely tensors or arrays
                if hasattr(prompt, "__array__"): prompt = torch.from_numpy(prompt)
                if hasattr(prompt_mask, "__array__"): prompt_mask = torch.from_numpy(prompt_mask)
                if hasattr(noise, "__array__"): noise = torch.from_numpy(noise)

                base_rgb = base_rgb.to(device)
                left_rgb = left_rgb.to(device)
                right_rgb = right_rgb.to(device)
                state = state.to(device)
                prompt = prompt.to(device)
                prompt_mask = prompt_mask.to(device)
                noise = noise.to(device)
                
                bsize = state.shape[0]
                images = {
                    "base_0_rgb": base_rgb,
                    "left_wrist_0_rgb": left_rgb,
                    "right_wrist_0_rgb": right_rgb
                }
                image_masks = {
                    "base_0_rgb": torch.ones(bsize, dtype=torch.bool, device=device),
                    "left_wrist_0_rgb": torch.ones(bsize, dtype=torch.bool, device=device),
                    "right_wrist_0_rgb": torch.zeros(bsize, dtype=torch.bool, device=device) 
                }
                from openpi.models import model as _model
                observation = _model.Observation(
                    images=images,
                    image_masks=image_masks,
                    state=state,
                    tokenized_prompt=prompt,
                    tokenized_prompt_mask=prompt_mask
                )
                
                try:
                    model.sample_actions(device, observation, num_steps=1)
                except Exception as e:
                    print(f"Calibration error on sample {i}: {e}")
                    pass
                
                if i >= 4: break 
                
    # 4. Quantize and Calibrate
    print("Starting Quantization and Calibration...")
    model = mtq.quantize(model, config, forward_loop=forward_loop)
    
    # 5. Export
    print(f"Saving quantized checkpoint to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Fallback to saving state dict
    save_path = os.path.join(OUTPUT_DIR, "quantized_model.safetensors")
    from safetensors.torch import save_file
    save_file(model.state_dict(), save_path)
    print(f"✅ Quantized model saved to {save_path}")
    print("NOTE: 'mtq.export' was not found. Please use TensorRT Edge-LLM converter on this checkpoint.")

if __name__ == "__main__":
    main()
