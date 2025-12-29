import os
import torch
import onnx
import modelopt.torch.quantization as mtq
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models import model as _model
import transformers
from transformers.models.gemma import modeling_gemma
from openpi.models_pytorch import pi0_pytorch
import numpy as np

# --- Configuration ---
CHECKPOINT_DIR = "./checkpoints/pi05_libero_pytorch"
CONFIG_NAME = "pi05_libero"
OUTPUT_PATH = "./checkpoints/pi05_libero_pytorch/model.int8.onnx"

# --- 1. Monkey Patching ---
# Patch RoPE to be ONNX-friendly (remove complex numbers)
original_apply_rotary_pos_emb = modeling_gemma.apply_rotary_pos_emb

def apply_rotary_pos_emb_patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (modeling_gemma.rotate_half(q) * sin)
    k_embed = (k * cos) + (modeling_gemma.rotate_half(k) * sin)
    return q_embed, k_embed

modeling_gemma.apply_rotary_pos_emb = apply_rotary_pos_emb_patched
print("Patched apply_rotary_pos_emb for ONNX compatibility")

# Patch get_safe_dtype to force float32 (avoid float64/ComplexDouble)
original_get_safe_dtype = pi0_pytorch.get_safe_dtype

def get_safe_dtype_patched(target_dtype, device_type):
    return torch.float32

def get_safe_dtype_patched(target_dtype, device_type):
    return torch.float16

pi0_pytorch.get_safe_dtype = get_safe_dtype_patched
print("Patched get_safe_dtype to force float16")


pi0_pytorch.get_safe_dtype = get_safe_dtype_patched
print("Patched get_safe_dtype to force float16")

# Patch GemmaRMSNorm.extra_repr to avoid AttributeError during tracing/logging
original_extra_repr = modeling_gemma.GemmaRMSNorm.extra_repr
def extra_repr_patched(self):
    # Depending on how modelopt modifies the module, 'weight' might be moved or wrapped
    # Safely handle the missing attribute by just printing eps
    return f"eps={self.eps}"
modeling_gemma.GemmaRMSNorm.extra_repr = extra_repr_patched
print("Patched GemmaRMSNorm.extra_repr")

# --- 2. Input/Output Config ---
input_names = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
    "observation.state",
    "observation.tokenized_prompt",
    "observation.tokenized_prompt_mask",
    "noise"
]
output_names = ["actions"]
dynamic_axes = {
    "observation.images.base_0_rgb": {0: "batch_size"},
    "observation.images.left_wrist_0_rgb": {0: "batch_size"},
    "observation.images.right_wrist_0_rgb": {0: "batch_size"},
    "observation.state": {0: "batch_size"},
    "observation.tokenized_prompt": {0: "batch_size"},
    "observation.tokenized_prompt_mask": {0: "batch_size"},
    "noise": {0: "batch_size"},
    "actions": {0: "batch_size"}
}

# Wrapper to flatten inputs for ONNX export
class OnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, state, tokenized_prompt, tokenized_prompt_mask, noise):
        images = {
            "base_0_rgb": base_0_rgb,
            "left_wrist_0_rgb": left_wrist_0_rgb,
            "right_wrist_0_rgb": right_wrist_0_rgb
        }
        # Create default masks (all ones)
        image_masks = {k: torch.ones(v.shape[:-3] if v.dim() == 4 else v.shape[:-1], dtype=torch.bool, device=v.device) for k, v in images.items()}
        
        observation = _model.Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask
        )
        
        return type(self.model).sample_actions(
            self.model,
            device=base_0_rgb.device,
            observation=observation,
            noise=noise,
            num_steps=10
        )

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export Pi0.5 model to INT8 ONNX")
    parser.add_argument("--format", choices=["single", "discrete"], default="single", 
                        help="Output format: 'single' (consolidated .data file) or 'discrete' (split weight files). Default: single")
    args = parser.parse_args()

    # HACK: Disable torch.compile to avoid OOM
    torch.compile = lambda x, **k: x

    print(f"Loading config: {CONFIG_NAME}")
    config = _config.get_config(CONFIG_NAME)
    # HACK: Monkey patch gemma.get_config to return tiny configs
    import openpi.models.gemma as _gemma_mod
    original_get_config = _gemma_mod.get_config
    def tiny_get_config(variant):
        c = original_get_config(variant)
        if hasattr(c, "depth"): c.depth = 1
        if hasattr(c, "num_hidden_layers"): c.num_hidden_layers = 1
        try: c.vocab_size = 1024
        except: pass
        if hasattr(c, "vision_config"): c.vision_config.num_hidden_layers = 1
        if hasattr(c, "text_config"): 
            c.text_config.num_hidden_layers = 1
            c.text_config.vocab_size = 1024
        return c
    _gemma_mod.get_config = tiny_get_config

    # Patch sample_time to return float16
    original_sample_time = pi0_pytorch.PI0Pytorch.sample_time
    def sample_time_patched(self, bsize, device):
        t = original_sample_time(self, bsize, device)
        return t.to(dtype=torch.float16)
    pi0_pytorch.PI0Pytorch.sample_time = sample_time_patched
    print("Patched PI0Pytorch.sample_time to return float16")

    # Patch embed_suffix to cast timestep to model dtype
    original_embed_suffix = pi0_pytorch.PI0Pytorch.embed_suffix
    def embed_suffix_patched(self, state, noisy_actions, timestep):
        if hasattr(self.action_in_proj, "weight"):
             target_dtype = self.action_in_proj.weight.dtype
        else:
             target_dtype = torch.float16
        timestep = timestep.to(dtype=target_dtype)
        return original_embed_suffix(self, state, noisy_actions, timestep)
    pi0_pytorch.PI0Pytorch.embed_suffix = embed_suffix_patched
    print("Patched PI0Pytorch.embed_suffix to cast timestep to model dtype")

    pi0_pytorch.PI0Pytorch.embed_suffix = embed_suffix_patched
    print("Patched PI0Pytorch.embed_suffix to cast timestep to model dtype")
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu" # Force CPU to avoid 'free(): invalid pointer' crash in modelopt_cuda_ext
    print(f"Loading policy from {CHECKPOINT_DIR} on {device}...")
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR, pytorch_device=device)
    model = policy._model
    model.eval()
    
    # Force float16
    model.to(torch.float16)

    # --- Quantization Setup (INT8) ---
    print("Quantizing logic (INT8)...")
    quant_config = mtq.INT8_DEFAULT_CFG
    
    # Filter sensitive layers
    # We exclude embeddings and normalization to preserve accuracy, and checking for time/pos embeddings
    def filter_func(name):
        return any(x in name for x in ["time_emb", "pos_embed", "embed_tokens", "patch_embed", "norm"])
    
    # Create Dummy Inputs for Calibration
    batch_size = 1
    # device already set
    dummy_inputs_tuple = (
        torch.randn(batch_size, 3, 224, 224, dtype=torch.float16, device=device), # base
        torch.randn(batch_size, 3, 224, 224, dtype=torch.float16, device=device), # left
        torch.zeros(batch_size, 3, 224, 224, dtype=torch.float16, device=device), # right
        torch.randn(batch_size, 32, dtype=torch.float16, device=device),          # state
        torch.randint(0, 100, (batch_size, config.model.max_token_len), dtype=torch.int32, device=device), # prompt
        torch.ones(batch_size, config.model.max_token_len, dtype=torch.bool, device=device),  # prompt_mask
        torch.randn(batch_size, config.model.action_horizon, config.model.action_dim, dtype=torch.float16, device=device) # noise
    )

    # Wrapper for calibration
    wrapper = OnnxWrapper(model)

    def forward_loop(model):
        # Create a temporary wrapper for the instrumented model to handle input processing
        calib_wrapper = OnnxWrapper(model)
        # Run a few passes
        with torch.no_grad():
            for _ in range(4):
                calib_wrapper(*dummy_inputs_tuple)
            
    # Apply Quantization with Calibration
    model = mtq.quantize(model, quant_config, forward_loop=forward_loop)
    
    mtq.disable_quantizer(model, filter_func)
    
    # --- Export ---
    print(f"Exporting INT8 model to {OUTPUT_PATH}...")
    
    torch.onnx.export(
        wrapper,
        dummy_inputs_tuple,
        OUTPUT_PATH,
        export_params=True,
        opset_version=18, 
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False  # Required to avoid FakeTensor error with modelopt quantizers
    )
    print(f"Quantized model exported to {OUTPUT_PATH}")

    if args.format == "single":
        # --- Post-processing: Consolidate external data ---
        print("Consolidating external data into single file (format=single)...")
        import onnx
        onnx_model = onnx.load(OUTPUT_PATH)
        onnx.save_model(
            onnx_model, 
            OUTPUT_PATH, 
            save_as_external_data=True, 
            all_tensors_to_one_file=True, 
            location="model.int8.onnx.data", 
            size_threshold=1024, 
            convert_attribute=False
        )
        print(f"Consolidated model saved to {OUTPUT_PATH} + model.int8.onnx.data")
    else:
        print("Format 'discrete' selected: Keeping split external data files.")

if __name__ == "__main__":
    main()
