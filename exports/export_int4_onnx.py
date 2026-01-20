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
OUTPUT_PATH = "./checkpoints/pi05_libero_pytorch/model.int4.onnx"

# --- 1. Monkey Patching ---
original_apply_rotary_pos_emb = modeling_gemma.apply_rotary_pos_emb

def apply_rotary_pos_emb_patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (modeling_gemma.rotate_half(q) * sin)
    k_embed = (k * cos) + (modeling_gemma.rotate_half(k) * sin)
    return q_embed, k_embed

modeling_gemma.apply_rotary_pos_emb = apply_rotary_pos_emb_patched
print("Patched apply_rotary_pos_emb for ONNX compatibility")

original_get_safe_dtype = pi0_pytorch.get_safe_dtype

def get_safe_dtype_patched(target_dtype, device_type):
    if not torch.cuda.is_available():
        return torch.float32
    return torch.float16

pi0_pytorch.get_safe_dtype = get_safe_dtype_patched
print("Patched get_safe_dtype to force float16 (or float32 on CPU)")

original_extra_repr = modeling_gemma.GemmaRMSNorm.extra_repr
def extra_repr_patched(self):
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
    # HACK: Monkey patch gemma.get_config to return tiny configs
    # DISABLED
    import openpi.models.gemma as _gemma_mod
    config = _config.get_config(CONFIG_NAME)
    original_get_config = _gemma_mod.get_config
    print("Tiny model patch DISABLED.")
    original_sample_time = pi0_pytorch.PI0Pytorch.sample_time
    def sample_time_patched(self, bsize, device):
        t = original_sample_time(self, bsize, device)
        target_dtype = torch.float32 if device == "cpu" or (isinstance(device, torch.device) and device.type == "cpu") else torch.float16
        return t.to(dtype=target_dtype)
    pi0_pytorch.PI0Pytorch.sample_time = sample_time_patched
    print("Patched PI0Pytorch.sample_time to return correct dtype")

    # Patch embed_suffix to cast timestep to model dtype
    original_embed_suffix = pi0_pytorch.PI0Pytorch.embed_suffix
    def embed_suffix_patched(self, state, noisy_actions, timestep):
        if hasattr(self.action_in_proj, "weight"):
             target_dtype = self.action_in_proj.weight.dtype
        else:
             target_dtype = torch.float32
        timestep = timestep.to(dtype=target_dtype)
        return original_embed_suffix(self, state, noisy_actions, timestep)
    pi0_pytorch.PI0Pytorch.embed_suffix = embed_suffix_patched
    print("Patched PI0Pytorch.embed_suffix to cast timestep to model dtype")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading policy from {CHECKPOINT_DIR} on {device}...")
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR, pytorch_device=device)
    model = policy._model
    model.eval()
    
    # Determine dtype based on device
    # Force Float32 for export stability (ModelOpt handles QDQ)
    export_dtype = torch.float32
    print("Exporting: Forcing Float32 precision for stability.")
    # if device == "cpu":
    #     export_dtype = torch.float32
    #     print("Exporting on CPU: Forcing Float32 precision.")
    # else:
    #     export_dtype = torch.float16
    #     print("Exporting on GPU: Using Float16 precision.")

    model.to(export_dtype)

    # --- Quantization Setup (INT4) ---
    print("Quantizing logic (INT4 weight only)...")
    # Using INT4_BLOCKWISE_WEIGHT_ONLY_CFG
    quant_config = mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG
    
    def filter_func(name):
        return any(x in name for x in ["time_emb", "pos_embed", "embed_tokens", "patch_embed", "norm"])
    
    batch_size = 1
    dummy_inputs_tuple = (
        torch.randn(batch_size, 3, 224, 224, dtype=export_dtype, device=device), # base
        torch.randn(batch_size, 3, 224, 224, dtype=export_dtype, device=device), # left
        torch.zeros(batch_size, 3, 224, 224, dtype=export_dtype, device=device), # right
        torch.randn(batch_size, 32, dtype=export_dtype, device=device),          # state
        torch.randint(0, 100, (batch_size, config.model.max_token_len), dtype=torch.int32, device=device), # prompt
        torch.ones(batch_size, config.model.max_token_len, dtype=torch.bool, device=device),  # prompt_mask
        torch.randn(batch_size, config.model.action_horizon, config.model.action_dim, dtype=export_dtype, device=device) # noise
    )

    wrapper = OnnxWrapper(model)

    def forward_loop(model):
        calib_wrapper = OnnxWrapper(model)
        with torch.no_grad():
            for _ in range(4):
                calib_wrapper(*dummy_inputs_tuple)
            
    # Apply Quantization with Calibration
    model = mtq.quantize(model, quant_config, forward_loop=forward_loop)
    
    mtq.disable_quantizer(model, filter_func)
    
    print(f"Exporting INT4 model to {OUTPUT_PATH}...")
    
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
        dynamo=False
    )
    print(f"Quantized model exported to {OUTPUT_PATH}")

    # Consolidate
    print("Consolidating external data...")
    import onnx
    onnx_model = onnx.load(OUTPUT_PATH)
    onnx.save_model(
        onnx_model, 
        OUTPUT_PATH, 
        save_as_external_data=True, 
        all_tensors_to_one_file=True, 
        location="model.int4.onnx.data", 
        size_threshold=1024, 
        convert_attribute=False
    )
    print(f"Consolidated model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
