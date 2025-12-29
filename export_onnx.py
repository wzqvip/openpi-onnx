import torch
import torch.nn as nn
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models import model as _model
import os
import argparse

# --- Configuration ---
# Default paths (can be overriden or changed here)
CHECKPOINT_DIR = "./checkpoints/pi05_libero_pytorch"
CONFIG_NAME = "pi05_libero"
OUTPUT_ONNX_PATH = "./checkpoints/pi05_libero_pytorch/model.onnx"

# --- Monkey Patches ---
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def custom_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # Custom implementation skipping complex numbers
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def patch_get_safe_dtype(target_dtype, device_type):
    return torch.float16

# --- Wrapper for Tracing ---
class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, 
                state, tokenized_prompt, tokenized_prompt_mask, noise):
        
        batch_size = base_0_rgb.shape[0]
        device = base_0_rgb.device
        
        # Reconstruct Observation object
        images = {
            "base_0_rgb": base_0_rgb,
            "left_wrist_0_rgb": left_wrist_0_rgb,
            "right_wrist_0_rgb": right_wrist_0_rgb,
        }
        image_masks = {k: torch.ones(batch_size, dtype=torch.bool, device=device) for k in images}
        
        obs = _model.Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask
        )
        
        # Call sample_actions directly
        return type(self.model).sample_actions(
            self.model,
            device=device,
            observation=obs,
            noise=noise,
            num_steps=10 
        )

def main():
    parser = argparse.ArgumentParser(description="Export Pi0.5 model to ONNX")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                        help="Export precision. Default: fp16")
    args = parser.parse_args()
    
    # OUTPUT PATH update based on dtype
    if args.dtype == "fp32":
        global OUTPUT_ONNX_PATH
        OUTPUT_ONNX_PATH = OUTPUT_ONNX_PATH.replace(".onnx", ".fp32.onnx")

    # HACK: Disable torch.compile to avoid OOM
    torch.compile = lambda x, **k: x

    # Apply Patches
    import transformers.models.gemma.modeling_gemma
    transformers.models.gemma.modeling_gemma.apply_rotary_pos_emb = custom_apply_rotary_pos_emb
    
    import openpi.models_pytorch.pi0_pytorch
    # Only patch safe_dtype to float16 if we are exporting fp16
    if args.dtype == "fp16":
        openpi.models_pytorch.pi0_pytorch.get_safe_dtype = patch_get_safe_dtype
    else:
        # For FP32, we map to float32
        def patch_get_safe_dtype_fp32(target, device): return torch.float32
        openpi.models_pytorch.pi0_pytorch.get_safe_dtype = patch_get_safe_dtype_fp32

    # Monkey-patch GemmaRMSNorm.extra_repr to avoid AttributeError during export logging
    def safe_extra_repr(self):
        try:
            return f"{tuple(self.weight.shape)}, eps={self.eps}"
        except AttributeError:
            return f"weight=<traced>, eps={self.eps}"
            
    transformers.models.gemma.modeling_gemma.GemmaRMSNorm.extra_repr = safe_extra_repr

    # Load Model
    config = _config.get_config(CONFIG_NAME)
    
    # HACK: Monkey patch gemma.get_config to return tiny configs
    # DISABLED for benchmarking full model
    import openpi.models.gemma as _gemma_mod
    original_get_config = _gemma_mod.get_config
    # def tiny_get_config(variant):
    #     c = original_get_config(variant)
    #     if hasattr(c, "depth"): c.depth = 1
    #     if hasattr(c, "num_hidden_layers"): c.num_hidden_layers = 1
    #     try: c.vocab_size = 1024
    #     except: pass
    #     if hasattr(c, "vision_config"): c.vision_config.num_hidden_layers = 1
    #     if hasattr(c, "text_config"): 
    #         c.text_config.num_hidden_layers = 1
    #         c.text_config.vocab_size = 1024
    #     return c
    # _gemma_mod.get_config = tiny_get_config
    # print("Monkey-patched openpi.models.gemma.get_config for tiny model.")
    print("Tiny model patch DISABLED.")
    
    # Patch sample_time
    original_sample_time = openpi.models_pytorch.pi0_pytorch.PI0Pytorch.sample_time
    def sample_time_patched(self, bsize, device):
        t = original_sample_time(self, bsize, device)
        return t.to(dtype=torch.float16 if args.dtype=="fp16" else torch.float32)
    openpi.models_pytorch.pi0_pytorch.PI0Pytorch.sample_time = sample_time_patched
    print(f"Patched PI0Pytorch.sample_time to return {args.dtype}")

    # Patch embed_suffix
    original_embed_suffix = openpi.models_pytorch.pi0_pytorch.PI0Pytorch.embed_suffix
    import sys
    def embed_suffix_patched(self, state, noisy_actions, timestep):
        # Force correct dtype
        target_dtype = torch.float16 if args.dtype=="fp16" else torch.float32
        print(f"DEBUG: embed_suffix_patched called. timestep.dtype={timestep.dtype}, casting to {target_dtype}")
        sys.stdout.flush()
        timestep = timestep.to(dtype=target_dtype)
        # Verify cast
        print(f"DEBUG: timestep cast to {timestep.dtype}")
        sys.stdout.flush()
        return original_embed_suffix(self, state, noisy_actions, timestep)
    openpi.models_pytorch.pi0_pytorch.PI0Pytorch.embed_suffix = embed_suffix_patched
    print(f"Patched PI0Pytorch.embed_suffix to cast timestep to {args.dtype}")
    
    # Force CPU for export to ensure stability (avoid VRAM OOM for FP32)
    device = "cpu"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading policy from {CHECKPOINT_DIR} on {device}...")
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR, pytorch_device=device)
    model = policy._model
    
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    
    # Model Dtype
    model_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model.to(model_dtype)
    model.eval()
    
    wrapper = OnnxWrapper(model)
    
    # Dummy Inputs [Batch, Channel, Height, Width]
    B = 1
    dummy_inputs = (
        torch.randn(B, 3, 224, 224, dtype=model_dtype, device=device), # base
        torch.randn(B, 3, 224, 224, dtype=model_dtype, device=device), # left_wrist
        torch.zeros(B, 3, 224, 224, dtype=model_dtype, device=device), # right_wrist
        torch.randn(B, 32, dtype=model_dtype, device=device),          # state
        torch.randint(0, 100, (B, config.model.max_token_len), dtype=torch.int32, device=device), # prompt
        torch.ones(B, config.model.max_token_len, dtype=torch.bool, device=device),  # prompt_mask
        torch.randn(B, config.model.action_horizon, config.model.action_dim, dtype=model_dtype, device=device) # noise
    )
    
    input_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", 
                   "state", "tokenized_prompt", "tokenized_prompt_mask", "noise"]
    output_names = ["actions"]
    
    print(f"Exporting to {OUTPUT_ONNX_PATH}...")
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        OUTPUT_ONNX_PATH,
        opset_version=18,  # Increased to mismatch 2.11 default and avoid buggy version converter
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={k: {0: "batch_size"} for k in input_names + output_names}
    )
    print(f"Exported to {OUTPUT_ONNX_PATH}")

    print(f"Exported to {OUTPUT_ONNX_PATH}")

    # Always consolidate if single file is desired
    # if args.format == "single":
    if True:
        print("Consolidating external data...")
        import onnx
        onnx_model = onnx.load(OUTPUT_ONNX_PATH)
        onnx.save_model(
            onnx_model, 
            OUTPUT_ONNX_PATH, 
            save_as_external_data=True, 
            all_tensors_to_one_file=True, 
            location=os.path.basename(OUTPUT_ONNX_PATH) + ".data", 
            size_threshold=1024, 
            convert_attribute=False
        )
        print(f"Consolidated to {OUTPUT_ONNX_PATH}.data")

if __name__ == "__main__":
    main()
