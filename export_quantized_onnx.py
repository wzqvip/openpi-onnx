import torch
import torch.nn as nn
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models import model as _model
import os
import re

# ModelOpt Import
import modelopt.torch.quantization as mtq
try:
    import modelopt.torch.quantization.export_onnx
    print("Imported modelopt.torch.quantization.export_onnx")
except ImportError:
    print("Could not import modelopt.torch.quantization.export_onnx")

# --- Configuration ---
CHECKPOINT_DIR = "./checkpoints/pi05_libero_pytorch"
CONFIG_NAME = "pi05_libero"
OUTPUT_ONNX_PATH = "./checkpoints/pi05_libero_pytorch/model.mxfp8.onnx"

class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, 
                state, tokenized_prompt, tokenized_prompt_mask, noise):
        
        batch_size = base_0_rgb.shape[0]
        device = base_0_rgb.device
        
        # Create Observation
        images = {
            "base_0_rgb": base_0_rgb,
            "left_wrist_0_rgb": left_wrist_0_rgb,
            "right_wrist_0_rgb": right_wrist_0_rgb,
        }
        
        image_masks = {
            "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        }
        
        # Reconstruct Observation
        obs = _model.Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask
        )
        
        # Bypass torch.compile by calling the class method
        return type(self.model).sample_actions(
            self.model,
            device=device,
            observation=obs,
            noise=noise,
            num_steps=10 
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def custom_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors (Monkey-patched)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def patch_get_safe_dtype(target_dtype, device_type):
    """Patch to force float32 to avoid float64/ComplexDouble in ONNX."""
    return torch.float16

def filter_func(name):
    """Filter function to exclude certain layers from quantization."""
    # Exclude embeddings, norms, and time projections which typically need higher precision
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|"
        r"pos_embed|time_text_embed|context_embedder|norm|norm_out|x_embedder|patch_embed|embed_tokens).*"
    )
    return pattern.match(name) is not None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export Pi0.5 model to Quantized ONNX (MXFP8)")
    parser.add_argument("--format", choices=["single", "discrete"], default="single", 
                        help="Output format: 'single' (consolidated .data file) or 'discrete' (split weight files). Default: single")
    args = parser.parse_args()

    # HACK: Disable torch.compile to avoid OOM
    torch.compile = lambda x, **k: x

    print(f"Loading config: {CONFIG_NAME}")
    
    # --- Apply Patches ---
    import transformers.models.gemma.modeling_gemma
    transformers.models.gemma.modeling_gemma.apply_rotary_pos_emb = custom_apply_rotary_pos_emb
    print("Patched apply_rotary_pos_emb for ONNX compatibility")
    
    # MANUAL REGISTRATION OF MISSING SYMBOLIC
    try:
        from torch.onnx import register_custom_op_symbolic
        import modelopt.torch.quantization.export_onnx as export_module

        def dynamic_block_quantize_op_symbolic(g, inputs, block_size, amax, num_bits, exponent_bits, scale_num_bits, scale_exponent_bits):
            # Check for MXFP8 configuration: (8, 4) -> E4M3 and (9, 8) -> E8M0
            # Note: args come as constants/values. symbolic_helper might be needed if they are internal nodes, 
            # but usually scalar args are passed directly or can be inferred.
            # Assuming standard MXFP8 for now as per logs.
            return export_module.export_mxfp8(
                g,
                inputs,
                onnx_quantizer_type="dynamic",
                block_size=32 # Hardcoding or extracting? Log said 32. 
                # extract scalar if possible? 
                # sym_help._parse_arg(block_size, 'i')? 
                # Let's just pass block_size assuming it behaves like the Function symbolic
            )
        
        register_custom_op_symbolic("tensorrt::dynamic_block_quantize_op", dynamic_block_quantize_op_symbolic, 17)
        register_custom_op_symbolic("tensorrt.dynamic_block_quantize_op", dynamic_block_quantize_op_symbolic, 17)
        register_custom_op_symbolic("dynamic_block_quantize_op", dynamic_block_quantize_op_symbolic, 17)
        print("Registered custom symbolic for tensorrt::dynamic_block_quantize_op (and variants)")
    except Exception as e:
        print(f"Failed to register custom symbolic: {e}")

    import openpi.models_pytorch.pi0_pytorch
    openpi.models_pytorch.pi0_pytorch.get_safe_dtype = patch_get_safe_dtype
    print("Patched get_safe_dtype to force float32")

    config = _config.get_config(CONFIG_NAME)
    # HACK: Monkey patch gemma.get_config to return tiny configs
    import openpi.models.gemma as _gemma_mod
    original_get_config = _gemma_mod.get_config
    # Monkey-patch GemmaRMSNorm.extra_repr to avoid AttributeError if weight is missing
    # (Common during export/quantization if weights are manipulated)
    from transformers.models.gemma import modeling_gemma
    def safe_extra_repr(self):
        try:
            return f"{tuple(self.weight.shape)}, eps={self.eps}"
        except AttributeError:
            return f"eps={self.eps} (weight missing)"
    modeling_gemma.GemmaRMSNorm.extra_repr = safe_extra_repr

    # Tiny model patching
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
    original_sample_time = openpi.models_pytorch.pi0_pytorch.PI0Pytorch.sample_time
    def sample_time_patched(self, bsize, device):
        t = original_sample_time(self, bsize, device)
        return t.to(dtype=torch.float16)
    openpi.models_pytorch.pi0_pytorch.PI0Pytorch.sample_time = sample_time_patched
    print("Patched PI0Pytorch.sample_time to return float16")

    # Patch embed_suffix to cast timestep to model dtype
    original_embed_suffix = openpi.models_pytorch.pi0_pytorch.PI0Pytorch.embed_suffix
    def embed_suffix_patched(self, state, noisy_actions, timestep):
        # Ensure we don't access uninitialized weights if quantization replaced them? 
        # But action_in_proj acts as a good reference.
        if hasattr(self.action_in_proj, "weight"):
             target_dtype = self.action_in_proj.weight.dtype
        else:
             target_dtype = torch.float16 # Default fall-back
        timestep = timestep.to(dtype=target_dtype)
        return original_embed_suffix(self, state, noisy_actions, timestep)
    openpi.models_pytorch.pi0_pytorch.PI0Pytorch.embed_suffix = embed_suffix_patched
    print("Patched PI0Pytorch.embed_suffix to cast timestep to model dtype")
    
    print(f"Loading policy from: {CHECKPOINT_DIR}")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Exporting on device: {device_name}")
    
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR, pytorch_device=device_name)
    model = policy._model
    
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        
    print("Converting model to float16...")
    model.to(torch.float16)
    model.eval()
    
    # --- Quantization (MXFP8) ---
    print("Applying MXFP8 Quantization...")
    # Using MXFP8 Default Config
    quant_config = mtq.MXFP8_DEFAULT_CFG
    
    model = mtq.quantize(model, quant_config)
    
    # Disable quantization for sensitive layers
    mtq.disable_quantizer(model, filter_func)
    
    mtq.print_quant_summary(model)
    
    # --- Export ---
    wrapper = OnnxWrapper(model)
    
    # Create dummy inputs
    B = 1
    dummy_inputs = (
        torch.randn(B, 3, 224, 224, dtype=torch.float16, device=device_name), # base
        torch.randn(B, 3, 224, 224, dtype=torch.float16, device=device_name), # left_wrist
        torch.zeros(B, 3, 224, 224, dtype=torch.float16, device=device_name), # right_wrist
        torch.randn(B, 32, dtype=torch.float16, device=device_name),          # state
        torch.randint(0, 100, (B, config.model.max_token_len), dtype=torch.int32, device=device_name), # prompt
        torch.ones(B, config.model.max_token_len, dtype=torch.bool, device=device_name),  # prompt_mask
        torch.randn(B, config.model.action_horizon, config.model.action_dim, dtype=torch.float16, device=device_name) # noise
    )
    
    print(f"Exporting to {OUTPUT_ONNX_PATH}...")
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        OUTPUT_ONNX_PATH,
        export_params=True,
        opset_version=17, 
        do_constant_folding=True,
        dynamo=False, # Disable Dynamo to use legacy tracing and modelopt's function symbolic
        input_names=[
            "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", 
            "state", "tokenized_prompt", "tokenized_prompt_mask", "noise"
        ],
        output_names=["actions"],
        dynamic_axes={
            "base_0_rgb": {0: "batch_size"},
            "left_wrist_0_rgb": {0: "batch_size"},
            "right_wrist_0_rgb": {0: "batch_size"},
            "state": {0: "batch_size"},
            "tokenized_prompt": {0: "batch_size"},
            "tokenized_prompt_mask": {0: "batch_size"},
            "noise": {0: "batch_size"},
            "actions": {0: "batch_size"}
        }
    )
    print(f"Quantized model exported to {OUTPUT_ONNX_PATH}")

    if args.format == "single":
        # --- Post-processing: Consolidate external data ---
        print("Consolidating external data into single file (format=single)...")
        import onnx
        onnx_model = onnx.load(OUTPUT_ONNX_PATH)
        onnx.save_model(
            onnx_model, 
            OUTPUT_ONNX_PATH, 
            save_as_external_data=True, 
            all_tensors_to_one_file=True, 
            location=f"{os.path.basename(OUTPUT_ONNX_PATH)}.data", 
            size_threshold=1024, 
            convert_attribute=False
        )
        print(f"Consolidated model saved to {OUTPUT_ONNX_PATH} + .data")
    else:
        print("Format 'discrete' selected: Keeping split external data files.")

if __name__ == "__main__":
    main()
