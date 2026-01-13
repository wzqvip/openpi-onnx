
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
import copy

# --- Configuration ---
CHECKPOINT_DIR = "./checkpoints/pi05_libero_pytorch"
CONFIG_NAME = "pi05_libero"
OUTPUT_PATH = "./checkpoints/pi05_libero_pytorch/model.w8a16.onnx"

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
    if not torch.cuda.is_available():
        return torch.float32
    return torch.float16

pi0_pytorch.get_safe_dtype = get_safe_dtype_patched
print("Patched get_safe_dtype to force float16 (or float32 on CPU)")

# Patch GemmaRMSNorm.extra_repr
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

# Wrapper
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
    import argparse
    parser = argparse.ArgumentParser(description="Export Pi0.5 model to W8A16 ONNX")
    parser.add_argument("--format", choices=["single", "discrete"], default="single", 
                        help="Output format: 'single' (consolidated .data file) or 'discrete' (split weight files). Default: single")
    args = parser.parse_args()

    # Disable torch.compile
    torch.compile = lambda x, **k: x

    print(f"Loading config: {CONFIG_NAME}")
    config = _config.get_config(CONFIG_NAME)
    
    # HACK: Disable tiny model patch
    import openpi.models.gemma as _gemma_mod
    original_get_config = _gemma_mod.get_config
    print("Tiny model patch DISABLED.")

    # Patch sample_time
    original_sample_time = pi0_pytorch.PI0Pytorch.sample_time
    def sample_time_patched(self, bsize, device):
        t = original_sample_time(self, bsize, device)
        # Check if we are running fp16 or fp32
        # We will determine target dtype later
        return t
    pi0_pytorch.PI0Pytorch.sample_time = sample_time_patched

    device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    # Force CPU to avoid CUDA kernel errors on Thor
    print(f"Loading policy from {CHECKPOINT_DIR} on {device}...")
    
    # Use FP32 if on CPU to avoid "Float vs Half" errors
    # If on GPU, we can try FP16 directly, but let's stick to FP32->FP16 for stability if needed.
    # Actually, for W8A16, running in FP32 during calibration/export is safer for modelopt, then we cast.
    # But modelopt quantizes based on current weight values.
    # Let's use FP32 for the model execution.
    exec_dtype = torch.float32
    if device == "cuda":
        # We can try float16 on cuda
        # But let's stick to FP32 fallback logic for consistency if CPU is used.
        pass

    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR, pytorch_device=device)
    model = policy._model
    model.eval()
    model.to(dtype=exec_dtype)
    
    # Patch sample time again to match exec_dtype
    def sample_time_patched_2(self, bsize, device):
        t = original_sample_time(self, bsize, device)
        return t.to(dtype=exec_dtype)
    pi0_pytorch.PI0Pytorch.sample_time = sample_time_patched_2
    
    # Patch embed_suffix to match exec_dtype
    original_embed_suffix = pi0_pytorch.PI0Pytorch.embed_suffix
    def embed_suffix_match(self, state, noisy_actions, timestep):
        timestep = timestep.to(dtype=exec_dtype)
        return original_embed_suffix(self, state, noisy_actions, timestep)
    pi0_pytorch.PI0Pytorch.embed_suffix = embed_suffix_match
    

    # --- Quantization Setup (W8A16) ---
    print("Quantizing logic (W8A16 - Weight Only INT8)...")
    
    # W8A16 Config: Enable Weight Quantizer (INT8), Disable Input Quantizer
    quant_config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    quant_config["quant_cfg"]["*input_quantizer"] = {"enable": False}
    
    def filter_func(name):
        return any(x in name for x in ["time_emb", "pos_embed", "embed_tokens", "patch_embed", "norm"])
    
    print("Applying quantization (Weight Only)...")
    # Quantize in FP32 mode
    model = mtq.quantize(model, quant_config) 
    mtq.disable_quantizer(model, filter_func)
    
    wrapper = OnnxWrapper(model)
    
    # Dummy Inputs (FP32)
    batch_size = 1
    dummy_inputs_tuple = (
        torch.randn(batch_size, 3, 224, 224, dtype=exec_dtype, device=device), # base
        torch.randn(batch_size, 3, 224, 224, dtype=exec_dtype, device=device), # left
        torch.zeros(batch_size, 3, 224, 224, dtype=exec_dtype, device=device), # right
        torch.randn(batch_size, 32, dtype=exec_dtype, device=device),          # state
        torch.randint(0, 100, (batch_size, config.model.max_token_len), dtype=torch.int32, device=device), # prompt
        torch.ones(batch_size, config.model.max_token_len, dtype=torch.bool, device=device),  # prompt_mask
        torch.randn(batch_size, config.model.action_horizon, config.model.action_dim, dtype=exec_dtype, device=device) # noise
    )

    # --- Export ---
    print(f"Exporting W8A16 model (initially FP32 trace) to {OUTPUT_PATH}...")
    
    # Temporary path
    temp_path = OUTPUT_PATH + ".temp.onnx"
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_inputs_tuple,
        temp_path,
        export_params=True,
        opset_version=18, 
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False
    )
    print(f"Temp FP32-quantized model exported to {temp_path}")

    # --- Convert to FP16 (W8A16) ---
    # --- Convert to FP16 (W8A16) ---
    # SKIPPING FP16 conversion due to corruption bug.
    # The exported model (temp_path) has QDQ nodes. Activations are FP32 in graph, but TRT --fp16 handles mixed precision.
    print("Skipping broken FP16 conversion. Using FP32 export with QDQ nodes...")
    
    import onnx
    model_export = onnx.load(temp_path)
    
    # PATCH CumSum directly here
    print("Patching CumSum nodes...")
    from onnx import helper, TensorProto
    new_nodes = []
    patched_count = 0
    for node in model_export.graph.node:
        if node.op_type == "CumSum":
            input_name = node.input[0]
            cast_out = input_name + "_cast_int32"
            cast_node = helper.make_node(
                "Cast",
                inputs=[input_name],
                outputs=[cast_out],
                to=TensorProto.INT32,
                name=node.name + "_cast_patch"
            )
            node.input[0] = cast_out
            new_nodes.append(cast_node)
            new_nodes.append(node)
            patched_count += 1
        else:
            new_nodes.append(node)
    
    if patched_count > 0:
        model_export.graph.ClearField("node")
        model_export.graph.node.extend(new_nodes)
        print(f"Patched {patched_count} CumSum nodes.")

    onnx.save(model_export, OUTPUT_PATH)
    print(f"Saved W8A16 (QDQ) model to {OUTPUT_PATH}")
    
    
    # if os.path.exists(temp_path):
    #     os.remove(temp_path)
    
    print(f"DEBUG: File size of {OUTPUT_PATH}: {os.path.getsize(OUTPUT_PATH)} bytes")

    if args.format == "single":
        print("Consolidating external data...")
        import onnx
        onnx_model = onnx.load(OUTPUT_PATH)
        onnx.save_model(
            onnx_model, 
            OUTPUT_PATH, 
            save_as_external_data=True, 
            all_tensors_to_one_file=True, 
            location=os.path.basename(OUTPUT_PATH) + ".data", 
            size_threshold=1024, 
            convert_attribute=False
        )
        print(f"Consolidated model saved to {OUTPUT_PATH} + .data")

if __name__ == "__main__":
    main()
