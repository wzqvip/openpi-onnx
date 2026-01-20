
import torch
import torch.nn as nn
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models import model as _model
import os
import argparse
import onnx
from onnxconverter_common import float16
from onnx import helper, TensorProto
import transformers.models.gemma.modeling_gemma
import openpi.models_pytorch.pi0_pytorch
import sys

# --- Configuration ---
CHECKPOINT_DIR = "./checkpoints/pi05_libero_pytorch"
CONFIG_NAME = "pi05_libero"
OUTPUT_ONNX_PATH = "./checkpoints/pi05_libero_pytorch/model.fp16.onnx"
TEMP_FP32_PATH = "./checkpoints/pi05_libero_pytorch/model.fp16.temp_fp32.onnx"

# --- Monkey Patches ---
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def custom_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Wrapper for Tracing
class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, 
                state, tokenized_prompt, tokenized_prompt_mask, noise):
        
        batch_size = base_0_rgb.shape[0]
        device = base_0_rgb.device
        
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
        
        return type(self.model).sample_actions(
            self.model,
            device=device,
            observation=obs,
            noise=noise,
            num_steps=10 
        )

def patch_cumsum_nodes(model):
    print("Scanning for CumSum nodes to patch...")
    graph = model.graph
    node_out_type = {}
    for node in graph.node:
        for out in node.output:
            node_out_type[out] = node.op_type
            
    new_nodes = []
    patched_count = 0
    
    for node in graph.node:
        if node.op_type == "CumSum":
            input_name = node.input[0]
            # Always patch CumSum to take Int32 if correct?
            # Or assume if we put a Cast(Int32) it works.
            # Let's insert Cast to Int32
            
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
            
    print(f"Patched {patched_count} CumSum nodes.")
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    return model

def main():
    print("--- Starting Safe FP16 Export ---")
    
    # Disable torch compile
    torch.compile = lambda x, **k: x

    # 1. Apply Patches
    transformers.models.gemma.modeling_gemma.apply_rotary_pos_emb = custom_apply_rotary_pos_emb
    
    # Force get_safe_dtype to return float32 (since we are tracing in fp32)
    openpi.models_pytorch.pi0_pytorch.get_safe_dtype = lambda target, device: torch.float32

    # Patch Extra Repr
    def safe_extra_repr(self):
        try:
            return f"{tuple(self.weight.shape)}, eps={self.eps}"
        except AttributeError:
            return f"weight=<traced>, eps={self.eps}"
    transformers.models.gemma.modeling_gemma.GemmaRMSNorm.extra_repr = safe_extra_repr

    # Load Config and Model
    config = _config.get_config(CONFIG_NAME)
    
    # Patch sample_time to return float32
    original_sample_time = openpi.models_pytorch.pi0_pytorch.PI0Pytorch.sample_time
    def sample_time_patched(self, bsize, dev):
        t = original_sample_time(self, bsize, dev)
        return t.to(dtype=torch.float32)
    openpi.models_pytorch.pi0_pytorch.PI0Pytorch.sample_time = sample_time_patched

    # Patch embed_suffix to cast to float32
    original_embed_suffix = openpi.models_pytorch.pi0_pytorch.PI0Pytorch.embed_suffix
    def embed_suffix_patched(self, state, noisy_actions, timestep):
        timestep = timestep.to(dtype=torch.float32)
        return original_embed_suffix(self, state, noisy_actions, timestep)
    openpi.models_pytorch.pi0_pytorch.PI0Pytorch.embed_suffix = embed_suffix_patched

    # Load Policy on CPU
    device = "cpu"
    print(f"Loading policy from {CHECKPOINT_DIR} on {device}...")
    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR, pytorch_device=device)
    model = policy._model
    
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        
    model.to(torch.float32)
    model.eval()
    
    wrapper = OnnxWrapper(model)
    wrapper.eval()
    
    # Dummy Inputs for FP32 trace
    B = 1
    dummy_inputs = (
        torch.randn(B, 3, 224, 224, dtype=torch.float32, device=device), 
        torch.randn(B, 3, 224, 224, dtype=torch.float32, device=device), 
        torch.zeros(B, 3, 224, 224, dtype=torch.float32, device=device), 
        torch.randn(B, 32, dtype=torch.float32, device=device),          
        torch.randint(0, 100, (B, config.model.max_token_len), dtype=torch.int32, device=device), 
        torch.ones(B, config.model.max_token_len, dtype=torch.bool, device=device),  
        torch.randn(B, config.model.action_horizon, config.model.action_dim, dtype=torch.float32, device=device) 
    )
    
    input_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", 
                   "state", "tokenized_prompt", "tokenized_prompt_mask", "noise"]
    output_names = ["actions"]
    
    # 2. Export FP32 Model
    print(f"Exporting intermediate FP32 model to {TEMP_FP32_PATH}...")
    if not os.path.exists(os.path.dirname(TEMP_FP32_PATH)):
        os.makedirs(os.path.dirname(TEMP_FP32_PATH))
        
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        TEMP_FP32_PATH,
        opset_version=18,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={k: {0: "batch_size"} for k in input_names + output_names},
        dynamo=False
    )
    print("FP32 Export done.")
    
    # 3. Convert to FP16
    print("Converting to FP16...")
    model_fp32 = onnx.load(TEMP_FP32_PATH)
    print(f"FP32 Model loaded. Nodes: {len(model_fp32.graph.node)}")
    
    # keep_io_types=False means inputs/outputs will also be float16
    try:
        model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=False)
        if len(model_fp16.graph.node) == 0:
             raise ValueError("Converted model has 0 nodes")
        
        # Restore Opset if lost
        if len(model_fp16.opset_import) == 0:
            model_fp16.opset_import.extend(model_fp32.opset_import)
            
        final_model = model_fp16
        print("FP16 Conversion successful.")
        
        # Patch CumSum on FP16 model
        final_model = patch_cumsum_nodes(final_model)
        
    except Exception as e:
        print(f"FP16 Conversion failed: {e}")
        print("Falling back to FP32 model (patching CumSum on FP32)...")
        final_model = patch_cumsum_nodes(model_fp32)
        target_output_path = OUTPUT_ONNX_PATH.replace(".fp16.onnx", ".fp32.onnx")

    # 5. Save Final Model
    print(f"Saving final model to {target_output_path}...")
    if os.path.exists(target_output_path):
        os.remove(target_output_path)
    data_path = os.path.basename(target_output_path) + ".data"
    data_full_path = os.path.join(os.path.dirname(target_output_path), data_path)
    if os.path.exists(data_full_path):
        os.remove(data_full_path)
        
    onnx.save_model(
        final_model,
        target_output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
        size_threshold=1024,
        convert_attribute=False
    )
    
    # Cleanup temp
    if os.path.exists(TEMP_FP32_PATH):
        os.remove(TEMP_FP32_PATH)
        
    print(f"Success! Model saved to {target_output_path}")


if __name__ == "__main__":
    main()
