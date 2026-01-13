
import torch
import torch.nn as nn
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models import model as _model
import os
import argparse
import onnx
from onnx import helper, TensorProto
import transformers.models.gemma.modeling_gemma
import openpi.models_pytorch.pi0_pytorch
import sys

# --- Configuration ---
CHECKPOINT_DIR = "./checkpoints/pi05_libero_pytorch"
CONFIG_NAME = "pi05_libero"
OUTPUT_ONNX_PATH = "./checkpoints/pi05_libero_pytorch/model.fp16.onnx"

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
    new_nodes = []
    patched_count = 0
    
    for node in graph.node:
        if node.op_type == "CumSum":
            input_name = node.input[0]
            # Insert Cast to Int32
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
    print("--- Starting Native FP16 Export on GPU ---")
    
    # Disable torch compile
    torch.compile = lambda x, **k: x

    # 1. Apply Patches
    transformers.models.gemma.modeling_gemma.apply_rotary_pos_emb = custom_apply_rotary_pos_emb
    
    # Force get_safe_dtype to return float16 for GPU execution
    openpi.models_pytorch.pi0_pytorch.get_safe_dtype = lambda target, device: torch.float16

    # Patch Extra Repr
    def safe_extra_repr(self):
        try:
            return f"{tuple(self.weight.shape)}, eps={self.eps}"
        except AttributeError:
            return f"weight=<traced>, eps={self.eps}"
    transformers.models.gemma.modeling_gemma.GemmaRMSNorm.extra_repr = safe_extra_repr

    # Load Config and Model
    config = _config.get_config(CONFIG_NAME)
    
    # Patch sample_time to return float16
    original_sample_time = openpi.models_pytorch.pi0_pytorch.PI0Pytorch.sample_time
    def sample_time_patched(self, bsize, dev):
        t = original_sample_time(self, bsize, dev)
        return t.to(dtype=torch.float16)
    openpi.models_pytorch.pi0_pytorch.PI0Pytorch.sample_time = sample_time_patched

    # Patch embed_suffix to cast to float16
    original_embed_suffix = openpi.models_pytorch.pi0_pytorch.PI0Pytorch.embed_suffix
    def embed_suffix_patched(self, state, noisy_actions, timestep):
        timestep = timestep.to(dtype=torch.float16)
        return original_embed_suffix(self, state, noisy_actions, timestep)
    openpi.models_pytorch.pi0_pytorch.PI0Pytorch.embed_suffix = embed_suffix_patched

    # Load Policy on GPU
    device = "cuda"
    print(f"Loading policy from {CHECKPOINT_DIR} on {device}...")
    try:
        policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR, pytorch_device=device)
    except Exception as e:
        print(f"Failed to load on CUDA: {e}")
        return

    model = policy._model
    
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        
    model.to(dtype=torch.float16, device=device)
    model.eval()
    
    wrapper = OnnxWrapper(model)
    
    # Dummy Inputs for FP16 trace
    B = 1
    dummy_inputs = (
        torch.randn(B, 3, 224, 224, dtype=torch.float16, device=device), 
        torch.randn(B, 3, 224, 224, dtype=torch.float16, device=device), 
        torch.zeros(B, 3, 224, 224, dtype=torch.float16, device=device), 
        torch.randn(B, 32, dtype=torch.float16, device=device),          
        torch.randint(0, 100, (B, config.model.max_token_len), dtype=torch.int32, device=device), 
        torch.ones(B, config.model.max_token_len, dtype=torch.bool, device=device),  
        torch.randn(B, config.model.action_horizon, config.model.action_dim, dtype=torch.float16, device=device) 
    )
    
    input_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", 
                   "state", "tokenized_prompt", "tokenized_prompt_mask", "noise"]
    output_names = ["actions"]
    
    # 2. Export FP16 Model directly
    print(f"Exporting FP16 model to {OUTPUT_ONNX_PATH}...")
    if not os.path.exists(os.path.dirname(OUTPUT_ONNX_PATH)):
        os.makedirs(os.path.dirname(OUTPUT_ONNX_PATH))
    
    # Temporarily save to checking path first
    temp_path = OUTPUT_ONNX_PATH + ".temp"
    
    try:
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            temp_path,
            opset_version=18,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={k: {0: "batch_size"} for k in input_names + output_names}
        )
    except Exception as e:
        print(f"Export failed: {e}")
        return

    print("FP16 Export done. Patching CumSum...")
    
    # 3. Patch CumSum directly on the exported model
    model_onnx = onnx.load(temp_path)
    model_patched = patch_cumsum_nodes(model_onnx)
    
    # 5. Save Final Model
    print(f"Saving final W16A16 model to {OUTPUT_ONNX_PATH}...")
    if os.path.exists(OUTPUT_ONNX_PATH):
        os.remove(OUTPUT_ONNX_PATH)
    data_path = os.path.basename(OUTPUT_ONNX_PATH) + ".data"
    data_full_path = os.path.join(os.path.dirname(OUTPUT_ONNX_PATH), data_path)
    if os.path.exists(data_full_path):
        os.remove(data_full_path)
        
    onnx.save_model(
        model_patched,
        OUTPUT_ONNX_PATH,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
        size_threshold=1024,
        convert_attribute=False
    )
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    print("Success! Native W16A16 export complete.")
    print(f"Output: {OUTPUT_ONNX_PATH}")

if __name__ == "__main__":
    main()
