
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
CHECKPOINT_DIR = "./checkpoints/pi05_libero_pytorch_new"
CONFIG_NAME = "pi05_libero"
OUTPUT_DIR = "./dist/final_w4a4"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model.w4a4.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Monkey Patching (Same as before) ---
original_apply_rotary_pos_emb = modeling_gemma.apply_rotary_pos_emb
def apply_rotary_pos_emb_patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (modeling_gemma.rotate_half(q) * sin)
    k_embed = (k * cos) + (modeling_gemma.rotate_half(k) * sin)
    return q_embed, k_embed
modeling_gemma.apply_rotary_pos_emb = apply_rotary_pos_emb_patched

# Patch get_safe_dtype
original_get_safe_dtype = pi0_pytorch.get_safe_dtype
def get_safe_dtype_patched(target_dtype, device_type):
    if not torch.cuda.is_available():
        return torch.float32
    return torch.float16
pi0_pytorch.get_safe_dtype = get_safe_dtype_patched

# Patch GemmaRMSNorm.extra_repr
original_extra_repr = modeling_gemma.GemmaRMSNorm.extra_repr
def extra_repr_patched(self):
    return f"eps={self.eps}"
modeling_gemma.GemmaRMSNorm.extra_repr = extra_repr_patched

# --- Input/Output definitions ---
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
    torch.compile = lambda x, **k: x
    config = _config.get_config(CONFIG_NAME)
    
    # HACK: Disable tiny model patch
    import openpi.models.gemma as _gemma_mod
    original_get_config = _gemma_mod.get_config
    
    # Patch sample time
    original_sample_time = pi0_pytorch.PI0Pytorch.sample_time
    def sample_time_patched(self, bsize, device):
        return original_sample_time(self, bsize, device)
    pi0_pytorch.PI0Pytorch.sample_time = sample_time_patched

    device = "cpu"
    print(f"Loading policy from {CHECKPOINT_DIR} on {device}...")
    exec_dtype = torch.float32

    policy = policy_config.create_trained_policy(config, CHECKPOINT_DIR, pytorch_device=device)
    model = policy._model
    model.eval()
    model.to(dtype=exec_dtype)
    
    # Patch sample time again to match exec_dtype
    def sample_time_patched_2(self, bsize, device):
        t = original_sample_time(self, bsize, device)
        return t.to(dtype=exec_dtype)
    pi0_pytorch.PI0Pytorch.sample_time = sample_time_patched_2
    
    original_embed_suffix = pi0_pytorch.PI0Pytorch.embed_suffix
    def embed_suffix_match(self, state, noisy_actions, timestep):
        timestep = timestep.to(dtype=exec_dtype)
        return original_embed_suffix(self, state, noisy_actions, timestep)
    pi0_pytorch.PI0Pytorch.embed_suffix = embed_suffix_match
    
    # --- W4A4 Quantization Config ---
    print("Configuring W4A4 (INT4 Weights + INT4 Activations)...")
    
    # Start with INT8 Default and modify
    quant_config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    
    # Global Config Modification
    # Weights: 4-bit
    # Inputs (Activations): 4-bit
    new_cfg = {
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": 4, 
                "block_sizes": {-1: 128}, # Blockwise supported? Or per-channel?
                "enable": True
            },
            "*input_quantizer": {
                "num_bits": 4,
                "enable": True
            },
            # Common filter
             "*output_quantizer": {"enable": False}
        },
        "algorithm": "max" # Simple calibration
    }
    
    # Update config
    quant_config = new_cfg
    
    def filter_func(name):
        return any(x in name for x in ["time_emb", "pos_embed", "embed_tokens", "patch_embed", "norm"])
    
    print("Applying W4A4 quantization...")
    # NOTE: calibration step is needed for activations!
    # We will use dummy data for "max" calibration since we don't have a dataset loader ready here.
    
    batch_size = 1
    dummy_inputs_tuple = (
        torch.randn(batch_size, 3, 224, 224, dtype=exec_dtype, device=device),
        torch.randn(batch_size, 3, 224, 224, dtype=exec_dtype, device=device),
        torch.zeros(batch_size, 3, 224, 224, dtype=exec_dtype, device=device),
        torch.randn(batch_size, 32, dtype=exec_dtype, device=device),
        torch.randint(0, 100, (batch_size, config.model.max_token_len), dtype=torch.int32, device=device),
        torch.ones(batch_size, config.model.max_token_len, dtype=torch.bool, device=device),
        torch.randn(batch_size, config.model.action_horizon, config.model.action_dim, dtype=exec_dtype, device=device)
    )

    def calibration_loop(model):
        model(*dummy_inputs_tuple)
        
    wrapper = OnnxWrapper(model)
    
    # Quantize with calibration
    model_quantized = mtq.quantize(wrapper, quant_config, forward_loop=calibration_loop)
    mtq.disable_quantizer(model_quantized, filter_func)

    # --- Export ---
    print(f"Exporting W4A4 model to {OUTPUT_PATH}...")
    temp_path = OUTPUT_PATH + ".temp.onnx"
    
    torch.onnx.export(
        model_quantized,
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
    
    # --- Post-Processing (CumSum + External Data) ---
    print("Patching CumSum and Saving Final...")
    import onnx
    from onnx import helper, TensorProto
    
    model_export = onnx.load(temp_path)
    new_nodes = []
    patched = 0
    for node in model_export.graph.node:
        if node.op_type == "CumSum":
            input_name = node.input[0]
            cast_out = input_name + "_cast_int32"
            cast_node = helper.make_node("Cast", inputs=[input_name], outputs=[cast_out], to=TensorProto.INT32, name=node.name + "_cast_patch")
            node.input[0] = cast_out
            new_nodes.append(cast_node)
            new_nodes.append(node)
            patched += 1
        else:
            new_nodes.append(node)
            
    if patched > 0:
        model_export.graph.ClearField("node")
        model_export.graph.node.extend(new_nodes)
        
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    data_path = os.path.basename(OUTPUT_PATH) + ".data"
    full_data_path = os.path.join(os.path.dirname(OUTPUT_PATH), data_path)
    if os.path.exists(full_data_path):
        os.remove(full_data_path)
        
    onnx.save_model(
        model_export,
        OUTPUT_PATH,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
        size_threshold=1024,
        convert_attribute=False
    )
    print(f"Success! W4A4 saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
