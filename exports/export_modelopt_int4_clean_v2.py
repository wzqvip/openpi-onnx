#!/usr/bin/env python3
"""
Export INT4 model following INT8 successful pattern
Uses INT4_BLOCKWISE_WEIGHT_ONLY_CFG (no AWQ search)
"""

import sys
from unittest.mock import MagicMock
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

def noop_decorator(*args, **kwargs):
    if len(args) >= 1 and callable(args[0]):
        return args[0]
    def wrapper(target):
        return target
    return wrapper

sys.modules["typeguard"] = MagicMock()
sys.modules["typeguard"].typechecked = noop_decorator
sys.modules["jaxtyping"] = MagicMock()
sys.modules["jaxtyping"].jaxtyped = noop_decorator
sys.modules["jaxtyping._decorator"] = MagicMock()

import os
import torch
import onnx
import numpy as np
from tqdm import tqdm
import modelopt.torch.quantization as mtq
from openpi.training import config as _config
from openpi.models_pytorch import pi0_pytorch
import transformers
from transformers.models.gemma import modeling_gemma
from onnx import TensorProto, helper

# Monkey patches
def apply_rotary_pos_emb_patched(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (modeling_gemma.rotate_half(q) * sin)
    k_embed = (k * cos) + (modeling_gemma.rotate_half(k) * sin)
    return q_embed, k_embed

modeling_gemma.apply_rotary_pos_emb = apply_rotary_pos_emb_patched

def get_safe_dtype_patched(target_dtype, device_type):
    return target_dtype

pi0_pytorch.get_safe_dtype = get_safe_dtype_patched

# Config
CHECKPOINT_DIR = "/home/taco/checkpoints/pi05_libero_onnx_compat"
CONFIG_NAME = "pi05_libero"
OUTPUT_PATH = "/home/taco/checkpoints/pi05_libero_onnx_compat/model.int4_clean.modelopt.onnx"
CALIBRATION_FILE = "calibration_data.pt"

class OnnxWrapperINT4(torch.nn.Module):
    """Wrapper for ONNX export matching calibration data format"""
    def __init__(self, model, num_steps=10):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        
    def forward(self, base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, 
                state, prompt, prompt_mask, noise):
        """Forward matching calibration tuple format"""
        device = base_0_rgb.device
        bsize = base_0_rgb.shape[0]
        
        # Process images
        images = torch.stack([base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb], dim=1)
        images_proc = self.model.image_proc(images)
        state_proc = self.model.state_proc(state)
        
        # Get prefix embeddings
        prefix_embs, prefix_pad_masks, prefix_att_2d_masks = self.model.get_prefix_embs(
            images_proc, prompt, prompt_mask, state_proc
        )
        
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        
        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        
        # Unrolled denoising loop
        dt = -1.0 / self.num_steps
        dt_tensor = torch.tensor(dt, dtype=self.model.action_in_proj.weight.dtype, device=device)
        x_t = noise
        
        for i in range(self.num_steps):
            time = torch.tensor(1.0 + i * dt, dtype=self.model.action_in_proj.weight.dtype, device=device)
            expanded_time = time.expand(bsize)
            v_t = self.model.denoise_step(state_proc, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt_tensor * v_t
        
        return x_t

def calibrate_model(wrapper, calibration_data):
    """Run calibration with real samples"""
    print(f"\nRunning calibration with {len(calibration_data)} samples...")
    wrapper.eval()
    
    with torch.no_grad():
        for i, sample in enumerate(tqdm(calibration_data, desc="Calibrating")):
            try:
                # Convert numpy to tensor and move to CPU
                sample = tuple(
                    torch.from_numpy(t) if isinstance(t, np.ndarray) else (
                        t.to("cpu") if isinstance(t, torch.Tensor) else t
                    )
                    for t in sample
                )
                
                # Check for NaNs
                for t in sample:
                    if isinstance(t, torch.Tensor) and torch.isnan(t).any():
                        raise ValueError("NaN in input")
                
                _ = wrapper(*sample)
                
            except Exception as e:
                print(f"Warning: sample {i} failed: {e}")
                continue
    
    print("✅ Calibration complete")

def main():
    print("="*80)
    print("INT4 EXPORT (Clean Version)")
    print("="*80)
    
    torch.compile = lambda x, **k: x
    
    # Load config
    print(f"\nLoading config: {CONFIG_NAME}")
    config = _config.get_config(CONFIG_NAME)
    
    import dataclasses
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, action_dim=32))
    
    # Load model
    print("Loading model...")
    model = pi0_pytorch.PI0Pytorch(config.model)
    
    from safetensors.torch import load_file
    ckpt_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
    sd = load_file(ckpt_path)
    model.load_state_dict(sd, strict=False)
    
    model.eval()
    model.to(dtype=torch.float32)
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    
    # Wrap model
    print("\nWrapping model...")
    wrapper = OnnxWrapperINT4(model, num_steps=10)
    
    # Load calibration data
    if not os.path.exists(CALIBRATION_FILE):
        print(f"ERROR: {CALIBRATION_FILE} not found!")
        return
    
    print(f"Loading calibration data from {CALIBRATION_FILE}...")
    calibration_data = torch.load(CALIBRATION_FILE, weights_only=False)
    print(f"Loaded {len(calibration_data)} samples")
    
    # Apply INT4 quantization
    print("\nApplying INT4 quantization...")
    quant_cfg = mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG
    print("Config: INT4_BLOCKWISE_WEIGHT_ONLY (no AWQ search)")
    print("Using 1 sample for fast calibration...")
    
    mtq.quantize(wrapper, quant_cfg, forward_loop=lambda m: calibrate_model(m, calibration_data[:1]))
    print("✅ Quantization complete")
    
    # Export to ONNX
    print(f"\nExporting to ONNX: {OUTPUT_PATH}")
    print("This will take 30-40 minutes...")
    
    dummy_input = calibration_data[0]
    dummy_input = tuple(
        torch.from_numpy(t) if isinstance(t, np.ndarray) else (
            t.to("cpu") if isinstance(t, torch.Tensor) else t
        )
        for t in dummy_input
    )
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    intermediate_path = OUTPUT_PATH.replace(".onnx", "_intermediate.onnx")
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        intermediate_path,
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb",
                     "state", "prompt", "prompt_mask", "noise"],
        output_names=["actions"],
        dynamic_axes={
            "base_0_rgb": {0: "batch"},
            "left_wrist_0_rgb": {0: "batch"},
            "right_wrist_0_rgb": {0: "batch"},
            "state": {0: "batch"},
            "prompt": {0: "batch"},
            "prompt_mask": {0: "batch"},
            "noise": {0: "batch"},
            "actions": {0: "batch"},
        },
    )
    
    print("✅ ONNX export complete")
    
    # Patch CumSum nodes
    print("\nPatching CumSum nodes...")
    model_proto = onnx.load(intermediate_path)
    patched_count = 0
    new_nodes = []
    
    for node in model_proto.graph.node:
        if node.op_type == "CumSum":
            original_output = node.output[0]
            cumsum_out = node.name + "_cumsum_int32"
            cast_in = node.name + "_cast_in"
            
            cast_in_node = helper.make_node(
                "Cast", inputs=node.input[:1], outputs=[cast_in],
                to=TensorProto.INT32, name=node.name + "_cast_in_patch"
            )
            
            node.input[0] = cast_in
            node.output[0] = cumsum_out
            
            cast_out_node = helper.make_node(
                "Cast", inputs=[cumsum_out], outputs=[original_output],
                to=TensorProto.INT64, name=node.name + "_cast_out_patch"
            )
            
            new_nodes.extend([cast_in_node, node, cast_out_node])
            patched_count += 1
        else:
            new_nodes.append(node)
    
    if patched_count > 0:
        model_proto.graph.ClearField("node")
        model_proto.graph.node.extend(new_nodes)
        print(f"✅ Patched {patched_count} CumSum nodes")
    
    # FINAL SAVE - NO POST-PROCESSING
    print(f"\n💾 Saving final model...")
    onnx.save(model_proto, OUTPUT_PATH)
    os.remove(intermediate_path)
    
    # Verify
    print(f"\n🔍 Verifying...")
    try:
        test_model = onnx.load(OUTPUT_PATH)
        print(f"✅ Verification passed!")
        print(f"   IR: {test_model.ir_version}, Opset: {test_model.opset_import[0].version}")
    except Exception as e:
        print(f"❌ Verification FAILED: {e}")
        return
    
    file_size_gb = os.path.getsize(OUTPUT_PATH) / (1024**3)
    
    print(f"\n{'='*80}")
    print(f"✅ INT4 EXPORT COMPLETE!")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Size: {file_size_gb:.2f} GB")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
