#!/usr/bin/env python3
"""
Export INT4 (4-bit weight only) model - CLEAN version without problematic post-processing.
Uses INT4_BLOCKWISE_WEIGHT_ONLY_CFG for faster quantization.

This version REMOVES the block_size attribute removal step that was causing protobuf corruption.
TensorRT can handle models with block_size attributes - the corruption was the real issue.
"""

import sys
from unittest.mock import MagicMock
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

# PATCH: Disable Runtime Type Checking
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
import pathlib
import torch
import onnx
import modelopt.torch.quantization as mtq
from openpi.training import config as _config
from openpi.models import model as _model
import transformers
from transformers.models.gemma import modeling_gemma
from openpi.models_pytorch import pi0_pytorch
import numpy as np
from tqdm import tqdm
from onnx import TensorProto, helper

# PATCH: JAX handling
os.environ["JAX_PLATFORM_NAME"] = "cpu"
def mock_custom_jvp(fun, **kwargs):
    def wrapper(*args, **kwargs):
        return fun(*args, **kwargs)
    def defjvp(jvp_fun):
        return jvp_fun
    wrapper.defjvp = defjvp
    return wrapper

try:
    import jax
    jax.jit = noop_decorator
    jax.custom_jvp = mock_custom_jvp
except ImportError:
    pass

# Patch ONNX helper
if not hasattr(onnx.helper, 'float32_to_bfloat16'):
    onnx.helper.float32_to_bfloat16 = lambda x: x 

import onnx_graphsurgeon as gs

# Configuration
CHECKPOINT_DIR = "/home/taco/checkpoints/pi05_libero_onnx_compat"
CONFIG_NAME = "pi05_libero"
OUTPUT_PATH = "/home/taco/checkpoints/pi05_libero_onnx_compat/model.int4_clean.modelopt.onnx"
CALIBRATION_FILE = "calibration_data.pt"

# Monkey Patching
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

print("Starting INT4 Export (Clean Version - No Block Size Removal)...")

def main():
    # Load config
    print(f"Loading config: {CONFIG_NAME}")
    config = _config.get_config(CONFIG_NAME)
    
    import dataclasses
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, action_dim=32))
    
    # Load model
    print("Loading PyTorch model...")
    model = pi0_pytorch.PI0Pytorch(config.model)
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    from safetensors.torch import load_file
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(dtype=torch.float32, device="cuda")
    
    print(f"✅ Model loaded successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Load calibration data
    print(f"\nLoading calibration data from: {CALIBRATION_FILE}")
    calib_data = torch.load(CALIBRATION_FILE, weights_only=False)
    
    if isinstance(calib_data, dict):
        calib_data = [calib_data]
    elif isinstance(calib_data, tuple):
        calib_data = list(calib_data)
    
    print(f"✅ Loaded {len(calib_data)} calibration samples")
    
    # Prepare calibration function
    def calibrate_fn():
        print("Running calibration...")
        for i, data in enumerate(tqdm(calib_data[:1], desc="Calibration")):
            # Handle different calibration data formats
            if isinstance(data, tuple):
                data = data[0] if len(data) > 0 else {}
            
            if isinstance(data, np.ndarray):
                print(f"Warning: calibration data is numpy array, skipping")
                continue
            
            if not isinstance(data, dict):
                print(f"Warning: calibration data type {type(data)} not supported, skipping")
                continue
            
            # Move to GPU
            obs = {}
            for k, v in data.items():
                if isinstance(v, dict):
                    obs[k] = {kk: vv.cuda() if isinstance(vv, torch.Tensor) else vv for kk, vv in v.items()}
                elif isinstance(v, torch.Tensor):
                    obs[k] = v.cuda()
                else:
                    obs[k] = v
            
            with torch.no_grad():
                try:
                    _ = model(obs)
                except Exception as e:
                    print(f"Warning: calibration sample {i} failed: {e}")
                    continue
    
    # Apply INT4 quantization (blockwise, no AWQ search)
    print("\nApplying INT4 quantization (blockwise weight only)...")
    quant_cfg = mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG
    
    print("Quantization config:")
    print(f"  - Algorithm: INT4 Blockwise Weight Only")
    print(f"  - No AWQ search (fast)")
    print(f"  - Expected time: 2-5 minutes")
    
    model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_fn)
    print("✅ Quantization complete")
    
    # Count quantizers
    quantizers = [name for name, _ in model.named_modules() if 'quantizer' in name.lower()]
    print(f"✅ Inserted {len(quantizers)} quantizers")
    
    # Prepare dummy input
    print("\nPreparing ONNX export inputs...")
    dummy_image = torch.randn(1, 3, 224, 224).cuda()
    dummy_state = torch.randn(1, 8).cuda()
    dummy_prompt = torch.randint(0, 10000, (1, 200), dtype=torch.int32).cuda()
    dummy_prompt_mask = torch.ones(1, 200, dtype=torch.bool).cuda()
    dummy_noise = torch.randn(1, 10, 32).cuda()
    
    dummy_input = {
        "image": {
            "base_0_rgb": dummy_image,
            "left_wrist_0_rgb": dummy_image,
            "right_wrist_0_rgb": dummy_image,
        },
        "state": dummy_state,
        "prompt": dummy_prompt,
        "prompt_mask": dummy_prompt_mask,
        "noise": dummy_noise,
    }
    
    # Export to ONNX
    print(f"\nExporting to ONNX: {OUTPUT_PATH}")
    print("This may take 30-40 minutes...")
    
    intermediate_path = OUTPUT_PATH.replace(".onnx", "_intermediate.onnx")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            intermediate_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=[
                "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb",
                "state", "prompt", "prompt_mask", "noise"
            ],
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
    
    print("✅ Initial ONNX export complete")
    
    # GraphSurgeon cleanup
    print("\nRunning GraphSurgeon cleanup...")
    graph = gs.import_onnx(onnx.load(intermediate_path))
    graph.cleanup().toposort()
    model_proto = gs.export_onnx(graph)
    print("✅ GraphSurgeon cleanup complete")
    
    # Patch CumSum nodes
    print("\nPatching CumSum nodes for TensorRT compatibility...")
    patched_count = 0
    new_nodes = []
    
    for node in model_proto.graph.node:
        if node.op_type == "CumSum":
            original_output_name = node.output[0]
            cumsum_out_name = node.name + "_cumsum_int32_output"
            cast_in_name = node.name + "_cast_in_output"
            
            cast_in_node = helper.make_node(
                "Cast",
                inputs=node.input[:1],
                outputs=[cast_in_name],
                to=TensorProto.INT32,
                name=node.name + "_cast_in_patch"
            )
            
            node.input[0] = cast_in_name
            node.output[0] = cumsum_out_name
            
            cast_out_node = helper.make_node(
                "Cast",
                inputs=[cumsum_out_name],
                outputs=[original_output_name],
                to=TensorProto.INT64,
                name=node.name + "_cast_out_patch"
            )
            
            new_nodes.append(cast_in_node)
            new_nodes.append(node)
            new_nodes.append(cast_out_node)
            patched_count += 1
        else:
            new_nodes.append(node)
    
    if patched_count > 0:
        model_proto.graph.ClearField("node")
        model_proto.graph.node.extend(new_nodes)
        print(f"✅ Patched {patched_count} CumSum nodes")
    
    # FINAL SAVE - NO FURTHER POST-PROCESSING
    print(f"\n💾 Saving final ONNX model...")
    onnx.save(model_proto, OUTPUT_PATH)
    os.remove(intermediate_path)
    
    # Verify the saved file
    print(f"\n🔍 Verifying saved ONNX file...")
    try:
        test_model = onnx.load(OUTPUT_PATH)
        print(f"✅ ONNX file verification passed!")
        print(f"   - IR version: {test_model.ir_version}")
        print(f"   - Opset: {test_model.opset_import[0].version}")
        print(f"   - Producer: {test_model.producer_name} {test_model.producer_version}")
    except Exception as e:
        print(f"❌ ONNX file verification FAILED: {e}")
        return
    
    # Check file size
    file_size_gb = os.path.getsize(OUTPUT_PATH) / (1024**3)
    
    print(f"\n{'='*80}")
    print(f"✅ INT4 CLEAN EXPORT COMPLETE!")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Size: {file_size_gb:.2f} GB")
    print(f"Expected Compression: 3-4x (vs FP32)")
    print(f"Expected Accuracy: 90-97% (to be verified with LIBERO evaluation)")
    print(f"\n⚠️  NOTE: This version keeps block_size attributes for TensorRT.")
    print(f"   Previous corruption was caused by post-processing, not the attributes.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
