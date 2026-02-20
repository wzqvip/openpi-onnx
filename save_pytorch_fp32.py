#!/usr/bin/env python3
"""
直接使用PyTorch模型而不导出，并通过TensorRT动态编译
或者使用一个存在的working ONNX进行对比
"""

import sys
import torch
import json
import logging
from pathlib import Path

sys.path.insert(0, '/home/taco/openpi')

from safetensors.torch import load_file
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CheckPyTorchFP32")

def main():
    # Load config
    config_path = 'checkpoints/pi05_libero_pytorch/config.json'
    with open(config_path) as f:
        config_dict = json.load(f)
    
    logger.info(f"Config: {config_dict}")
    
    # Create config object
    config = Pi0Config(
        action_dim=config_dict.get("action_dim", 32),
        action_horizon=config_dict.get("action_horizon", 10),
        paligemma_variant=config_dict.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=config_dict.get("action_expert_variant", "gemma_300m"),
        dtype=config_dict.get("precision", "bfloat16"),
    )
    
    # Create and load model
    logger.info("Creating PyTorch model...")
    model = pi0_pytorch.PI0Pytorch(config)
    
    # Load weights from safetensors
    logger.info("Loading model weights from safetensors...")
    state_dict = load_file('checkpoints/pi05_libero_pytorch/model.safetensors')
    
    # Check dtype before conversion
    first_param = list(state_dict.values())[0]
    logger.info(f"Checkpoint dtype: {first_param.dtype}")
    
    # Convert to FP32
    logger.info("Converting model to FP32...")
    state_dict = {k: v.float() if v.dtype in [torch.float16, torch.bfloat16] else v 
                  for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.float()
    model.eval()
    
    logger.info("✓ Model loaded and converted to FP32 successfully!")
    
    # Save as a pure PyTorch checkpoint for reference
    output_pt = 'checkpoints/pi05_libero_pytorch/model_fp32.pt'
    logger.info(f"Saving PyTorch FP32 checkpoint: {output_pt}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_dict,
    }, output_pt)
    
    file_size = Path(output_pt).stat().st_size / (1024 * 1024)
    logger.info(f"✓ PyTorch checkpoint saved!")
    logger.info(f"  Output: {output_pt}")
    logger.info(f"  Size: {file_size:.1f} MB")
    
    # List all model parameters and their dtypes
    logger.info("\nModel parameter dtypes:")
    dtype_counts = {}
    for name, param in model.named_parameters():
        dt = str(param.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
    
    for dtype, count in dtype_counts.items():
        logger.info(f"  {dtype}: {count} parameters")

if __name__ == "__main__":
    main()
