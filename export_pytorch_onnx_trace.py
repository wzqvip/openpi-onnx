#!/usr/bin/env python3
"""
Export FP32 ONNX using torch.jit.trace (simpler approach)
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
logger = logging.getLogger("ExportONNX")

def main():
    # Load config
    config_path = 'checkpoints/pi05_libero_pytorch_jax/config.json'
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
    state_dict = load_file('checkpoints/pi05_libero_pytorch_jax/model.safetensors')
    
    # Convert to FP32
    logger.info("Converting model to FP32...")
    state_dict = {k: v.float() if v.dtype in [torch.float16, torch.bfloat16] else v 
                  for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model = model.float()
    model.eval()
    
    logger.info("Model loaded successfully!")
    
    # Create dummy inputs matching model's forward signature
    logger.info("Creating dummy inputs...")
    batch_size = 1
    
    # Based on forward call signature
    dummy_images = [torch.randn(batch_size, 3, 224, 224, dtype=torch.float32) for _ in range(3)]
    dummy_img_masks = [torch.ones(batch_size, 256, dtype=torch.bool) for _ in range(3)]
    dummy_lang_tokens = torch.randint(0, 256000, (batch_size, 200), dtype=torch.int32)
    dummy_lang_mask = torch.ones(batch_size, 200, dtype=torch.bool)
    
    # Trace the model
    logger.info("Tracing model with torch.jit.trace...")
    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (dummy_images, dummy_img_masks, dummy_lang_tokens, dummy_lang_mask)
        )
    
    # Convert to ONNX
    output_path = 'checkpoints/pi05_libero_onnx_compat/model.fp32.pytorch.jax.onnx'
    logger.info(f"Converting traced model to ONNX: {output_path}")
    
    torch.onnx.export(
        traced,
        (dummy_images, dummy_img_masks, dummy_lang_tokens, dummy_lang_mask),
        f=output_path,
        input_names=[
            'images_0', 'images_1', 'images_2',
            'img_masks_0', 'img_masks_1', 'img_masks_2',
            'lang_tokens', 'lang_mask'
        ],
        output_names=['actions'],
        opset_version=18,
    )
    
    # Check output
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"✓ Export successful!")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Size: {file_size:.1f} MB")

if __name__ == "__main__":
    main()
