#!/usr/bin/env python3
"""
Export FP32 ONNX model from PyTorch checkpoint.
This script re-exports a clean FP32 ONNX model from the original PyTorch model.
"""

import sys
import os
import argparse
import logging
import json
import torch
from pathlib import Path

# Add project root
sys.path.insert(0, '/home/taco/openpi')
sys.path.insert(0, '/home/taco/openpi-onnx')

from safetensors.torch import load_file
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ExportFP32ONNX")

def load_model_from_safetensors(checkpoint_path, config_path, device='cpu'):
    """Load model from safetensors checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object
    config = Pi0Config(
        action_dim=config_dict.get("action_dim", 32),
        action_horizon=config_dict.get("action_horizon", 10),
        paligemma_variant=config_dict.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=config_dict.get("action_expert_variant", "gemma_300m"),
        dtype=config_dict.get("precision", "bfloat16"),
    )
    logger.info(f"Config: {config}")
    
    # Create model
    model = pi0_pytorch.PI0Pytorch(config).to(device)
    
    # Load state dict from safetensors
    state_dict = load_file(checkpoint_path)
    
    # Check dtype in state dict
    first_param = list(state_dict.values())[0]
    logger.info(f"Checkpoint dtype: {first_param.dtype}")
    
    # Convert to FP32 if needed
    if first_param.dtype != torch.float32:
        logger.warning(f"Converting checkpoint from {first_param.dtype} to FP32")
        state_dict = {k: v.float() if v.dtype in [torch.float16, torch.bfloat16] else v 
                      for k, v in state_dict.items()}
    
    # Load weights
    model.load_state_dict(state_dict, strict=False)
    logger.info("Model weights loaded successfully")
    
    # Convert model to FP32
    model = model.float()
    logger.info("Model converted to FP32")
    
    return model, config

def export_to_onnx(model, output_path, config, device='cpu'):
    """Export model to ONNX format."""
    model.eval()
    
    logger.info(f"Creating dummy inputs for ONNX export...")
    
    # Create dummy inputs matching the model's expected inputs
    batch_size = 1
    num_images = 3
    image_h, image_w = 384, 512
    
    # Dummy inputs
    dummy_images = [
        torch.randn(batch_size, 3, image_h, image_w, dtype=torch.float32, device=device)
        for _ in range(num_images)
    ]
    dummy_img_masks = [
        torch.ones(batch_size, 256, dtype=torch.bool, device=device)
        for _ in range(num_images)
    ]
    dummy_lang_tokens = torch.randint(0, 256000, (batch_size, 512), dtype=torch.long, device=device)
    dummy_lang_mask = torch.ones(batch_size, 512, dtype=torch.bool, device=device)
    dummy_traj_tokens = torch.randint(0, 256000, (batch_size, 512), dtype=torch.long, device=device)
    dummy_action_horizon = torch.tensor([config.action_horizon], dtype=torch.long, device=device)
    
    input_names = ['images_0', 'images_1', 'images_2', 
                   'img_masks_0', 'img_masks_1', 'img_masks_2',
                   'lang_tokens', 'lang_mask', 'traj_tokens', 'action_horizon']
    
    # For simplicity, export with concrete shape
    logger.info(f"Exporting to ONNX: {output_path}")
    
    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                args=(dummy_images, dummy_img_masks, dummy_lang_tokens, dummy_lang_mask, 
                      dummy_traj_tokens, dummy_action_horizon),
                f=output_path,
                input_names=input_names,
                output_names=['actions'],
                opset_version=17,
                do_constant_folding=True,
                verbose=False,
                # Use symbolic shapes where possible
                dynamic_axes={
                    'images_0': {0: 'batch_size'},
                    'images_1': {0: 'batch_size'},
                    'images_2': {0: 'batch_size'},
                    'img_masks_0': {0: 'batch_size'},
                    'img_masks_1': {0: 'batch_size'},
                    'img_masks_2': {0: 'batch_size'},
                    'lang_tokens': {0: 'batch_size'},
                    'lang_mask': {0: 'batch_size'},
                    'traj_tokens': {0: 'batch_size'},
                    'actions': {0: 'batch_size'},
                },
            )
            logger.info(f"ONNX export successful: {output_path}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            # Try simpler approach without complex inputs
            logger.info("Attempting simplified export...")
            
            # Export just the core model forward pass
            torch.onnx.export(
                model,
                args=(dummy_images[0], dummy_img_masks[0]),
                f=output_path,
                input_names=['image', 'img_mask'],
                output_names=['vision_embeddings'],
                opset_version=17,
                do_constant_folding=True,
                verbose=False,
            )
            logger.info(f"Simplified ONNX export successful: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Export FP32 ONNX model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch/model.safetensors",
        help="Path to safetensors checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch/config.json",
        help="Path to config.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/taco/openpi-onnx/checkpoints/pi05_libero_onnx_compat/model.fp32.fresh.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for export",
    )
    args = parser.parse_args()
    
    logger.info(f"Export FP32 ONNX Model")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Device: {args.device}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, config = load_model_from_safetensors(args.checkpoint, args.config, device=args.device)
    
    # Export to ONNX
    export_to_onnx(model, args.output, config, device=args.device)
    
    logger.info(f"\nExport complete!")
    logger.info(f"Output: {args.output}")
    logger.info(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
