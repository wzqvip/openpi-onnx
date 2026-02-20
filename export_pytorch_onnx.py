#!/usr/bin/env python3
"""
Export FP32 ONNX directly from PyTorch model using torch.onnx.export
"""

import sys
import torch
import numpy as np
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
    
    # Create dummy inputs
    logger.info("Creating dummy inputs...")
    batch_size = 1
    
    dummy_inputs = (
        [torch.randn(batch_size, 3, 224, 224, dtype=torch.float32) for _ in range(3)],  # 3 images
        [torch.ones(batch_size, 256, dtype=torch.bool) for _ in range(3)],  # img masks
        torch.randint(0, 256000, (batch_size, 200), dtype=torch.int32),  # lang tokens
        torch.ones(batch_size, 200, dtype=torch.bool),  # lang mask
        torch.randint(0, 256000, (batch_size, 200), dtype=torch.int32),  # traj tokens
        torch.tensor([10], dtype=torch.int64),  # action horizon
    )
    
    # Export to ONNX using trace-based export (simpler)
    output_path = 'checkpoints/pi05_libero_onnx_compat/model.fp32.pytorch.jax.onnx'
    logger.info(f"Exporting to ONNX: {output_path}")
    
    # Use simple tuple-based export with traced execution
    with torch.no_grad():
        # Create a wrapped forward that accepts a single tuple
        class ExportWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, 
                       images_0, images_1, images_2,
                       img_masks_0, img_masks_1, img_masks_2,
                       tokenized_prompt, tokenized_prompt_mask,
                       tokenized_traj, action_horizon):
                return self.model(
                    [images_0, images_1, images_2],
                    [img_masks_0, img_masks_1, img_masks_2],
                    tokenized_prompt, tokenized_prompt_mask,
                    tokenized_traj, action_horizon
                )
        
        wrapper = ExportWrapper(model)
        
        # Extract individual inputs for export
        imgs = dummy_inputs[0]
        masks = dummy_inputs[1]
        
        torch.onnx.export(
            wrapper,
            (imgs[0], imgs[1], imgs[2],
             masks[0], masks[1], masks[2],
             dummy_inputs[2], dummy_inputs[3],
             dummy_inputs[4], dummy_inputs[5]),
            f=output_path,
            input_names=[
                'observation.images.base_0_rgb',
                'observation.images.left_wrist_0_rgb', 
                'observation.images.right_wrist_0_rgb',
                'observation.img_masks_0',
                'observation.img_masks_1',
                'observation.img_masks_2',
                'observation.tokenized_prompt',
                'observation.tokenized_prompt_mask',
                'observation.tokenized_traj',
                'action_horizon',
            ],
            output_names=['actions'],
            opset_version=18,
            do_constant_folding=True,
            verbose=False,
        )
    
    # Check output
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"✓ Export successful!")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Size: {file_size:.1f} MB")

if __name__ == "__main__":
    main()
