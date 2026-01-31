#!/usr/bin/env python3
"""
Export independent Vision Encoder (SigLIP) to ONNX for Split-Stack Architecture.
"""

import sys
import os
import torch
import onnx
from openpi.training import config as _config
from openpi.models import model as _model
from openpi.models_pytorch import pi0_pytorch
import numpy as np

# Mocks and Patches
from unittest.mock import MagicMock
sys.modules["lerobot"] = MagicMock()
sys.modules["lerobot.common"] = MagicMock()
sys.modules["lerobot.common.datasets"] = MagicMock()
sys.modules["lerobot.common.datasets.lerobot_dataset"] = MagicMock()

# Patch JAX
os.environ["JAX_PLATFORM_NAME"] = "cpu"
def noop(*args, **kwargs): return args[0] if len(args) > 0 and callable(args[0]) else MagicMock()
sys.modules["jax"] = MagicMock()
sys.modules["typeguard"] = MagicMock()
sys.modules["jaxtyping"] = MagicMock()

# Config
CHECKPOINT_DIR = "/home/taco/checkpoints/pi05_libero_onnx_compat"
CONFIG_NAME = "pi05_libero"
OUTPUT_PATH = "/home/taco/checkpoints/pi05_libero_onnx_compat/vision_encoder.onnx"

class VisionTowerWrapper(torch.nn.Module):
    def __init__(self, pi0_model):
        super().__init__()
        # Extract the PaliGemma model
        self.paligemma = pi0_model.paligemma_with_expert.paligemma
        # We want the vision tower + multi-modal projector
        # SigLIP vision tower
        self.vision_tower = self.paligemma.model.vision_tower
        # Projector to LLM dim
        self.mm_projector = self.paligemma.model.multi_modal_projector
    
    def forward(self, pixel_values):
        # 1. Vision Encoder (SigLIP)
        # Output: BaseModelOutputWithPooling
        vision_outputs = self.vision_tower(pixel_values)
        # Extract tensor
        image_features = vision_outputs.last_hidden_state
        
        # 2. Projector
        image_embeddings = self.mm_projector(image_features)
        
        # 3. Scaling (Standard in PaliGemma to match LLM variance)
        # Usually internal to embed_image/embed_tokens but let's check
        # embedding is usually: embeddings * sqrt(dim)
        # But for Images, PaliGemma usually just projects.
        
        return image_embeddings

def main():
    print(f"Loading config: {CONFIG_NAME}")
    import dataclasses
    config = _config.get_config(CONFIG_NAME)
    # PATCH: Override action_dim to match checkpoint
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, action_dim=32))
    
    print(f"Loading policy from {CHECKPOINT_DIR}...")
    model = pi0_pytorch.PI0Pytorch(config.model)
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, "model.safetensors")
    from safetensors.torch import load_file
    sd = load_file(ckpt_path)
    model.load_state_dict(sd, strict=False)
    model.eval()
    
    # Wrap Vision
    vision_model = VisionTowerWrapper(model)
    vision_model.eval()
    
    # Dummy Input
    # Shape: [Batch, Channels, Height, Width]
    # SigLIP 224x224
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    vision_model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    print(f"Exporting to {OUTPUT_PATH}...")
    torch.onnx.export(
        vision_model,
        dummy_input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["vision_embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "vision_embeddings": {0: "batch_size"}
        }
    )
    print("✅ Vision Encoder exported successfully.")

if __name__ == "__main__":
    main()
