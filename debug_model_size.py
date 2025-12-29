
import torch
from openpi.training import config as _config
from openpi.policies import policy_config
import os

checkpoint_dir = "/local/scratch1/wang.20306/openpi/checkpoints/pi05_libero_pytorch"
config_name = "pi05_libero"

def main():
    print(f"Loading config: {config_name}")
    config = _config.get_config(config_name)
    
    print(f"Loading policy from: {checkpoint_dir}")
    policy = policy_config.create_trained_policy(config, checkpoint_dir, pytorch_device="cpu")
    model = policy._model
    
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_count = sum(b.numel() for b in model.buffers())
    
    print(f"Total Parameters: {param_count:,}")
    print(f"Trainable Parameters: {trainable_count:,}")
    print(f"Buffer Parameters: {buffer_count:,}")
    
    # Check specific submodules
    if hasattr(model, 'paligemma'):
        print(f"PaliGemma params: {sum(p.numel() for p in model.paligemma.parameters()):,}")
    
    if hasattr(model, 'gemma_expert'):
        print(f"Gemma Expert params: {sum(p.numel() for p in model.gemma_expert.parameters()):,}")

if __name__ == "__main__":
    main()
