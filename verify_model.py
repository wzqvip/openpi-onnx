
import torch
import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models import model as _model
import os

# Define paths
checkpoint_dir = "/local/scratch1/wang.20306/openpi/checkpoints/pi05_libero_pytorch"
config_name = "pi05_libero"

def main():
    print(f"Loading config: {config_name}")
    config = _config.get_config(config_name)
    
    print(f"Loading policy from: {checkpoint_dir}")
    policy = policy_config.create_trained_policy(config, checkpoint_dir, pytorch_device="cpu")
    
    print("Policy loaded successfully.")
    model = policy._model
    print(f"Model class: {type(model)}")
    
    # Create dummy observation matching config specification
    batch_size = 1
    
    # Create images [B, H, W, C] - Observation doc says *b h w c
    # Note: transformers usually want [B, C, H, W] but preprocessing likely handles conversion.
    # Let's start with [B, H, W, 3] and float32 [-1, 1] per doc string in Observation.
    
    # Create images [B, C, H, W] for PyTorch transformers compatibility
    images = {
        "base_0_rgb": torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32),
        "left_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32),
        "right_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32),
    }
    
    image_masks = {
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool),
    }
    
    # State: [B, 32] (action_dim=32, state usually padded to action_dim in some transforms, 
    # but Libero data keys map random state(8) -> state.
    # Looking at config, it says action_dim=32. 
    # The model expects state [B, action_dim] in inputs_spec in pi0_config.py
    # So we provide 32-dim state.
    state = torch.zeros(batch_size, 32, dtype=torch.float32)
    
    # Tokenized prompt: [B, max_token_len]
    max_token_len = config.model.max_token_len
    tokenized_prompt = torch.zeros(batch_size, max_token_len, dtype=torch.int32)
    tokenized_prompt_mask = torch.ones(batch_size, max_token_len, dtype=torch.bool)
    
    obs = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask
    )
    
    print("Running inference with model.sample_actions...")
    try:
        # Generate dummy noise for deterministic run
        action_horizon = config.model.action_horizon
        action_dim = config.model.action_dim
        noise = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32)
        
        action = model.sample_actions(
            device="cpu",
            observation=obs,
            noise=noise,
            num_steps=5 # Reduce steps for verification
        )
        print("Inference successful.")
        print("Action shape:", action.shape)
        
    except Exception as e:
        print(f"Inference failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
