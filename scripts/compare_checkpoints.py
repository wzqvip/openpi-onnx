
import torch
import numpy as np
import os
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models import model as _model

# Config
OLD_CHECKPOINT = "./checkpoints/pi05_libero_pytorch"
NEW_CHECKPOINT = "./checkpoints/pi05_libero_pytorch_new"
CONFIG_NAME = "pi05_libero"

def main():
    # DISABLE COMPILE
    torch.compile = lambda x, **k: x

    print(f"Comparing {OLD_CHECKPOINT} vs {NEW_CHECKPOINT}")
    config = _config.get_config(CONFIG_NAME)
    
    # Patch safe dtype
    from openpi.models_pytorch import pi0_pytorch
    pi0_pytorch.get_safe_dtype = lambda target, device: torch.float32

    # Load Old
    print("Loading Old...")
    policy_old = policy_config.create_trained_policy(config, OLD_CHECKPOINT, pytorch_device="cpu")
    model_old = policy_old._model.eval()
    model_old.to(dtype=torch.float32) # FORCE FLOAT32
    
    # Load New
    print("Loading New...")
    policy_new = policy_config.create_trained_policy(config, NEW_CHECKPOINT, pytorch_device="cpu")
    model_new = policy_new._model.eval()
    model_new.to(dtype=torch.float32) # FORCE FLOAT32
    
    # 1. Compare Weights (Random Sample)
    print("\nWeight Comparison:")
    old_state = model_old.state_dict()
    new_state = model_new.state_dict()
    
    diff_keys = []
    max_diff = 0.0
    for key in old_state:
        if key not in new_state:
            # print(f"Missing key in new: {key}")
            continue
        t1 = old_state[key]
        t2 = new_state[key]
        if t1.shape != t2.shape:
             print(f"Shape mismatch {key}: {t1.shape} vs {t2.shape}")
             continue
        diff = (t1 - t2).abs().max().item()
        if diff > 1e-3:
            diff_keys.append((key, diff))
            max_diff = max(max_diff, diff)
            
    print(f"Total keys with diff > 1e-3: {len(diff_keys)}")
    if diff_keys:
        print(f"Top 5 Diff Keys: {sorted(diff_keys, key=lambda x: x[1], reverse=True)[:5]}")
    else:
        print("Weights Match!")

    # 2. Inference Comparison
    print("\nInference Comparison:")
    torch.manual_seed(42)
    inputs = {
        "base_0_rgb": torch.randn(1, 3, 224, 224),
        "left_wrist_0_rgb": torch.randn(1, 3, 224, 224),
        "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224),
        "state": torch.randn(1, 32),
        "tokenized_prompt": torch.randint(0, 100, (1, 12)).int(),
        "tokenized_prompt_mask": torch.ones(1, 12).bool(),
        "noise": torch.randn(1, 10, 32)
    }
    
    # Helper
    def run_model(model):
        images = {k: inputs[k] for k in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]}
        image_masks = {k: torch.ones(v.shape[:-3], dtype=torch.bool) for k, v in images.items()}
        obs = _model.Observation(
             images=images, image_masks=image_masks, state=inputs["state"],
             tokenized_prompt=inputs["tokenized_prompt"], tokenized_prompt_mask=inputs["tokenized_prompt_mask"]
        )
        with torch.no_grad():
             return model.sample_actions(device="cpu", observation=obs, noise=inputs["noise"], num_steps=10)

    out_old = run_model(model_old)
    out_new = run_model(model_new)
    
    inf_diff = (out_old - out_new).abs().max().item()
    inf_mse = ((out_old - out_new)**2).mean().item()
    print(f"Inference Max Diff: {inf_diff}")
    print(f"Inference MSE: {inf_mse}")

if __name__ == "__main__":
    main()
