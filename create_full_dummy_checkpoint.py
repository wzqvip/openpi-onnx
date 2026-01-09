import torch
import os
import safetensors.torch
from openpi.training import config as _config
from openpi.models_pytorch import pi0_pytorch
import json

def main():
    config_name = "pi05_libero"
    output_path = "./checkpoints/pi05_libero_pytorch"
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Creating FULL dummy checkpoint for {config_name}...")
    
    # HACK: Disable torch.compile to avoid OOM or slow compilation
    torch.compile = lambda x, **k: x
    print("Disabled torch.compile.")

    config = _config.get_config(config_name)
    
    # NOTE: NOT patching for tiny model. Using full config.
    
    # Create Model
    print("Initializing model structure in float16 (Full Size)...")
    # Set default dtype to float16 to save memory during init
    torch.set_default_dtype(torch.float16)
    
    # Force cpu or meta device initially if possible, but here we init directly.
    # On 128GB Thor, this should fit easily.
    with torch.device("cpu"):
        model = pi0_pytorch.PI0Pytorch(config.model)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with random weights (float16). Param count: {num_params}")
    
    # Save as safetensors
    print(f"Saving to {output_path}/model.safetensors...")
    safetensors.torch.save_model(model, os.path.join(output_path, "model.safetensors"))
    
    # Save config.json
    config_dict = {
        "action_dim": config.model.action_dim,
        "action_horizon": config.model.action_horizon,
        "paligemma_variant": config.model.paligemma_variant,
        "action_expert_variant": config.model.action_expert_variant,
        "precision": "float32",
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
        
    print("Full dummy checkpoint created.")

if __name__ == "__main__":
    main()
