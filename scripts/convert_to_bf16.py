import torch
import os
import json
from safetensors.torch import load_file, save_file
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch
import shutil

INPUT_DIR = "checkpoints/pi05_libero_pytorch"
OUTPUT_DIR = "checkpoints/pi05_libero_bf16"

def convert():
    print(f"Loading state dict from {INPUT_DIR}...")
    # Load safetensors file
    state_dict = load_file(os.path.join(INPUT_DIR, "model.safetensors"))
    
    # Update config for BF16
    config_path = os.path.join(INPUT_DIR, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config_dict["precision"] = "bfloat16"
    
    # Convert tensors to BF16 directly in dict
    print("Converting high-precision tensors to BF16...")
    bf16_state_dict = {}
    for k, v in state_dict.items():
        if v.is_floating_point():
            bf16_state_dict[k] = v.to(torch.bfloat16)
        else:
            bf16_state_dict[k] = v
            
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save Config
    print(f"Saving config to {OUTPUT_DIR}/config.json...")
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
        
    # Save model
    print(f"Saving model to {OUTPUT_DIR}/model.pt...")
    torch.save(bf16_state_dict, os.path.join(OUTPUT_DIR, "model.pt"))
    
    print("Done!")

if __name__ == "__main__":
    convert()
