
import torch
from safetensors.torch import load_file, save_file
import os
import argparse

def patch_checkpoint(path, target_action_dim=7):
    print(f"Loading checkpoint from {path}...")
    state_dict = load_file(path)
    
    new_state_dict = {}
    modified = False
    
    for key, val in state_dict.items():
        new_val = val
        if "action_in_proj" in key:
            # action_in_proj.weight: (hidden, action_dim)
            if "weight" in key:
                if val.shape[1] > target_action_dim:
                    print(f"Slicing {key} from {val.shape} to (..., {target_action_dim})")
                    new_val = val[:, :target_action_dim].contiguous()
                    modified = True
                else:
                    print(f"Skipping {key}: shape {val.shape} matches or is smaller than target {target_action_dim}")
            # bias? Linear usually doesn't have bias on input side? 
            # If it does, dimensions match hidden.
            
        elif "action_out_proj" in key:
            # action_out_proj.weight: (action_dim, hidden)
            if "weight" in key:
                if val.shape[0] > target_action_dim:
                    print(f"Slicing {key} from {val.shape} to ({target_action_dim}, ...)")
                    new_val = val[:target_action_dim, :].contiguous()
                    modified = True
            # action_out_proj.bias: (action_dim)
            elif "bias" in key:
                if val.shape[0] > target_action_dim:
                    print(f"Slicing {key} from {val.shape} to ({target_action_dim})")
                    new_val = val[:target_action_dim].contiguous()
                    modified = True
        
        new_state_dict[key] = new_val
        
    if modified:
        print("Saving modified checkpoint...")
        # Backup first
        backup_path = path + ".bak"
        if not os.path.exists(backup_path):
            os.rename(path, backup_path)
            print(f"Backed up original to {backup_path}")
        
        save_file(new_state_dict, path)
        print(f"Saved to {path}")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    patch_checkpoint("/home/taco/checkpoints/pi05_libero_onnx_compat/model.safetensors")
