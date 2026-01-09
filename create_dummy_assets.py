
import os
import json
import numpy as np
import pathlib
from openpi.shared import normalize
from openpi.training import config as _config

def main():
    checkpoint_dir = pathlib.Path("./checkpoints/pi05_libero_pytorch")
    # Path where code expects assets: assets/physical-intelligence/libero/norm_stats.json
    # derived from: assets_dir / asset_id / norm_stats.json
    # Config default assets_dir is checkpoint_dir/assets (replicated assets)
    
    asset_rel_path = "assets/physical-intelligence/libero"
    target_dir = checkpoint_dir / asset_rel_path
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating dummy norm_stats.json in {target_dir}...")
    
    # Defaults for Pi0Config
    action_dim = 32 # Default in pi0_config.py
    # Pi05 might use different? 
    # train config "pi05_libero" uses default action_dim?
    # Actually pi0_config default is 32.
    
    # We create stats for 'actions' and 'state' and 'joint_position' etc just in case
    # Libero keys: 'actions', 'state', 'joint_position', 'gripper_position'?
    # Repo uses 'actions', 'state' in RepackTransform.
    
    keys = ["actions", "state", "joint_position", "gripper_position"]
    
    stats = {}
    for k in keys:
        # Create identity normalization (mean 0, std 1)
        # Size 32 is safe upper bound? 
        # Actually providing 1D array of size 32 should work if broadcasting or matching size.
        # But if model expects 7, 32 might fail check.
        # Libero usually 7 dof + 1 gripper = 8? Or 14?
        # Let's create size 32.
        size = 32
        
        stats[k] = normalize.NormStats(
            mean=np.zeros(size),
            std=np.ones(size),
            q01=np.zeros(size)-1.0,
            q99=np.ones(size)+1.0
        )
        
    normalize.save(target_dir, stats)
    print("Norm stats created.")

if __name__ == "__main__":
    main()
