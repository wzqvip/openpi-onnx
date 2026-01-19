
import h5py
import numpy as np
import sys
import os

def print_structure(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_hdf5.py <file.hdf5>")
        return

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Inspecting: {filepath}")
    with h5py.File(filepath, "r") as f:
        # Inspect root
        f.visititems(print_structure)
        
        # Check first demo
        if "data" in f:
            demo_key = list(f["data"].keys())[0]
            demo = f["data"][demo_key]
            print(f"\n--- Demo {demo_key} Sample Values ---")
            
            # Check obs
            if "obs" in demo:
                obs = demo["obs"]
                print("Observations:")
                for k in obs.keys():
                    print(f"  {k}: {obs[k].shape}")
                    
            # Check actions
            if "actions" in demo:
                print(f"Actions: {demo['actions'].shape}")
                print(f"First Action: {demo['actions'][0]}")
                
            # Check states
            if "states" in demo:
                print(f"States: {demo['states'].shape}")

if __name__ == "__main__":
    main()
