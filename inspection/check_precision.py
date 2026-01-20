
import torch
import os

CHECKPOINT_DIR = "./checkpoints/pi05_libero_pytorch"

def check_precision():
    print(f"Checking precision of model in {CHECKPOINT_DIR}...")
    try:
        # Try to load a safetensor or bin
        from safetensors.torch import load_file
        files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.safetensors')]
        if files:
            p = os.path.join(CHECKPOINT_DIR, files[0])
            st = load_file(p)
            for k, v in list(st.items())[:3]:
                print(f"{k}: {v.dtype}")
        else:
            # Fallback to bin
            files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.bin') or f.endswith('.pt')]
            if files:
                p = os.path.join(CHECKPOINT_DIR, files[0])
                # This might be full state dict
                sd = torch.load(p, map_location='cpu')
                if isinstance(sd, dict):
                     for k, v in list(sd.items())[:3]:
                        if hasattr(v, 'dtype'):
                            print(f"{k}: {v.dtype}")
    except Exception as e:
        print(f"Error checking: {e}")

if __name__ == "__main__":
    check_precision()
