
from safetensors import safe_open
import sys

path = "/home/taco/checkpoints/pi05_libero_onnx_compat/model.safetensors"
out = "keys.txt"

print(f"Reading keys from {path}")
with safe_open(path, framework="pt", device="cpu") as f:
    keys = f.keys()
    print(f"Found {len(keys)} keys")
    with open(out, "w") as o:
        for k in keys[:50]:
            o.write(k + "\n")
print(f"Wrote 50 keys to {out}")
