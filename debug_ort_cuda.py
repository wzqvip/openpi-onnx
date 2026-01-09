import ctypes
import os
import sys

# Set RTLD_GLOBAL to ensure symbols are visible to providers
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)

root = "/home/taco/openpi-onnx/.venv/lib/python3.11/site-packages/onnxruntime/capi"
shared_path = os.path.join(root, "libonnxruntime_providers_shared.so")

# Preload shared provider library
try:
    ctypes.CDLL(shared_path, mode=ctypes.RTLD_GLOBAL)
    print(f"Preloaded {shared_path}")
except Exception as e:
    print(f"Failed to preload {shared_path}: {e}")

import onnxruntime as ort

print(f"Available providers: {ort.get_available_providers()}")

try:
    sess = ort.InferenceSession("checkpoints/pi05_libero_pytorch/model.fp32.onnx", providers=["TensorRTExecutionProvider", "CUDAExecutionProvider"])
    print(f"Session created with providers: {sess.get_providers()}")
except Exception as e:
    print(f"Error creating session: {e}")
