import ctypes
import os

root = "/home/taco/openpi-onnx/.venv/lib/python3.11/site-packages/onnxruntime/capi"
shared_path = os.path.join(root, "libonnxruntime_providers_shared.so")
cuda_path = os.path.join(root, "libonnxruntime_providers_cuda.so")

print(f"Attempting to load shared: {shared_path}")
try:
    # GLOBAL flag might be needed to expose symbols to subsequently loaded libs
    ctypes.CDLL(shared_path, mode=ctypes.RTLD_GLOBAL)
    print("Successfully loaded shared library")
except OSError as e:
    print(f"Failed to load shared library: {e}")

print(f"Attempting to load cuda: {cuda_path}")
try:
    lib = ctypes.CDLL(cuda_path)
    print("Successfully loaded cuda library")
except OSError as e:
    print(f"Failed to load cuda library: {e}")
