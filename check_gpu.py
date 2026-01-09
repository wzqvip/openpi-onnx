import onnxruntime as ort
print(f"Available providers: {ort.get_available_providers()}")

try:
    import torch
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
    print(f"PyTorch CUDA device name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not installed")
