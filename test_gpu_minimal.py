import torch
import torch.nn as nn

def test_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
    device = "cuda"
    torch.backends.cudnn.enabled = False
    print("CuDNN Disabled")
    
    # 1. Basic Tensor Ops
    print("Testing basic tensor ops...")
    try:
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        c = torch.matmul(a, b)
        print("Matmul successful")
    except Exception as e:
        print(f"Matmul failed: {e}")

    # 2. Conv2d (CuDNN check)
    print("Testing Conv2d (CuDNN)...")
    try:
        conv = nn.Conv2d(3, 64, kernel_size=3).to(device)
        x = torch.randn(1, 3, 224, 224, device=device)
        y = conv(x)
        print("Conv2d successful")
    except Exception as e:
        print(f"Conv2d failed: {e}")

    # 3. BF16 check
    print("Testing BF16...")
    try:
        a = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        c = torch.matmul(a, b)
        print("BF16 Matmul successful")
    except Exception as e:
        print(f"BF16 Matmul failed: {e}")

if __name__ == "__main__":
    test_gpu()
