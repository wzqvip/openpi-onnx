#!/usr/bin/env python3
"""
验证量化后的 PyTorch INT8 模型推理
"""

import torch
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, '/home/taco/openpi/src')
sys.path.insert(0, '/home/taco/openpi-onnx')

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def load_quantized_model(checkpoint_path: str, device: str = "cuda") -> PI0Pytorch:
    """加载量化模型"""
    print(f"[*] 加载量化模型...")
    model = torch.load(os.path.join(checkpoint_path, "model_int8_full.pt"), weights_only=False)
    model = model.to(device)
    model.eval()
    print(f"[✓] 模型加载完成")
    return model


def test_inference(model: PI0Pytorch, device: str = "cuda"):
    """测试推理"""
    print(f"\n[*] 测试推理...")
    
    # 模拟输入
    batch_size = 1
    test_input = {
        "base_rgb": torch.randn(batch_size, 3, 224, 224).to(device),
        "left_rgb": torch.randn(batch_size, 3, 224, 224).to(device),
        "right_rgb": torch.randn(batch_size, 3, 224, 224).to(device),
        "state": torch.randn(batch_size, 15).to(device),
        "tokenized_prompt": torch.randint(0, 30000, (batch_size, 77)).to(device),
        "tokenized_prompt_mask": torch.ones(batch_size, 77, dtype=torch.bool).to(device),
        "noise": torch.randn(batch_size, 10, 32).to(device),
        "timestep": torch.tensor([0], dtype=torch.long).to(device),
    }
    
    with torch.no_grad():
        try:
            output = model(**test_input)
            print(f"[✓] 推理成功！")
            print(f"    输出形状: {output.shape}")
            print(f"    输出值范围: [{output.min():.4f}, {output.max():.4f}]")
            print(f"    包含NaN: {torch.isnan(output).any()}")
            print(f"    包含Inf: {torch.isinf(output).any()}")
            return True
        except Exception as e:
            print(f"[✗] 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def compare_model_sizes():
    """对比模型大小"""
    print(f"\n[*] 模型大小对比:")
    
    fp32_path = "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_jax/model.safetensors"
    int8_path = "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_int8_dynamic/model_int8.pt"
    
    if os.path.exists(fp32_path):
        fp32_size = os.path.getsize(fp32_path) / 1e9
        print(f"    FP32 (safetensors): {fp32_size:.2f} GB")
    
    if os.path.exists(int8_path):
        int8_size = os.path.getsize(int8_path) / 1e9
        print(f"    INT8 (quantized):   {int8_size:.2f} GB")
        
        if os.path.exists(fp32_path):
            reduction = (1 - int8_size / fp32_size) * 100
            print(f"    压缩率: {reduction:.1f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 使用设备: {device}")
    
    model_path = "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_int8_dynamic"
    
    # 加载模型
    model = load_quantized_model(model_path, device=device)
    
    # 测试推理
    success = test_inference(model, device=device)
    
    # 对比大小
    compare_model_sizes()
    
    if success:
        print(f"\n[✓] 量化模型验证通过！")
    else:
        print(f"\n[✗] 量化模型验证失败！")


if __name__ == "__main__":
    main()
