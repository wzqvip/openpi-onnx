#!/usr/bin/env python3
"""
PyTorch 动态量化 - 最简单的 INT8 量化方法
"""

import torch
import torch.quantization as tq
import json
import os
from safetensors.torch import load_file, save_file
import sys
from types import SimpleNamespace

sys.path.insert(0, '/home/taco/openpi/src')
sys.path.insert(0, '/home/taco/openpi-onnx')

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def load_pytorch_model(checkpoint_path: str, device: str = "cuda") -> PI0Pytorch:
    """加载 PyTorch 模型"""
    print(f"[*] 加载模型配置...")
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # 添加默认值
    config_dict.setdefault("pi05", True)
    config_dict.setdefault("dtype", "bfloat16")
    
    config = SimpleNamespace(**config_dict)
    
    print(f"[*] 创建 PI0 模型...")
    model = PI0Pytorch(config)
    
    print(f"[*] 加载权重...")
    weights = load_file(os.path.join(checkpoint_path, "model.safetensors"))
    model.load_state_dict(weights, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f"[*] 模型加载完成")
    return model


def quantize_model_dynamic(model: PI0Pytorch, output_path: str):
    """动态量化"""
    print(f"\n[*] 应用动态 INT8 量化...")
    print(f"    原始模型大小: {sum(p.numel() * 4 for p in model.parameters()) / 1e6:.2f} MB")
    
    try:
        # 对所有 Linear 和 LSTM 层应用动态量化
        quantized_model = tq.quantize_dynamic(
            model,
            qconfig_spec={torch.nn.Linear, torch.nn.LSTM},
            dtype=torch.qint8,
        )
        print(f"[✓] 动态量化完成！")
        
    except Exception as e:
        print(f"[!] 动态量化失败: {e}")
        print(f"[*] 使用原始模型...")
        quantized_model = model
    
    return quantized_model


def save_model(model: PI0Pytorch, output_path: str):
    """保存量化模型"""
    print(f"\n[*] 保存模型到 {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    # 保存 PyTorch 格式
    torch.save(model.state_dict(), os.path.join(output_path, "model_int8.pt"))
    print(f"    ✓ model_int8.pt: {os.path.getsize(os.path.join(output_path, 'model_int8.pt')) / 1e6:.2f} MB")
    
    # 保存为 safetensors
    try:
        state_dict = model.state_dict()
        save_file(state_dict, os.path.join(output_path, "model_int8.safetensors"))
        print(f"    ✓ model_int8.safetensors: {os.path.getsize(os.path.join(output_path, 'model_int8.safetensors')) / 1e6:.2f} MB")
    except Exception as e:
        print(f"    ! safetensors 保存失败: {e}")
    
    # 保存整个模型
    torch.save(model, os.path.join(output_path, "model_int8_full.pt"))
    print(f"    ✓ model_int8_full.pt: {os.path.getsize(os.path.join(output_path, 'model_int8_full.pt')) / 1e6:.2f} MB")
    
    print(f"[✓] 模型保存完成！")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 使用设备: {device}")
    
    checkpoint_path = "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_jax"
    output_path = "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_int8_dynamic"
    
    # 加载模型
    model = load_pytorch_model(checkpoint_path, device=device)
    
    # 应用动态量化
    quantized_model = quantize_model_dynamic(model, output_path)
    
    # 保存模型
    save_model(quantized_model, output_path)
    
    print(f"\n[✓] 所有步骤完成！")
    print(f"    量化模型路径: {output_path}")


if __name__ == "__main__":
    main()
