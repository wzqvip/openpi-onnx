#!/usr/bin/env python3
"""
PyTorch FP8 量化 - 8位浮点格式
使用 torch.float8_e4m3fn (NVIDIA 推荐格式)
"""

import torch
import json
import os
from pathlib import Path
from safetensors.torch import load_file, save_file
import sys
from types import SimpleNamespace
import time

sys.path.insert(0, '/home/taco/openpi/src')
sys.path.insert(0, '/home/taco/openpi-onnx')

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def load_pytorch_model(checkpoint_path: str, device: str = "cuda") -> PI0Pytorch:
    """加载 PyTorch 模型"""
    print(f"[*] 加载模型配置...")
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
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


def quantize_to_fp8(model: PI0Pytorch, output_path: str):
    """将模型量化为 FP8"""
    print(f"\n[*] 转换模型为 FP8 格式 (float8_e4m3fn)...")
    
    # 获取原始模型大小
    original_size = sum(p.numel() * 4 for p in model.parameters()) / 1e9
    print(f"    原始模型大小 (FP32): {original_size:.2f} GB")
    
    # 递归遍历所有模块，将权重转换为 FP8
    converted_params = 0
    total_params = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            total_params += 1
            try:
                # 转换为 float8_e4m3fn
                fp8_param = param.to(torch.float8_e4m3fn)
                # 替换原参数
                param.data = fp8_param
                converted_params += 1
                if converted_params % 50 == 0:
                    print(f"    已转换 {converted_params} 个参数...", end="\r")
            except Exception as e:
                print(f"    警告: 参数 {name} 转换失败: {e}")
    
    print(f"    ✓ 转换完成: {converted_params}/{total_params} 参数")
    
    # 保存模型
    print(f"\n[*] 保存 FP8 模型...")
    os.makedirs(output_path, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(output_path, "model_fp8.pt"))
    fp8_size = os.path.getsize(os.path.join(output_path, "model_fp8.pt")) / 1e9
    print(f"    ✓ model_fp8.pt: {fp8_size:.2f} GB")
    
    # 完整模型
    torch.save(model, os.path.join(output_path, "model_fp8_full.pt"))
    full_size = os.path.getsize(os.path.join(output_path, "model_fp8_full.pt")) / 1e9
    print(f"    ✓ model_fp8_full.pt: {full_size:.2f} GB")
    
    # 压缩率
    print(f"\n[✓] FP8 量化完成！")
    print(f"    原始大小: {original_size:.2f} GB")
    print(f"    FP8 大小: {fp8_size:.2f} GB")
    print(f"    压缩率:   {(1 - fp8_size/original_size)*100:.1f}%")


def quantize_to_fp8_auto(model: PI0Pytorch, output_path: str):
    """使用 torch._nn.to_fp8() 的自动 FP8 量化（如果可用）"""
    print(f"\n[*] 尝试自动 FP8 量化...")
    
    try:
        # 检查是否有自动转换函数
        if hasattr(torch.nn, 'to_fp8'):
            print(f"    使用 torch.nn.to_fp8()...")
            model_fp8 = torch.nn.to_fp8(model)
        elif hasattr(torch, 'quantization') and hasattr(torch.quantization, 'quantize_fp8'):
            print(f"    使用 torch.quantization.quantize_fp8()...")
            model_fp8 = torch.quantization.quantize_fp8(model)
        else:
            print(f"    ! 未找到自动 FP8 函数，跳过自动量化")
            return
        
        # 保存
        os.makedirs(output_path, exist_ok=True)
        torch.save(model_fp8.state_dict(), os.path.join(output_path, "model_fp8_auto.pt"))
        print(f"    ✓ 自动 FP8 量化完成")
        
    except Exception as e:
        print(f"    ! 自动 FP8 量化失败: {e}")


def compare_fp8_with_fp32():
    """对比 FP8 和 FP32 的模型大小"""
    print(f"\n[*] 模型大小对比:")
    
    paths = {
        "FP32 原始": "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_jax/model.safetensors",
        "INT8 动态": "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_int8_dynamic/model_int8.pt",
        "FP8 转换": "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_fp8/model_fp8.pt",
    }
    
    sizes = {}
    for label, path in paths.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / 1e9
            sizes[label] = size
            print(f"    {label:15} {size:7.2f} GB")
    
    if "FP32 原始" in sizes and "FP8 转换" in sizes:
        ratio = sizes["FP8 转换"] / sizes["FP32 原始"] * 100
        print(f"\n    FP8 vs FP32: {ratio:.1f}%")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 使用设备: {device}")
    print(f"[*] PyTorch 版本: {torch.__version__}")
    
    # 检查 FP8 支持
    try:
        test_tensor = torch.randn(2, 2).to(torch.float8_e4m3fn)
        print(f"[✓] FP8 (float8_e4m3fn) 支持: 是")
    except Exception as e:
        print(f"[!] FP8 支持检查失败: {e}")
    
    checkpoint_path = "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_jax"
    output_path = "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_fp8"
    
    # 加载模型
    model = load_pytorch_model(checkpoint_path, device=device)
    
    # FP8 量化
    start = time.time()
    quantize_to_fp8(model, output_path)
    elapsed = time.time() - start
    print(f"    耗时: {elapsed:.1f}s")
    
    # 尝试自动量化
    quantize_to_fp8_auto(model, output_path)
    
    # 对比大小
    compare_fp8_with_fp32()
    
    print(f"\n[✓] FP8 量化完成！")
    print(f"    模型路径: {output_path}")


if __name__ == "__main__":
    main()
