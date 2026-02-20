#!/usr/bin/env python3
"""
PyTorch INT8 量化脚本 - 使用静态 PTQ（Post-Training Quantization）
"""

import torch
import torch.quantization as tq
import json
import os
from pathlib import Path
from safetensors.torch import load_file, save_file
import sys
from types import SimpleNamespace

# 添加路径
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
    config_dict.setdefault("pi05", True)  # PI0.5 模型
    config_dict.setdefault("dtype", "bfloat16")
    
    # 将 dict 转换为 SimpleNamespace 对象
    config = SimpleNamespace(**config_dict)
    
    print(f"[*] 创建 PI0 模型...")
    model = PI0Pytorch(config)
    
    print(f"[*] 加载权重...")
    weights = load_file(os.path.join(checkpoint_path, "model.safetensors"))
    model.load_state_dict(weights, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print(f"[*] 模型加载完成")
    print(f"    配置: {config}")
    
    return model


def prepare_calibration_data(batch_size: int = 4):
    """准备校准数据 - 使用随机张量模拟"""
    print(f"[*] 准备校准数据 (batch_size={batch_size})...")
    
    calibration_data = {
        "base_rgb": torch.randn(batch_size, 3, 224, 224),
        "left_rgb": torch.randn(batch_size, 3, 224, 224),
        "right_rgb": torch.randn(batch_size, 3, 224, 224),
        "state": torch.randn(batch_size, 15),  # 机器人状态向量
        "tokenized_prompt": torch.randint(0, 30000, (batch_size, 77)),
        "tokenized_prompt_mask": torch.ones(batch_size, 77, dtype=torch.bool),
        "noise": torch.randn(batch_size, 10, 32),  # [horizon, action_dim]
    }
    
    return calibration_data


def quantize_model_static_ptq(model: PI0Pytorch, calibration_data: dict, output_path: str, device: str = "cuda"):
    """静态 PTQ 量化"""
    print(f"\n[*] 应用静态 PTQ 量化...")
    
    # 设置量化配置
    model.qconfig = tq.get_default_qconfig('fbgemm')  # 使用 fbgemm（CPU 优化）或 'qnnpack'
    
    # 准备模型用于量化
    print(f"[*] 准备模型...")
    tq.prepare(model, inplace=True)
    
    # 运行校准数据通过模型
    print(f"[*] 校准模型...")
    with torch.no_grad():
        for step in range(len(calibration_data["noise"])):
            print(f"    - 校准步骤 {step+1}/10...", end="\r")
            calib_input = {
                "base_rgb": calibration_data["base_rgb"].to(device),
                "left_rgb": calibration_data["left_rgb"].to(device),
                "right_rgb": calibration_data["right_rgb"].to(device),
                "state": calibration_data["state"].to(device),
                "tokenized_prompt": calibration_data["tokenized_prompt"].to(device),
                "tokenized_prompt_mask": calibration_data["tokenized_prompt_mask"].to(device),
                "noise": calibration_data["noise"][step:step+1].to(device),
                "timestep": torch.tensor([step], dtype=torch.long).to(device),
            }
            try:
                _ = model(**calib_input)
            except Exception as e:
                print(f"    警告: 校准步骤 {step} 失败: {e}")
    
    print(f"    - 校准完成!")
    
    # 转换为量化模型
    print(f"[*] 转换为量化模型...")
    tq.convert(model, inplace=True)
    
    # 保存量化模型
    print(f"[*] 保存量化模型到 {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    # 保存状态字典
    torch.save(model.state_dict(), os.path.join(output_path, "model_int8.pt"))
    
    # 也保存为 safetensors
    state_dict = model.state_dict()
    save_file(state_dict, os.path.join(output_path, "model_int8.safetensors"))
    
    # 保存配置
    config = {
        "quantization": "static_ptq_int8",
        "qconfig": "fbgemm",
        "device": device,
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"[✓] 量化完成！")
    print(f"    - model_int8.pt: {os.path.getsize(os.path.join(output_path, 'model_int8.pt')) / 1e6:.2f} MB")
    print(f"    - model_int8.safetensors: {os.path.getsize(os.path.join(output_path, 'model_int8.safetensors')) / 1e6:.2f} MB")


def verify_quantized_model(model: PI0Pytorch, calibration_data: dict, device: str = "cuda"):
    """验证量化模型"""
    print(f"\n[*] 验证量化模型...")
    
    with torch.no_grad():
        for step in range(3):  # 只验证前 3 步
            test_input = {
                "base_rgb": calibration_data["base_rgb"][:1].to(device),
                "left_rgb": calibration_data["left_rgb"][:1].to(device),
                "right_rgb": calibration_data["right_rgb"][:1].to(device),
                "state": calibration_data["state"][:1].to(device),
                "tokenized_prompt": calibration_data["tokenized_prompt"][:1].to(device),
                "tokenized_prompt_mask": calibration_data["tokenized_prompt_mask"][:1].to(device),
                "noise": calibration_data["noise"][step:step+1].to(device),
                "timestep": torch.tensor([step], dtype=torch.long).to(device),
            }
            
            try:
                output = model(**test_input)
                print(f"    ✓ 步骤 {step}: output.shape={output.shape}, dtype={output.dtype}")
                assert not torch.isnan(output).any(), f"步骤 {step} 包含 NaN"
                assert not torch.isinf(output).any(), f"步骤 {step} 包含 Inf"
            except Exception as e:
                print(f"    ✗ 步骤 {step} 失败: {e}")
                raise
    
    print(f"[✓] 验证通过！")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 使用设备: {device}")
    
    # 模型路径
    checkpoint_path = "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_jax"
    output_path = "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_int8"
    
    # 加载模型
    model = load_pytorch_model(checkpoint_path, device=device)
    
    # 准备校准数据
    calibration_data = prepare_calibration_data(batch_size=4)
    
    # 执行量化
    quantize_model_static_ptq(model, calibration_data, output_path, device=device)
    
    # 验证量化模型
    verify_quantized_model(model, calibration_data, device=device)
    
    print(f"\n[✓] 所有步骤完成！")
    print(f"    量化模型路径: {output_path}")


if __name__ == "__main__":
    main()
