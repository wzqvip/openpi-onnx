#!/usr/bin/env python3
"""
对比 FP32、INT8、FP8 模型的性能和精度

测试流程:
1. 加载时间对比
2. 模型大小对比
3. 参数大小对比
4. 单个推理步骤速度对比
5. 推理精度对比 (vs FP32 基准)
"""

import torch
import torch.nn as nn
import json
import os
import sys
import time
import numpy as np
from types import SimpleNamespace
import gc

sys.path.insert(0, '/home/taco/openpi/src')
sys.path.insert(0, '/home/taco/openpi-onnx')

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class PerformanceTest:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {}
        self.fp32_baseline = None
        
    def load_model(self, model_type: str):
        """加载模型"""
        print(f"\n[*] 加载 {model_type} 模型...")
        
        start = time.time()
        
        if model_type == "FP32":
            from safetensors.torch import load_file
            checkpoint_path = "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_jax"
            
            config_path = os.path.join(checkpoint_path, "config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            
            config_dict.setdefault("pi05", True)
            config_dict.setdefault("dtype", "bfloat16")
            config = SimpleNamespace(**config_dict)
            
            model = PI0Pytorch(config)
            weights = load_file(os.path.join(checkpoint_path, "model.safetensors"))
            model.load_state_dict(weights, strict=False)
            
        elif model_type == "INT8":
            model = torch.load(
                "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_int8_dynamic/model_int8_full.pt",
                weights_only=False
            )
        elif model_type == "FP8":
            model = torch.load(
                "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_fp8/model_fp8_full.pt",
                weights_only=False
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model.to(self.device)
        model.eval()
        
        load_time = time.time() - start
        
        print(f"    ✓ 加载时间: {load_time:.3f}s")
        self.results[model_type] = {"load_time": load_time}
        
        return model
    
    def measure_model_size(self, model_type: str):
        """测试模型文件大小"""
        paths = {
            "FP32": "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_jax/model.safetensors",
            "INT8": "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_int8_dynamic/model_int8.pt",
            "FP8": "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch_fp8/model_fp8.pt",
        }
        
        if model_type not in self.results:
            self.results[model_type] = {}
        
        if model_type in paths and os.path.exists(paths[model_type]):
            size_gb = os.path.getsize(paths[model_type]) / 1e9
            self.results[model_type]["file_size_gb"] = size_gb
            return size_gb
        return None
    
    def measure_parameters(self, model: nn.Module, model_type: str):
        """测试参数大小"""
        # 计算参数内存占用
        total_params = 0
        for param in model.parameters():
            total_params += param.numel() * param.element_size()
        
        param_size_mb = total_params / 1e6
        self.results[model_type]["param_size_mb"] = param_size_mb
        print(f"    参数大小: {param_size_mb:.1f} MB")
        
        return param_size_mb
    
    def generate_test_observation(self, batch_size: int = 1):
        """生成随机观测"""
        from collections import namedtuple
        
        # 创建 observation 数据结构
        Observation = namedtuple("Observation", ["image", "state"])
        
        # 图像: batch_size x 3 x (224, 224) 三个摄像头
        base_rgb = torch.randn(batch_size, 3, 224, 224, device=self.device)
        left_rgb = torch.randn(batch_size, 3, 224, 224, device=self.device)
        right_rgb = torch.randn(batch_size, 3, 224, 224, device=self.device)
        
        # 图像掩码
        image_masks = torch.ones(batch_size, 3, dtype=torch.bool, device=self.device)
        
        # 文本 tokens
        language_tokens = torch.randint(0, 30000, (batch_size, 77), device=self.device)
        language_masks = torch.ones(batch_size, 77, dtype=torch.bool, device=self.device)
        
        # 状态
        state = torch.randn(batch_size, 15, device=self.device)
        
        # 组织成观测
        observation = {
            "image": torch.stack([base_rgb, left_rgb, right_rgb], dim=1),  # batch x 3 x 3 x 224 x 224
            "image_mask": image_masks,
            "language_tokens": language_tokens,
            "language_mask": language_masks,
            "state": state,
        }
        
        return observation
    
    def benchmark_inference_step(self, model: PI0Pytorch, model_type: str, num_steps: int = 5):
        """测试单步推理速度"""
        print(f"\n[*] 推理速度测试 ({model_type}, {num_steps} 步)...")
        
        batch_size = 1
        times = []
        
        with torch.no_grad():
            try:
                # 预热
                print(f"    预热...", end="", flush=True)
                for _ in range(2):
                    observation = self.generate_test_observation(batch_size)
                    _ = model.sample_actions(observation, num_steps=1)
                print(" 完成")
                
                # 实际测试
                print(f"    运行...", end="", flush=True)
                for step in range(num_steps):
                    observation = self.generate_test_observation(batch_size)
                    
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    
                    start = time.time()
                    actions = model.sample_actions(observation, num_steps=10)
                    
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    
                    elapsed = time.time() - start
                    times.append(elapsed)
                
                print(" 完成")
                
                times = np.array(times)
                mean_time_ms = times.mean() * 1000
                std_time_ms = times.std() * 1000
                
                print(f"    结果 (推理10个动作步):")
                print(f"      平均延迟: {mean_time_ms:.2f}ms")
                print(f"      标准差:   {std_time_ms:.2f}ms")
                print(f"      吞吐量:   {1000/mean_time_ms:.1f} 次/秒")
                
                self.results[model_type]["inference_mean_ms"] = mean_time_ms
                self.results[model_type]["inference_std_ms"] = std_time_ms
                
            except Exception as e:
                print(f" 失败")
                print(f"    错误: {e}")
                self.results[model_type]["inference_mean_ms"] = None
    
    def benchmark_accuracy(self, model_fp32: PI0Pytorch, model_quant: PI0Pytorch, 
                          quant_type: str, num_tests: int = 5):
        """对比量化模型与 FP32 的精度"""
        print(f"\n[*] 精度对比测试 ({quant_type} vs FP32, {num_tests} 组测试)...")
        
        distances = []
        
        with torch.no_grad():
            for test_idx in range(num_tests):
                observation = self.generate_test_observation(batch_size=1)
                
                # FP32 推理
                actions_fp32 = model_fp32.sample_actions(observation, num_steps=10)
                
                # 量化模型推理
                actions_quant = model_quant.sample_actions(observation, num_steps=10)
                
                # 计算 L2 距离
                distance = torch.norm(actions_fp32 - actions_quant, p=2).item()
                distances.append(distance)
        
        distances = np.array(distances)
        print(f"    L2 距离统计:")
        print(f"      平均: {distances.mean():.6f}")
        print(f"      标准差: {distances.std():.6f}")
        print(f"      最小: {distances.min():.6f}")
        print(f"      最大: {distances.max():.6f}")
        
        self.results[quant_type]["l2_distance_mean"] = distances.mean()
        self.results[quant_type]["l2_distance_std"] = distances.std()
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*100)
        print("测试结果总结".center(100))
        print("="*100)
        
        # 模型大小
        print("\n【模型大小】")
        print(f"  {'模型':10} {'文件大小':15} {'参数大小':15} {'相对于FP32':15}")
        print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
        
        fp32_size = self.measure_model_size("FP32")
        for model_type in ["FP32", "INT8", "FP8"]:
            if model_type not in self.results:
                continue
            
            file_size = self.results[model_type].get("file_size_gb", 0)
            param_size = self.results[model_type].get("param_size_mb", 0)
            
            if fp32_size:
                ratio = file_size / fp32_size * 100
                print(f"  {model_type:10} {file_size:.2f} GB {param_size:14.0f}MB  {ratio:14.1f}%")
            else:
                print(f"  {model_type:10} {file_size:.2f} GB {param_size:14.0f}MB")
        
        # 加载时间
        print("\n【加载性能】")
        print(f"  {'模型':10} {'加载时间':15} {'相对于FP32':15}")
        print(f"  {'-'*10} {'-'*15} {'-'*15}")
        
        fp32_load = self.results.get("FP32", {}).get("load_time", 1)
        for model_type in ["FP32", "INT8", "FP8"]:
            if model_type not in self.results:
                continue
            
            load_time = self.results[model_type].get("load_time", 0)
            if fp32_load:
                ratio = load_time / fp32_load
                status = "✓ 快" if ratio < 1 else "✗ 慢"
                print(f"  {model_type:10} {load_time:.3f}s {ratio:.2f}x {status}")
        
        # 推理速度
        print("\n【推理性能】")
        print(f"  {'模型':10} {'延迟':15} {'吞吐量':15} {'相对于FP32':15}")
        print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
        
        fp32_inference = self.results.get("FP32", {}).get("inference_mean_ms", 1)
        for model_type in ["FP32", "INT8", "FP8"]:
            if model_type not in self.results:
                continue
            
            inference_ms = self.results[model_type].get("inference_mean_ms")
            if inference_ms is None:
                print(f"  {model_type:10} {'N/A':15}")
                continue
            
            throughput = 1000 / inference_ms
            ratio = inference_ms / fp32_inference if fp32_inference else 1
            status = "✓ 快" if ratio < 1 else "✗ 慢"
            print(f"  {model_type:10} {inference_ms:.2f}ms {throughput:.1f} op/s  {ratio:.2f}x {status}")
        
        # 精度
        print("\n【精度对比】")
        print(f"  {'模型':10} {'L2距离':20} {'状态':20}")
        print(f"  {'-'*10} {'-'*20} {'-'*20}")
        
        for model_type in ["INT8", "FP8"]:
            if model_type not in self.results:
                continue
            
            distance = self.results[model_type].get("l2_distance_mean")
            if distance is None:
                print(f"  {model_type:10} {'N/A':20}")
            else:
                status = "✓ 高精度" if distance < 0.1 else "! 需检查"
                print(f"  {model_type:10} {distance:.6f} {status}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] 模型性能和精度对比测试")
    print(f"    设备: {device}")
    print(f"    PyTorch: {torch.__version__}")
    
    tester = PerformanceTest(device=device)
    
    # 1. 测试模型大小
    print("\n[*] 1. 模型大小测试")
    for model_type in ["FP32", "INT8", "FP8"]:
        size = tester.measure_model_size(model_type)
        if size:
            print(f"    {model_type}: {size:.2f} GB")
    
    # 2. 加载所有模型
    models = {}
    for model_type in ["FP32", "INT8", "FP8"]:
        try:
            models[model_type] = tester.load_model(model_type)
        except Exception as e:
            print(f"    ✗ 加载失败: {e}")
    
    # 3. 测试参数大小
    print("\n[*] 2. 参数大小测试")
    for model_type, model in models.items():
        print(f"    {model_type}:")
        tester.measure_parameters(model, model_type)
    
    # 4. 推理速度测试
    print("\n[*] 3. 推理速度测试")
    for model_type, model in models.items():
        tester.benchmark_inference_step(model, model_type, num_steps=5)
    
    # 5. 精度对比（量化 vs FP32）
    if "FP32" in models and "INT8" in models:
        print("\n[*] 4. 精度对比测试")
        tester.benchmark_accuracy(models["FP32"], models["INT8"], "INT8", num_tests=5)
    
    if "FP32" in models and "FP8" in models:
        tester.benchmark_accuracy(models["FP32"], models["FP8"], "FP8", num_tests=5)
    
    # 6. 打印总结
    tester.print_summary()
    
    print(f"\n[✓] 测试完成")


if __name__ == "__main__":
    main()
