"""测试原始PyTorch模型的推理功能"""
import sys
import os
sys.path.insert(0, '/home/taco/openpi/src')

import torch
import numpy as np
from openpi.training import config as _config
from openpi.models_pytorch import pi0_pytorch

# 加载配置和模型
print("加载配置...")
config = _config.get_config("pi05_libero")

print("加载PyTorch模型...")
model = pi0_pytorch.PI0Pytorch(config.model)
checkpoint_path = "checkpoints/pi05_libero_pytorch/model.safetensors"
model.load_from_safetensors(checkpoint_path)
model.eval()

# 创建虚拟输入
print("\n创建测试输入...")
device = "cpu"
bsize = 1
model = model.to(device)

# 模拟观察输入
dummy_obs = {
    'base_0_rgb': torch.randn(bsize, 3, 224, 224, device=device),
    'left_wrist_0_rgb': torch.randn(bsize, 3, 224, 224, device=device),
    'right_wrist_0_rgb': torch.randn(bsize, 3, 224, 224, device=device),
    'state': torch.randn(bsize, 8, device=device),
}
dummy_task = "test task"
dummy_actions = None  # 推理模式不需要actions

print("\n运行前向推理...")
try:
    with torch.no_grad():
        output = model(dummy_obs, dummy_task, dummy_actions)
    
    print(f"✅ PyTorch模型推理成功!")
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
    print(f"输出均值: {output.mean().item():.4f}")
    print(f"输出是否有NaN: {torch.isnan(output).any().item()}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

