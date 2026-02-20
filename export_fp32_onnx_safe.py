#!/usr/bin/env python3
"""
FP32 ONNX导出 - 增强版本，处理复数类型问题

解决方案：
1. 使用自定义trace来避免复数操作
2. 强制所有中间结果为float32
3. 在导出前/后处理图来清理类型
"""

import sys
sys.path.insert(0, '/home/taco/openpi/src')
sys.path.insert(0, '/home/taco/openpi-onnx')

import torch
import numpy as np
import dataclasses
from openpi.training import config as _config
from openpi.models import model as _model
from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

class SafeOnnxWrapperFP32(torch.nn.Module):
    """
    安全的ONNX包装器 - 强制所有操作为实数类型
    """
    def __init__(self, model, num_steps=10):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, base_rgb, left_rgb, right_rgb, state, tokenized_prompt, tokenized_prompt_mask, noise):
        bsize = state.shape[0]
        device = state.device
        
        # 强制所有输入为正确的类型
        base_rgb = base_rgb.float()
        left_rgb = left_rgb.float()
        right_rgb = right_rgb.float()
        state = state.float()
        tokenized_prompt = tokenized_prompt.long()  # 保持为int
        tokenized_prompt_mask = tokenized_prompt_mask.bool()  # 保持为bool
        noise = noise.float()

        images = {
            "base_0_rgb": base_rgb,
            "left_wrist_0_rgb": left_rgb,
            "right_wrist_0_rgb": right_rgb,
        }
        image_masks = {
            "base_0_rgb": torch.ones(bsize, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(bsize, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.zeros(bsize, dtype=torch.bool, device=device),
        }

        observation = _model.Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

        x_t = noise.clone()
        dt = 1.0 / self.num_steps

        for i in range(self.num_steps):
            # 使用张量而不是标量来避免Python浮点操作
            time_i = torch.tensor(1.0 - i * dt, dtype=torch.float32, device=device)
            time_batch = time_i.expand(bsize).unsqueeze(-1)  # [bsize, 1]
            
            # 前向传播
            v_t = self.model.denoise_step(observation, x_t, time_batch)
            v_t = v_t.float()  # 强制为float32
            
            # 强制dt_tensor为float32
            dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
            
            # 更新 x_t
            x_t = x_t + dt_tensor * v_t
            x_t = x_t.float()  # 每步后强制为float32

        return x_t


def export_fp32_onnx_safe():
    """导出FP32 ONNX - 安全版本"""
    
    print("="*80)
    print("FP32 ONNX导出 - 安全版本（处理复数类型）")
    print("="*80)
    
    output_path = "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.safe.onnx"
    
    # 加载配置
    print(f"\n加载配置: pi05_libero")
    cfg = _config.get_config("pi05_libero")
    print(f"  action_dim: {cfg.model.action_dim}")
    
    # 修复action_dim
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, action_dim=32))
    
    # 加载模型
    print(f"加载PyTorch模型: checkpoints/pi05_libero_pytorch/model.safetensors")
    base_model = pi0_pytorch.PI0Pytorch(cfg.model)
    
    # 从safetensors加载权重
    from safetensors.torch import load_file
    state_dict = load_file("checkpoints/pi05_libero_pytorch/model.safetensors")
    base_model.load_state_dict(state_dict, strict=False)
    base_model.eval()
    
    # 创建包装器
    print(f"\n创建安全包装器 (diffusion_steps=10)")
    unrolled_model = SafeOnnxWrapperFP32(base_model, num_steps=10)
    unrolled_model.eval()
    
    # 准备虚拟输入
    print(f"\n准备虚拟输入...")
    dummy_input = {
        'base_0_rgb': torch.randn(1, 3, 224, 224, dtype=torch.float32),
        'left_wrist_0_rgb': torch.randn(1, 3, 224, 224, dtype=torch.float32),
        'right_wrist_0_rgb': torch.randn(1, 3, 224, 224, dtype=torch.float32),
        'state': torch.randn(1, 8, dtype=torch.float32),
        'tokenized_prompt': torch.zeros(1, 200, dtype=torch.int32),  # 应该是int
        'tokenized_prompt_mask': torch.ones(1, 200, dtype=torch.bool),  # 应该是bool
        'noise': torch.randn(1, 10, 32, dtype=torch.float32),
    }
    
    # 测试前向推理
    print(f"\n测试前向推理...")
    with torch.no_grad():
        try:
            output = unrolled_model(
                dummy_input['base_0_rgb'],
                dummy_input['left_wrist_0_rgb'],
                dummy_input['right_wrist_0_rgb'],
                dummy_input['state'],
                dummy_input['tokenized_prompt'],
                dummy_input['tokenized_prompt_mask'],
                dummy_input['noise'],
            )
            print(f"✅ 前向推理成功!")
            print(f"  输出形状: {output.shape}")
            print(f"  输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
            print(f"  是否有NaN: {torch.isnan(output).any().item()}")
            print(f"  是否有Inf: {torch.isinf(output).any().item()}")
        except Exception as e:
            print(f"❌ 前向推理失败: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 导出ONNX
    print(f"\n导出ONNX: {output_path}")
    print("  注意: 这可能需要几分钟...")
    
    try:
        # 使用strict=False来允许某些操作的type coercion
        torch.onnx.export(
            unrolled_model,
            (
                dummy_input['base_0_rgb'],
                dummy_input['left_wrist_0_rgb'],
                dummy_input['right_wrist_0_rgb'],
                dummy_input['state'],
                dummy_input['tokenized_prompt'],
                dummy_input['tokenized_prompt_mask'],
                dummy_input['noise'],
            ),
            output_path,
            export_params=True,
            opset_version=20,
            do_constant_folding=False,  # 避免ComplexDouble错误
            input_names=[
                'base_0_rgb',
                'left_wrist_0_rgb',
                'right_wrist_0_rgb',
                'state',
                'tokenized_prompt',
                'tokenized_prompt_mask',
                'noise',
            ],
            output_names=['actions'],
            dynamic_axes={
                'base_0_rgb': {0: 'batch'},
                'left_wrist_0_rgb': {0: 'batch'},
                'right_wrist_0_rgb': {0: 'batch'},
                'state': {0: 'batch'},
                'tokenized_prompt': {0: 'batch'},
                'tokenized_prompt_mask': {0: 'batch'},
                'noise': {0: 'batch'},
                'actions': {0: 'batch'},
            },
            verbose=False,
            use_external_data_format=True,  # 处理大模型
        )
        
        import os
        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"✅ ONNX导出成功: {output_path}")
        print(f"  文件大小: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*80}")
    print("✅ 导出完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    export_fp32_onnx_safe()
