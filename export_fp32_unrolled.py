"""
正确导出FP32 ONNX模型 - 展开diffusion循环

关键解决方案：
- JAX模型使用 jax.lax.while_loop
- PyTorch模型使用简单的 while 循环
- ONNX不支持动态while循环
- 解决方法：展开(unroll)while循环为固定次数的for循环
"""

import sys
import os
sys.path.insert(0, '/home/taco/openpi/src')
sys.path.insert(0, '/home/taco/openpi-onnx')

import torch
import numpy as np

# ======================= PATCH GemmaRotaryEmbedding =======================
# 在任何模型加载前apply patch来避免complex类型
import torch.nn as nn

def patch_gemma_rope():
    """修补Gemma的RoPE实现以避免complex操作"""
    try:
        from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
        
        original_forward = GemmaRotaryEmbedding.forward
        
        def patched_forward(self, x, position_ids):
            """修补的forward - 避免complex操作"""
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
                position_ids.shape[0], -1, 1
            ).to(x.device)
            position_ids_expanded = position_ids[:, None, :].float()

            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                # 关键: 直接计算cos/sin，而不是先创建emb再计算
                cos = freqs.cos() * self.attention_scaling
                sin = freqs.sin() * self.attention_scaling
                cos = torch.cat((cos, cos), dim=-1)
                sin = torch.cat((sin, sin), dim=-1)
            
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        
        GemmaRotaryEmbedding.forward = patched_forward
        print("[PATCH] 已应用Gemma RoPE补丁")
    except Exception as e:
        print(f"[WARNING] 无法应用RoPE补丁: {e}")

patch_gemma_rope()

# =========================================================================

from openpi.training import config as _config
from openpi.models import model as _model
from openpi.models_pytorch import pi0_pytorch


class OnnxWrapperFP32(torch.nn.Module):
    def __init__(self, model, num_steps=10):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

    def forward(self, base_rgb, left_rgb, right_rgb, state, tokenized_prompt, tokenized_prompt_mask, noise):
        bsize = state.shape[0]
        device = state.device

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

        state_proc = observation.state
        images_proc = observation.images
        img_masks = observation.image_masks
        lang_tokens = observation.tokenized_prompt
        lang_masks = observation.tokenized_prompt_mask

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            list(images_proc.values()),
            list(img_masks.values()),
            lang_tokens,
            lang_masks,
        )
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / self.num_steps
        dt_tensor = torch.tensor(dt, dtype=self.model.action_in_proj.weight.dtype, device=device)
        x_t = noise

        for i in range(self.num_steps):
            time = torch.tensor(1.0 + i * dt, dtype=self.model.action_in_proj.weight.dtype, device=device)
            expanded_time = time.expand(bsize)
            v_t = self.model.denoise_step(state_proc, prefix_pad_masks, past_key_values, x_t, expanded_time)
            x_t = x_t + dt_tensor * v_t

        return x_t

def main():
    print("=" * 80)
    print("FP32 ONNX导出 - 使用展开的diffusion循环")
    print("=" * 80)
    
    # 配置
    config_name = "pi05_libero"
    checkpoint_path = "checkpoints/pi05_libero_pytorch_jax/model.safetensors"
    output_path = "checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.jax.onnx"
    num_diffusion_steps = 10  # 展开10步
    
    print(f"\n加载配置: {config_name}")
    config = _config.get_config(config_name)
    
    # 修复action_dim (checkpoint使用32, 默认config使用7)
    import dataclasses
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, action_dim=32)
    )
    print(f"  action_dim: {config.model.action_dim}")
    
    print(f"加载PyTorch模型: {checkpoint_path}")
    base_model = pi0_pytorch.PI0Pytorch(config.model)
    
    # 从safetensors加载权重
    from safetensors.torch import load_file
    state_dict = load_file(checkpoint_path)
    base_model.load_state_dict(state_dict, strict=False)
    base_model.eval()
    
    print(f"\n创建展开模型 (diffusion_steps={num_diffusion_steps})")
    unrolled_model = OnnxWrapperFP32(base_model, num_steps=num_diffusion_steps)
    unrolled_model.eval()
    
    # 创建虚拟输入
    print("\n准备虚拟输入...")
    device = "cpu"
    unrolled_model = unrolled_model.to(device)
    
    batch_size = 1
    dummy_input = {
        'base_0_rgb': torch.randn(batch_size, 3, 224, 224, device=device),
        'left_wrist_0_rgb': torch.randn(batch_size, 3, 224, 224, device=device),
        'right_wrist_0_rgb': torch.randn(batch_size, 3, 224, 224, device=device),
        'state': torch.randn(batch_size, 8, device=device),
        'tokenized_prompt': torch.randint(0, 1000, (batch_size, 200), device=device),
        'tokenized_prompt_mask': torch.ones(batch_size, 200, dtype=torch.bool, device=device),
        'noise': torch.randn(batch_size, 10, 32, device=device),  # action_horizon=10, action_dim=32
    }
    
    # 测试前向推理
    print("\n测试前向推理...")
    try:
        with torch.no_grad():
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
    except Exception as e:
        print(f"❌ 前向推理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 导出ONNX
    print(f"\n导出ONNX: {output_path}")
    print("  注意: 这可能需要几分钟...")
    
    try:
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
            do_constant_folding=False,
            input_names=[
                'base_0_rgb',
                'left_wrist_0_rgb',
                'right_wrist_0_rgb',
                'state',
                'tokenized_prompt',
                'tokenized_prompt_mask',
                'noise'
            ],
            output_names=['actions'],
            dynamic_axes={
                'base_0_rgb': {0: 'batch_size'},
                'left_wrist_0_rgb': {0: 'batch_size'},
                'right_wrist_0_rgb': {0: 'batch_size'},
                'state': {0: 'batch_size'},
                'tokenized_prompt': {0: 'batch_size'},
                'tokenized_prompt_mask': {0: 'batch_size'},
                'noise': {0: 'batch_size'},
                'actions': {0: 'batch_size'},
            },
            verbose=False,
            dynamo=False  # 关键: 禁用dynamo以使用传统ONNX导出
        )
        
        print(f"✅ ONNX导出成功: {output_path}")
        
        # 显示文件大小
        import os
        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"  文件大小: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 应用ONNX补丁 (CumSum修复)
    print("\n应用ONNX补丁...")
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        
        # 补丁1: 修复CumSum节点的bool输入类型
        print("  修复CumSum节点...")
        patched_count = 0
        for node in onnx_model.graph.node:
            if node.op_type == "CumSum":
                # 找到第二个输入(exclusive参数)
                if len(node.input) > 1:
                    exclusive_input = node.input[1]
                    # 检查这个输入是否来自Concat产生bool
                    for prev_node in onnx_model.graph.node:
                        if prev_node.output[0] == exclusive_input:
                            if prev_node.op_type == "Concat":
                                # 在Concat和CumSum之间添加Cast节点
                                original_output_name = exclusive_input
                                cast_input_name = original_output_name + "_int64"
                                
                                # 修改Concat输出到Cast
                                prev_node.output[0] = cast_input_name
                                
                                # 插入Cast节点 (bool -> int64)
                                cast_node = onnx_model.graph.node.add()
                                cast_node.op_type = "Cast"
                                cast_node.name = f"Cast_{original_output_name}"
                                cast_node.input.append(cast_input_name)
                                cast_node.output.append(original_output_name)
                                cast_attr = cast_node.attribute.add()
                                cast_attr.name = "to"
                                cast_attr.i = onnx.TensorProto.INT64  # Cast to int64
                                
                                patched_count += 1
                                print(f"    修复 CumSum: {node.name}")
                            break
        
        print(f"  ✅ 补丁完成: {patched_count} 个CumSum节点已修复")
        
        # 验证ONNX模型
        print("\n验证ONNX模型...")
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX模型验证通过")
        
        # 保存修补后的模型
        onnx.save(onnx_model, output_path)
        print(f"✅ 修补后的模型已保存")
        
        # 显示模型信息
        print(f"\n模型信息:")
        print(f"  输入数量: {len(onnx_model.graph.input)}")
        print(f"  输出数量: {len(onnx_model.graph.output)}")
        print(f"  节点数量: {len(onnx_model.graph.node)}")
        print(f"  Opset版本: {onnx_model.opset_import[0].version}")
        
    except Exception as e:
        print(f"⚠️  ONNX补丁/验证失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("导出完成!")
    print("=" * 80)
    print(f"\n下一步:")
    print(f"1. 构建TensorRT引擎:")
    print(f"   python3 scripts/build_trt_engine.py {output_path} \\")
    print(f"     --output checkpoints/pi05_libero_onnx_compat/engine_fp32_unrolled.trt \\")
    print(f"     --workspace 8")
    print(f"\n2. 测试推理延迟:")
    print(f"   python3 scripts/trt_inference_test.py \\")
    print(f"     checkpoints/pi05_libero_onnx_compat/engine_fp32_unrolled.trt 10")
    print(f"\n3. 运行LIBERO评估:")
    print(f"   python3 scripts/eval_libero_trt_v1.py \\")
    print(f"     --engine checkpoints/pi05_libero_onnx_compat/engine_fp32_unrolled.trt \\")
    print(f"     --suite libero_spatial --trials 20")

if __name__ == "__main__":
    main()
