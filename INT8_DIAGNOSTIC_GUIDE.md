🚨 INT8 TensorRT 诊断指南
========================

## 问题概述

INT8模型的成功率从FP32的93.75%崩溃到约15%，所有失败都是超时。
关键线索：INT8输出的动作值范围 [-0.98, 0.87] 看起来被严重量化失真。

## 快速诊断清单

### 1️⃣ 验证INT8模型是否正确加载

```bash
# 检查TensorRT服务器是否正确加载了INT8引擎
tail -50 benchmark_logs/trt_server.log | grep -E "loading|engine|success"

# 查看服务器初始化日志
head -100 benchmark_logs/trt_server.log
```

### 2️⃣ 对比单个推理的FP32 vs INT8输出

创建诊断脚本 `diagnose_int8.py`:

```python
import numpy as np
import torch
from openpi.policies import policy_config

# 加载FP32模型
fp32_policy = policy_config.get_policy("pi05_libero", 
                                      checkpoint="checkpoints/pi05_libero_pytorch",
                                      device="cuda")

# 随机输入
dummy_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
dummy_state = np.random.randn(1, 8).astype(np.float32)
dummy_prompt_tokens = np.array([[2, 100, 200, 300]], dtype=np.int32)

# FP32推理
fp32_result = fp32_policy.infer({
    "observation/image": dummy_image,
    "observation/wrist_image": dummy_image,
    "observation/state": dummy_state,
    "observation/joint_position": dummy_state,
    "prompt": "dummy task"
})

print(f"FP32 actions range: [{fp32_result['actions'].min():.4f}, {fp32_result['actions'].max():.4f}]")
print(f"FP32 actions shape: {fp32_result['actions'].shape}")
print(f"FP32 actions stats: mean={np.mean(fp32_result['actions']):.4f}, std={np.std(fp32_result['actions']):.4f}")

# INT8推理 (通过WebSocket)
# ... 类似的推理逻辑
```

### 3️⃣ 检查归一化统计

```bash
# 查看是否正确加载了norm_stats
grep -A 2 "state_mean shape\|action_mean shape" benchmark_logs/int8_spatial_20trials.log | head -5

# 检查torch_norm_stats.json
python3 -c "
import json
with open('torch_norm_stats.json') as f:
    stats = json.load(f)
    if 'action' in stats:
        print('action_mean:', stats['action'].get('mean')[:5])
        print('action_std:', stats['action'].get('std')[:5])
"
```

### 4️⃣ 验证反归一化公式

在eval脚本中添加debug日志：

```python
# 在执行action前打印
print(f"Raw action from INT8: {actions[0, :7]}")  # 这是 [-0.98, 0.87] 范围

# 反归一化后
actions_7d = actions[:, :7] * action_std + action_mean
print(f"After unnormalize: {actions_7d[0]}")  # 应该是 [-1.5, 1.5] 范围
```

### 5️⃣ 检查INT8引擎的配置

```bash
# 列出TensorRT引擎文件信息
file checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine

# 查看引擎创建日期（是否是预期的版本）
ls -lh checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine
stat checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine
```

## 可能的原因排查表

| 症状 | 可能原因 | 检查命令 |
|------|---------|---------|
| 动作值范围错误 | 反归一化用错了统计 | 检查torch_norm_stats.json |
| 都是超时失败 | 动作无法驱动机器人 | 对比FP32 vs INT8推理输出 |
| TensorRT日志异常 | 引擎加载失败 | tail -100 benchmark_logs/trt_server.log |
| INT8输出都是NaN | 模型推理崩溃 | 检查serve_trt.py的错误处理 |

## 深度诊断

如果上面的检查没有找到问题，运行完整的诊断：

```bash
# 1. 保存一个INT8推理的完整trace
python3 -c "
import asyncio
import json
import numpy as np
from pathlib import Path

# 连接服务器，发送一个测试输入
# 记录完整的请求和响应
async def test():
    pass
"

# 2. 对比FP32和INT8的model.onnx文件结构
python3 -c "
import onnx
fp32_model = onnx.load('checkpoints/pi05_libero_onnx_compat/model.fp32.onnx')
int8_model = onnx.load('checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.onnx')

print('FP32 outputs:', [o.name for o in fp32_model.graph.output])
print('INT8 outputs:', [o.name for o in int8_model.graph.output])
"

# 3. 尝试FP32 ONNX的TensorRT版本（排除INT8问题）
python3 scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.fp32.onnx \
  --port=8017
```

## 如果INT8引擎真的有问题

### 方案A: 重新编译INT8引擎

```bash
# 需要NVIDIA ModelOpt和TensorRT
python3 -c "
from nvidia_modelopt.quantization import quantize_vit_model
from tensorrt_llm import ModelOptConfig

# 重新运行INT8量化和编译
# 参考: docs/conversion/int8_quantization_guide.md
"
```

### 方案B: 使用FP16作为过渡方案

```bash
# FP16比INT8更稳定，损失较小
python3 scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.fp16.engine \
  --port=8016
```

### 方案C: 等待量化优化

```bash
# 考虑使用更新的量化方法
# - INT4 (如果硬件支持)
# - QAT (量化感知训练)
# - 混合精度量化
```

## 记录诊断结果

运行诊断后，请更新此问题：

```markdown
## 诊断结果

**日期**: [你的诊断日期]
**发现**: [你找到的问题]
**根本原因**: [根本原因分析]
**建议的解决方案**: [建议]
```

## 参考资源

- TensorRT 文档: https://docs.nvidia.com/deeplearning/tensorrt/
- ModelOpt 文档: https://github.com/NVIDIA/TensorRT-Model-Optimizer
- INT8量化指南: [docs/conversion/int8_quantization_guide.md](../docs/conversion/int8_quantization_guide.md)

---

**优先级**: 🔴 高  
**阻塞**: INT8 vs FP32公平对比  
**预计解决时间**: 2-4小时
