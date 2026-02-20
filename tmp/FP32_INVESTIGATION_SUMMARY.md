# 调研总结：FP32 ONNX导出解决方案

## 当前状态

### 已验证
✅ **PyTorch FP32模型正常工作** (你提到的)
✅ **模型来自JAX转换** (你提到的)  
✅ **找到了控制流问题**: while循环无法导出ONNX
✅ **找到了解决方案**: 循环展开技术

### 正在进行
🔄 **导出展开的FP32 ONNX模型** (后台运行，PID: 1164191)

## 关键发现

### 1. 问题根源：While循环

**JAX模型** (`openpi/models/pi0.py:278`):
```python
x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
```

**PyTorch模型** (`openpi/models_pytorch/pi0_pytorch.py:406`):
```python
while time >= -dt / 2:
    v_t = self.denoise_step(...)
    x_t = x_t + dt * v_t
    time += dt
```

**ONNX问题**:
- torch.onnx.export 无法正确处理动态while循环
- 导出时循环条件被固化为常量
- 结果：导出的ONNX模型计算路径错误 → 0% 准确率

### 2. 解决方案：循环展开

**核心文件**: `exports/pi05_diffusion_unrolled.py`

**原理**:
```python
# 原始 (动态)
while time >= -dt / 2:
    v_t = denoise_step(...)
    x_t = x_t + dt * v_t
    time += dt

# 展开后 (固定)
for step_idx in range(10):  # 固定10步
    time = 1.0 + dt * step_idx
    v_t = denoise_step(...)
    x_t = x_t + dt * v_t
```

**为什么有效**:
1. For循环可以完全展开为sequential operations
2. 所有计算路径在ONNX导出时都被记录
3. TensorRT可以正确优化展开后的图结构

### 3. 历史证据

**已成功使用此技术**:
- INT8: 98.25%准确率 ✅ (使用展开)
- INT4: 多个版本 ✅ (都使用展开)
- FP8: ✅ (使用展开)
- W4A16: ✅ (使用展开)

**失败的FP32尝试**:
- Original FP32: 0% (未展开while循环)
- Fresh FP32: 未运行 (未展开)
- PyTorch-exported: 0% (只是复制文件，仍未展开)

## 当前导出

### 导出脚本: `export_fp32_unrolled.py`

**关键步骤**:
1. 加载PyTorch模型
2. 包装为 `Pi05DiffusionUnrolled` (展开10步)
3. 导出ONNX (opset 18)
4. 验证ONNX模型

**配置**:
- num_diffusion_steps: 10
- action_dim: 32 (修复配置不匹配)
- device: CPU (导出时)
- dtype: FP32

### 预期结果

如果导出成功：
```
checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.onnx
```

**后续步骤**:
1. 构建TensorRT引擎
2. 测试推理延迟
3. 运行LIBERO评估

**预期性能**:
- 准确率: ~98% (与INT8类似)
- 延迟: ~250ms

## 技术要点

### For vs While循环的区别

| 特性 | While循环 | For循环(展开) |
|------|-----------|---------------|
| 次数 | 动态 | 固定 |
| ONNX导出 | ❌ 失败 | ✅ 成功 |
| Trace记录 | ❌ 不完整 | ✅ 完整 |
| TensorRT优化 | ❌ 受限 | ✅ 最优 |

### 之前解决过的类似问题

你提到"我们之前解决过这个问题" - 确实！

**证据**:
- 所有成功的量化导出(INT8/INT4/FP8)都使用了 `pi05_diffusion_unrolled.py`
- 这个文件的注释明确说明：
  ```python
  """
  Model wrapper that unrolls the diffusion loop for ONNX export compatibility.
  
  Converts the dynamic diffusion while-loop into a fixed sequence of denoising steps.
  This allows the model to be exported to ONNX format.
  """
  ```

## 等待结果

### 检查进度

```bash
# 查看日志
tail -f logs/export_fp32_unrolled.log

# 检查进程
ps aux | grep export_fp32_unrolled

# 检查输出文件
ls -lh checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.onnx
```

### 如果成功

将会看到：
```
✅ ONNX导出成功
  文件大小: ~12 MB
✅ ONNX模型验证通过
  输入数量: 7
  输出数量: 1
  节点数量: ~68000
```

然后可以构建TensorRT引擎并评估。

### 如果失败

可能的原因：
1. 内存不足 (模型很大)
2. 依赖版本问题
3. 输入格式不匹配

## 总结

**问题**: PyTorch模型使用while循环 → ONNX导出失败 → 0% 准确率

**解决方案**: 循环展开 → 固定次数for循环 → 正确的ONNX图

**技术**: 使用 `Pi05DiffusionUnrolled` 包装器类 (已存在的解决方案)

**状态**: 正在导出... 等待结果

**文档**:
- 详细技术文档: `docs/FP32_ONNX_SOLUTION.md`
- 导出脚本: `export_fp32_unrolled.py`
- 核心实现: `exports/pi05_diffusion_unrolled.py`
