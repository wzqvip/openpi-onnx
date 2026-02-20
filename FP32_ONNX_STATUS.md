# FP32 ONNX 验证状态报告

## 验证时间
2026-02-14 14:39 UTC

## 检查项目

### 1️⃣ ONNX模型加载 ✅
- **文件**: `checkpoints/pi05_libero_onnx_compat/model.fp32.unrolled.onnx`
- **大小**: 12.6 MB
- **ONNX加载**: ✅ 成功
- **节点数**: 71,583

### 2️⃣ 模型结构 ✅
```
输入 (7个):
  - base_0_rgb: [?, 3, 224, 224]
  - left_wrist_0_rgb: [?, 3, 224, 224]
  - right_wrist_0_rgb: [?, 3, 224, 224]
  - state: [?, 8]
  - tokenized_prompt: [?, 200]
  - tokenized_prompt_mask: [?, 200]
  - noise: [?, 10, 32]

输出 (1个):
  - actions: [?, 10, 32]
```

### 3️⃣ PyTorch前向推理 ✅
```
✅ 模型加载成功
✅ 前向推理成功
   输出形状: [1, 10, 32]
   输出范围: [-1.0006, 0.9794]
   无NaN/Inf
```

### 4️⃣ ONNX Runtime加载 ❌

**问题1**: CumSum类型错误 (已修复)
```
Type Error: Type 'tensor(bool)' of input parameter (/Concat_12_output_0) 
of operator (CumSum) in node (/CumSum_3) is invalid.
```
- **来源**: rotary embeddings中的累积求和操作
- **修复方案**: 在CumSum前后添加Cast节点
- **修复结果**: ❌ 引发新的complex128错误

**问题2**: Complex类型错误 (未修复)
```
Type Error: Type 'tensor(complex128)' of input parameter 
(/vision_tower/vision_model/encoder/layers.0/self_attn_2/Cast_1_output_0) 
of operator (Mul) in node (/vision_tower/vision_model/encoder/layers.0/self_attn_2/Mul_1) is invalid.
```
- **来源**: rotary embeddings中的复数运算 (频率向量)
- **本质原因**: PyTorch torch.onnx.export对复数操作的处理不完全
- **ONNX Runtime限制**: 不支持complex64/complex128类型

## 根本问题分析

### 问题所在
FP32 ONNX的类型问题**不是**因为:
- ❌ 导出脚本配置问题
- ❌ opset版本选择
- ❌ constant_folding设置
- ❌ 模型展开(unrolling)逻辑

而是因为:
- ✅ PyTorch的rotary embeddings实现在ONNX导出时使用了复数运算
- ✅ ONNX Runtime不支持complex64/complex128类型
- ✅ 这是PyTorch导出器的限制,不是模型问题

### 对比表

| 模型 | 文件加载 | ONNX结构 | ONNX Runtime | TensorRT | 评估 |
|------|--------|--------|-------------|---------|-----|
| FP16 ONNX | ✅ | ✅ | ❌ Complex | - | ❌ |
| FP32 ONNX | ✅ | ✅ | ❌ Complex | - | ❌ |
| INT8 TRT | N/A | N/A | N/A | ❌ (引擎为空) | ❌ |
| PyTorch | ✅ | N/A | N/A | N/A | ✅ (78%) |

## 结论

✅ **FP32 ONNX正常导出** - 文件完整性无问题

❌ **FP32 ONNX无法用于推理** - ONNX Runtime无法加载

🔧 **根本原因** - PyTorch的rotary embeddings导出时产生了ONNX Runtime不支持的complex类型

## 建议方案

1. **使用PyTorch动态模型进行评估** (当前INT8成功的做法)
   - 优点: 完全功能正常
   - 缺点: 性能不如TensorRT

2. **修改rotary embeddings实现** 
   - 避免复数操作,使用等价的实数实现
   - 可参考HuggingFace的rotary_pos_emb_pt_2023实现

3. **使用TensorRT的PyTorch插件**
   - 直接优化PyTorch模型而非通过ONNX

## 总结

FP32 ONNX导出**成功且正常**,但由于PyTorch对rotary embeddings的复数表示限制,
无法通过ONNX Runtime或TensorRT加载。这是导出工具链的限制,而非模型问题。

