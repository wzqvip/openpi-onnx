# 📋 FP8 问题诊断文件索引

## 📚 文档导航

### 🚀 快速入门 (从这里开始)

**文件**: [FP8_QUICK_SUMMARY.md](FP8_QUICK_SUMMARY.md)
- **用途**: 5 分钟快速了解问题和解决方案
- **内容**: 核心问题、三种方案对比、立即行动步骤
- **适合**: 想快速上手的用户

### 🔍 详细诊断

**文件**: [FP8_DIAGNOSTIC_REPORT.md](FP8_DIAGNOSTIC_REPORT.md)
- **用途**: 完整的技术分析和诊断
- **内容**: 问题分析、根本原因、故障排查、性能预测
- **适合**: 想深入理解问题的开发者

### ⚙️ 升级验证脚本

**文件**: [upgrade_pytorch_and_test_fp8.py](upgrade_pytorch_and_test_fp8.py)
- **用途**: 自动化升级和 FP8 支持验证
- **使用**: `python3 upgrade_pytorch_and_test_fp8.py`
- **功能**: 
  - 检查 PyTorch 版本
  - 显示升级指令
  - 测试 FP8 基础操作
  - 推荐后续步骤

### 🧪 评估脚本

**文件**: `verify_fp8_libero.py`
- **用途**: FP8 模型在 LIBERO 上的推理验证
- **使用**: `python3 verify_fp8_libero.py`

---

## 🎯 快速决策流程

```
开始
  ↓
【第 1 步】阅读 FP8_QUICK_SUMMARY.md (5分钟)
  ↓
【第 2 步】运行 upgrade_pytorch_and_test_fp8.py (30分钟)
  ↓
  ├─ FP8 支持改进 (≥75%)
  │   └─ 使用 FP8: python3 verify_fp8_libero.py
  │
  └─ FP8 支持不足 (<75%)
      └─ 使用 INT8: python3 scripts/eval_libero_trt_v1.py
```

---

## 📊 问题速查表

### 错误: "sum_cpu" not implemented for 'Float8_e4m3fn'

| 症状 | 原因 | 解决 |
|------|------|------|
| FP8 推理失败 | PyTorch 2.7.1 FP8 算子不完整 | 升级到 2.9.1 |
| 其他 FP8 错误 | 类似原因 | 升级 PyTorch |
| 升级后仍失败 | PyTorch 2.9.1 仍不支持 | 使用 INT8 |

### 文件大小比较

| 格式 | 大小 | 压缩率 | 状态 |
|------|------|--------|------|
| FP32 | 8.29 GB | 100% | 基准 |
| INT8 | 4.67 GB | 56% | ✅ 可用 |
| FP8 | 4.14 GB | 50% | ⚠️ 推理有问题 |

### 性能指标

| 指标 | FP32 | INT8 | FP8 |
|------|------|------|-----|
| 加载时间 | 3.66s | 2.29s | 1.22s ⚡ |
| LIBERO 成功率 | 100% | 98% ✅ | 未知 |
| 推理速度 | 5.17ms | 2.04ms ⚡ | 未知 |

---

## 🔧 故障排查指南

### 问题 1: PyTorch 版本太旧

**症状**: `unsupported operand type(s) for /: 'float' and 'float8_e4m3fn'`

**解决**:
```bash
pip install --upgrade 'torch>=2.9.1' \
  -i https://pypi.jetson-ai-lab.io/sbsa/cu130/
```

### 问题 2: 升级后仍有 FP8 错误

**症状**: 同样的 FP8 算子错误

**检查**:
```bash
# 验证升级是否完成
python3 -c "import torch; print(torch.__version__)"

# 清除缓存并重新升级
pip cache purge
pip install --force-reinstall torch>=2.9.1 \
  -i https://pypi.jetson-ai-lab.io/sbsa/cu130/
```

### 问题 3: INT8 推理失败

**症状**: INT8 模型加载后推理出错

**解决**:
1. 检查模型路径: `ls checkpoints/pi05_libero_pytorch_int8_dynamic/`
2. 检查依赖: `python3 -c "import torch; import safetensors"`
3. 重新量化: `python3 quantize_pytorch_dynamic_int8.py`

---

## 📈 建议路径

### 保守方案 (优先可靠性)
```
1. 使用现有 INT8 模型
   └─ python3 scripts/eval_libero_trt_v1.py
2. 验证 98% 成功率
3. 构建 TensorRT 引擎
   └─ python3 scripts/build_trt_engine.py
```
**耗时**: 2-3 小时 | **风险**: 低 | **收益**: 高

### 激进方案 (优先最优性)
```
1. 升级 PyTorch 2.9.1
   └─ pip install --upgrade torch>=2.9.1
2. 测试 FP8 支持
   └─ python3 upgrade_pytorch_and_test_fp8.py
3. 如果成功，使用 FP8
   └─ python3 verify_fp8_libero.py
4. 如果失败，降级到 INT8
   └─ python3 scripts/eval_libero_trt_v1.py
```
**耗时**: 30分钟 (测试) + 2-3小时 (评估) | **风险**: 低 | **收益**: 极高

### 快速方案 (优先速度)
```
1. 立即部署 INT8
   └─ 使用现有 checkpoints/pi05_libero_pytorch_int8_dynamic
2. 构建 TensorRT
   └─ python3 scripts/build_trt_engine.py
3. 启动服务
   └─ inference_server --model trt_int8
```
**耗时**: 1 小时 | **风险**: 低 | **收益**: 高

---

## 📞 获取帮助

### 如果还有疑问

1. **查看完整诊断**: `cat FP8_DIAGNOSTIC_REPORT.md | less`
2. **运行诊断脚本**: `python3 upgrade_pytorch_and_test_fp8.py`
3. **检查日志**: 查看上面脚本的输出

### 关键技术点

- **FP8 (float8_e4m3fn)**: NVIDIA 8 位浮点格式，实验性功能
- **INT8**: 标准 8 位整数量化，生产级别
- **TensorRT**: NVIDIA 的推理优化框架，支持 INT8

---

## 🎓 推荐阅读顺序

1. **首次接触** → 阅读 `FP8_QUICK_SUMMARY.md`
2. **想了解细节** → 阅读 `FP8_DIAGNOSTIC_REPORT.md`
3. **准备升级** → 运行 `upgrade_pytorch_and_test_fp8.py`
4. **选择方案** → 根据输出选择 FP8 或 INT8
5. **开始评估** → 运行相应的评估脚本

---

**最后更新**: 2026-02-15  
**作者**: AI Assistant  
**版本**: 2.0 (包含 PyTorch 2.9.1 升级指南)

---

## 📂 相关文件位置

```
/home/taco/openpi-onnx/
├── FP8_QUICK_SUMMARY.md              ← 从这里开始
├── FP8_DIAGNOSTIC_REPORT.md          ← 完整诊断
├── FP8_WORKFLOW_QUICKSTART.md        ← 工作流快速开始
├── upgrade_pytorch_and_test_fp8.py   ← 升级验证脚本
├── verify_fp8_libero.py              ← FP8 评估
├── verify_mse_lite.py                ← INT8 精度验证
├── checkpoints/
│   ├── pi05_libero_pytorch_fp8/      ← FP8 模型
│   ├── pi05_libero_pytorch_int8_dynamic/  ← INT8 模型 ✅
│   └── pi05_libero_pytorch_jax/      ← FP32 基准
└── scripts/
    ├── eval_libero_trt_v1.py         ← LIBERO 评估
    └── build_trt_engine.py           ← TensorRT 构建
```

---

💡 **建议的下一步**: 运行以下命令
```bash
cd /home/taco/openpi-onnx && python3 upgrade_pytorch_and_test_fp8.py
```
