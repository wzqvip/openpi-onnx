# PyTorch FP32 Benchmark - 最终完整结果

**生成时间**: 2026-02-09 11:34:00  
**模型**: pi05_libero_pytorch (13GB checkpoint)  
**评估框架**: LIBERO Benchmark  
**配置**: 每任务10次试验，seed=42

---

## 📊 总体统计

| 指标 | 数值 |
|------|------|
| **总成功率** | **93.25%** (373/400 episodes) |
| **平均延迟** | **515.22 ms** |
| **GPU显存** | **8.10 GB** (所有套件一致) |
| **测试范围** | 4个套件 × 10任务 × 10次 = 400 episodes |

---

## 🎯 各套件详细结果

| 套件 | 成功率 | 成功/总数 | 平均延迟 (ms) | 中位数延迟 (ms) | P99延迟 (ms) | GPU显存 (GB) |
|------|--------|-----------|---------------|-----------------|--------------|--------------|
| **libero_spatial** | **98.0%** | 98/100 | 266.48 | 261.04 | 304.97 | 8.10 |
| **libero_goal** | **94.0%** | 94/100 | 690.33 | 699.71 | 856.55 | 8.10 |
| **libero_object** | **96.0%** | 96/100 | 646.49 | 693.56 | 853.59 | 8.10 |
| **libero_10** | **85.0%** | 85/100 | 457.56 | 469.00 | 743.89 | 8.10 |

---

## 💡 关键发现

### 准确率分析
- ✅ **最佳表现**: libero_spatial (98%) - 空间关系任务
- ✅ **优秀表现**: libero_object (96%) - 物体操作任务
- ✅ **良好表现**: libero_goal (94%) - 目标导向任务
- ⚠️ **挑战任务**: libero_10 (85%) - 综合复杂任务

### 延迟性能
- 🚀 **最快**: libero_spatial (266ms) - 适合实时控制
- ⏱️ **中等**: libero_10 (458ms) - 可接受延迟
- 🐢 **较慢**: libero_object (646ms), libero_goal (690ms) - 需优化

### 显存占用
- 💾 **稳定**: 所有套件均为8.10GB
- 📈 **与模型一致**: 13GB checkpoint → 8.10GB 运行时显存
- ✅ **可部署**: 单卡可运行，无需多卡

---

## 🔄 与INT8对比 (计划)

### FP32 PyTorch (已完成)
- ✅ 成功率: 93.25%
- ✅ 延迟: 515ms (平均)
- ✅ 显存: 8.10GB
- ✅ 状态: 已验证工作

### INT8 TensorRT (待修复)
- ❌ 成功率: 0% (推理bug)
- ⏳ 延迟: 待测
- ⏳ 显存: 预计4-5GB
- ⚠️ 状态: 推理功能需修复

### 预期对比
修复TensorRT推理后，预期：
- 📊 **准确率**: INT8 ~90-95% (轻微损失)
- ⚡ **速度**: INT8 提升2-3倍
- 💾 **显存**: INT8 减少40-50%

---

## 📁 测试详情

### 环境信息
- Python: 3.12
- PyTorch: Latest (with venv)
- CUDA: Available
- 设备: NVIDIA GPU (8GB+ VRAM)

### 测试命令
```bash
# 运行单个套件
PYTHONPATH=/home/taco/openpi-onnx/third_party/libero:$PYTHONPATH \
python3 scripts/eval_libero_torch.py \
  --checkpoint=checkpoints/pi05_libero_pytorch \
  --config=pi05_libero \
  --task_suite_name=libero_spatial \
  --num_trials_per_task=10 \
  --seed=42
```

### 日志文件
- libero_spatial: `/home/taco/pytorch_benchmark_spatial.log`
- libero_goal: `/home/taco/pytorch_benchmark_goal.log`
- libero_object: `/home/taco/pytorch_benchmark_object.log`
- libero_10: `/home/taco/pytorch_benchmark_10.log`

---

## 🎓 结论

### 性能评估
1. **准确率稳定**: 93.25%平均成功率证明模型质量优秀
2. **延迟可控**: 除goal/object外，大部分任务<500ms
3. **显存合理**: 8.10GB适合单卡部署

### 下一步工作
1. 🔧 **修复TensorRT推理**: 参考PyTorch实现修正inference function
2. 📊 **重新测试INT8**: 修复后运行完整benchmark
3. 📈 **性能对比**: 生成FP32 vs INT8详细对比报告
4. 🚀 **优化延迟**: 分析goal/object的慢速原因

### 建议
- ✅ **生产部署**: 可使用PyTorch FP32版本
- ⏳ **量化优化**: 等待TensorRT修复后评估INT8收益
- 🎯 **针对性优化**: libero_10需要额外关注和改进
