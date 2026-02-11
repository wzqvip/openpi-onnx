# PyTorch FP32 Benchmark - 快速参考

⚡ **一页纸总结** - 所有关键数据一目了然

---

## 📊 核心数据

| 指标 | 数值 |
|------|------|
| 🎯 **总成功率** | **93.25%** (373/400) |
| ⚡ **平均延迟** | **515 ms** |
| 💾 **GPU显存** | **8.10 GB** |
| ✅ **状态** | 生产就绪 |

---

## 📈 各套件表现

```
libero_spatial: 98% ████████████████████ (266ms)  ⚡最快
libero_object:  96% ███████████████████  (646ms)
libero_goal:    94% ██████████████████   (690ms)  🐢最慢  
libero_10:      85% █████████████████    (458ms)  ⚠️最难
```

---

## 🎯 对比INT8 (待测)

| 项目 | FP32 ✅ | INT8 ⏳ |
|------|---------|---------|
| 成功率 | 93% | 待修复 |
| 延迟 | 515ms | 预期~200ms |
| 显存 | 8.1GB | 预期~4GB |

---

## 🚀 快速命令

**查看完整结果**:
```bash
cat ~/PYTORCH_FP32_FINAL_RESULTS.md
```

**查看对比报告**:
```bash
cat ~/FP32_INT8_COMPARISON.md
```

**重新运行测试**:
```bash
cd /home/taco && source .venv/bin/activate
PYTHONPATH=openpi-onnx/third_party/libero:$PYTHONPATH \
python3 openpi-onnx/scripts/eval_libero_torch.py \
  --checkpoint=checkpoints/pi05_libero_pytorch \
  --config=pi05_libero \
  --task_suite_name=libero_spatial \
  --num_trials_per_task=10
```

---

## 📁 文档索引

| 文档 | 说明 |
|------|------|
| [PYTORCH_FP32_FINAL_RESULTS.md](PYTORCH_FP32_FINAL_RESULTS.md) | 完整测试结果 |
| [FP32_INT8_COMPARISON.md](FP32_INT8_COMPARISON.md) | FP32 vs INT8对比 |
| [openpi-onnx/README.md](openpi-onnx/README.md) | 项目主文档 |
| [PYTORCH_FP32_PROGRESS.md](PYTORCH_FP32_PROGRESS.md) | 进度追踪 |

---

**生成时间**: 2026-02-09 | **模型**: pi05_libero_pytorch
