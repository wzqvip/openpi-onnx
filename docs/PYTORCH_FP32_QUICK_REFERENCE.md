# PyTorch FP32 Benchmark - Quick Reference

⚡ **One-page summary** - key results at a glance

---

## 📊 Core Metrics

| Metric | Value |
|--------|-------|
| 🎯 **Overall success rate** | **93.25%** (373/400) |
| ⚡ **Mean latency** | **515 ms** |
| 💾 **GPU memory** | **8.10 GB** |
| ✅ **Status** | Production ready |

---

## 📈 By Suite

```
libero_spatial: 98% ████████████████████ (266ms)  ⚡fastest
libero_object:  96% ███████████████████  (646ms)
libero_goal:    94% ██████████████████   (690ms)  🐢slowest  
libero_10:      85% █████████████████    (458ms)  ⚠️hardest
```

---

## 🎯 INT8 Comparison

| Item | FP32 ✅ | INT8 ✅ |
|------|---------|---------|
| Success rate | 93.25% | 98.25% |
| Latency | 515 ms (inference) | 10.43 s mean (episode wall time) |
| GPU memory | 8.10 GB | ~4.95 GB |

> INT8 latency is end-to-end episode wall time from tqdm logs; inference-only latency is not logged.

---

## 🚀 Quick Commands

**View full results**:
```bash
cat ~/PYTORCH_FP32_FINAL_RESULTS.md
```

**View comparison report**:
```bash
cat ~/FP32_INT8_COMPARISON.md
```

**Re-run the benchmark**:
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

## 📁 Document Index

| Document | Description |
|----------|-------------|
| [PYTORCH_FP32_FINAL_RESULTS.md](PYTORCH_FP32_FINAL_RESULTS.md) | Full benchmark results |
| [FP32_INT8_COMPARISON.md](FP32_INT8_COMPARISON.md) | FP32 vs INT8 comparison |
| [openpi-onnx/README.md](openpi-onnx/README.md) | Project README |
| [PYTORCH_FP32_PROGRESS.md](PYTORCH_FP32_PROGRESS.md) | Progress tracking |

---

**Generated**: 2026-02-09 | **Model**: pi05_libero_pytorch
