# PyTorch FP32 Benchmark - Final Results

**Generated**: 2026-02-09 11:34:00  
**Model**: pi05_libero_pytorch (13GB checkpoint)  
**Benchmark**: LIBERO  
**Config**: 10 trials per task, seed=42

---

## 📊 Overall Summary

| Metric | Value |
|--------|-------|
| **Overall success rate** | **93.25%** (373/400 episodes) |
| **Mean latency** | **515.22 ms** |
| **GPU memory** | **8.10 GB** (consistent across suites) |
| **Scope** | 4 suites × 10 tasks × 10 trials = 400 episodes |

---

## 🎯 Per-Suite Results

| Suite | Success rate | Success/Total | Mean latency (ms) | Median latency (ms) | P99 latency (ms) | GPU memory (GB) |
|------|--------------|---------------|-------------------|---------------------|------------------|-----------------|
| **libero_spatial** | **98.0%** | 98/100 | 266.48 | 261.04 | 304.97 | 8.10 |
| **libero_goal** | **94.0%** | 94/100 | 690.33 | 699.71 | 856.55 | 8.10 |
| **libero_object** | **96.0%** | 96/100 | 646.49 | 693.56 | 853.59 | 8.10 |
| **libero_10** | **85.0%** | 85/100 | 457.56 | 469.00 | 743.89 | 8.10 |

---

## 💡 Key Findings

### Accuracy
- ✅ **Best**: libero_spatial (98%) - spatial reasoning
- ✅ **Strong**: libero_object (96%) - object manipulation
- ✅ **Good**: libero_goal (94%) - goal-oriented tasks
- ⚠️ **Challenging**: libero_10 (85%) - complex tasks

### Latency
- 🚀 **Fastest**: libero_spatial (266ms) - suitable for real-time control
- ⏱️ **Moderate**: libero_10 (458ms) - acceptable latency
- 🐢 **Slowest**: libero_object (646ms), libero_goal (690ms) - needs optimization

### GPU Memory
- 💾 **Stable**: 8.10GB across suites
- 📈 **Model-consistent**: 13GB checkpoint → 8.10GB runtime VRAM
- ✅ **Deployable**: single-GPU friendly

---

## 🔄 INT8 Comparison (Final)

### FP32 PyTorch (completed)
- ✅ Success rate: 93.25%
- ✅ Latency: 515ms (mean)
- ✅ GPU memory: 8.10GB
- ✅ Status: verified

### INT8 TensorRT (completed)
- ✅ Success rate: 98.25%
- ✅ Latency: ~162 ms mean inference time (P99 ~167 ms)
- ✅ GPU memory: ~4.95GB
- ✅ Status: verified (v1 path)

### Notes
- INT8 latency above is inference latency reported by evaluation logs.

---

## 📁 Benchmark Details

### Environment
- Python: 3.12
- PyTorch: Latest (with venv)
- CUDA: Available
- Device: NVIDIA GPU (8GB+ VRAM)

### Command
```bash
# Run a single suite
PYTHONPATH=/home/taco/openpi-onnx/third_party/libero:$PYTHONPATH \
python3 scripts/eval_libero_torch.py \
  --checkpoint=checkpoints/pi05_libero_pytorch \
  --config=pi05_libero \
  --task_suite_name=libero_spatial \
  --num_trials_per_task=10 \
  --seed=42
```

### Log Files
- libero_spatial: `/home/taco/pytorch_benchmark_spatial.log`
- libero_goal: `/home/taco/pytorch_benchmark_goal.log`
- libero_object: `/home/taco/pytorch_benchmark_object.log`
- libero_10: `/home/taco/pytorch_benchmark_10.log`

---

## 🎓 Conclusion

### Performance Summary
1. **Stable accuracy**: 93.25% average success rate confirms model quality.
2. **Manageable latency**: most tasks <500ms, except goal/object.
3. **Reasonable VRAM**: 8.10GB fits a single GPU.

### Next Steps
1. 📊 **Publish comparison**: keep FP32 vs INT8 comparison in sync.
2. 🚀 **Optimize latency**: analyze goal/object slowdowns.
3. 📈 **Profile inference**: add inference-only timing.
4. 🧪 **Optional quantization**: explore INT4 if needed.

### Recommendations
- ✅ **Production**: FP32 PyTorch is safe to deploy.
- ✅ **Quantization**: INT8 v1 path is stable; monitor quality over time.
- 🎯 **Focus**: libero_10 still deserves additional analysis.
