# Quick Reference - FP32 vs INT8 Benchmark

## One-Command Full Benchmark

```bash
bash scripts/run_full_benchmark.sh
```

**Time**: 4-6 hours | **Output**: `./benchmark_results/`

---

## Manual Testing

### Start Server (Choose One)

**FP32 (13 GB, Fast)**
```bash
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.fp32.modelopt.engine \
  --port=8012 &
```

**INT8 (4.6 GB, Small)**
```bash
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine \
  --port=8012 &
```

### Run Benchmark

```bash
sleep 5  # Wait for server
python scripts/benchmark_trt_models.py \
  --model_type=fp32 \
  --num_trials=10 \
  --task_suite_name=all \
  --port=8012
```

Replace `fp32` with `int8` for INT8 testing.

---

## Results

Results saved to: `benchmark_results/benchmark_{model}_{trials}trials.json`

View summary:
```bash
python scripts/generate_comparison_report.py ./benchmark_results/
```

---

## Expected Performance

| Metric | FP32 | INT8 |
|--------|------|------|
| Success Rate | ~98% | 96.88% ✅ |
| Avg Latency | ~85ms | ~120ms |
| Engine Size | 13 GB | 4.6 GB |
| Efficiency | Baseline | 65% smaller |

---

## Documentation

- **BENCHMARK_GUIDE.md** - Complete guide
- **docs/conversion/FP32_FP4_INT8_COMPARISON.md** - Technical details
- **INT8_SUMMARY.md** - Quantization theory

---

**Updated**: Feb 2026 | **Branch**: INT8
