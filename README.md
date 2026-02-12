# OpenPI-ONNX

This repository covers the OpenPI ONNX/TensorRT workflow: JAX → PyTorch conversion, INT8/FP4 quantization, and benchmark results for LIBERO tasks.

## Documentation Map

### JAX → PyTorch

- [scripts/convert_jax_to_torch.py](scripts/convert_jax_to_torch.py) - conversion entry point
- [examples/convert_jax_model_to_pytorch.py](examples/convert_jax_model_to_pytorch.py) - example conversion script

### ONNX Export & INT8 Quantization

- [docs/conversion/pi05_onnx_conversion_guide.md](docs/conversion/pi05_onnx_conversion_guide.md) - ONNX export + INT8 workflow
- [docs/conversion/norm_stats.md](docs/conversion/norm_stats.md) - normalization stats format
- [docs/quantization_calibration_explained.md](docs/quantization_calibration_explained.md) - calibration data notes

### FP4 Notes

- [docs/conversion/FP32_FP4_INT8_COMPARISON.md](docs/conversion/FP32_FP4_INT8_COMPARISON.md) - FP32/INT8/FP4 comparison notes

### Benchmarking & Results

- [benchmark_results/FP32_RESULTS_20TRIALS.md](benchmark_results/FP32_RESULTS_20TRIALS.md) - FP32 baseline results (20 trials)
- [INT8_FINAL_RESULTS.md](INT8_FINAL_RESULTS.md) - INT8 full results
- [INT8_SUMMARY.md](INT8_SUMMARY.md) - INT8 summary (accuracy/latency/VRAM)
- [benchmark_results/FP32_INT8_COMPARISON_PRELIMINARY.md](benchmark_results/FP32_INT8_COMPARISON_PRELIMINARY.md) - FP32 vs INT8 comparison

## Results Summary

| Precision | Accuracy | Latency | VRAM | Notes |
|-----------|----------|---------|------|-------|
| **FP32 (PyTorch)** | **93.75%** (750/800) | **262.41 ms mean** (P99 278.79 ms) | **8.10 GB** | Inference latency
| **INT8 (TensorRT v1)** | **98.25%** (786/800) | **10.43 s mean** (episode wall time) | **~4.95 GB** | End-to-end episode time

> INT8 latency is end-to-end episode wall time derived from tqdm logs. Inference-only latency is not logged by `eval_libero_trt_v1.py`.

### FP32 PyTorch Baseline (20 trials per task)

**Config**: 20 trials per task × 4 suites = 800 episodes | seed: 42 | date: 2026-02-11/12

| Suite | Accuracy | Success/Total | Avg Latency (ms) | Median (ms) | P99 (ms) | GPU Memory (GB) |
|-------|----------|---------------|------------------|-------------|----------|-----------------|
| **libero_spatial** | **99.5%** | 199/200 | 263.23 | 261.68 | 286.90 | 8.10 |
| **libero_goal** | **91.0%** | 182/200 | 259.49 | 258.43 | 271.99 | 8.10 |
| **libero_object** | **95.0%** | 190/200 | 264.36 | 263.70 | 283.00 | 8.10 |
| **libero_10** | **89.5%** | 179/200 | 262.56 | 262.46 | 273.26 | 8.10 |
| **Overall** | **93.75%** | **750/800** | **262.41** | **261.57** | **278.79** | **8.10** |

Full table and logs: [benchmark_results/FP32_RESULTS_20TRIALS.md](benchmark_results/FP32_RESULTS_20TRIALS.md)

### INT8 TensorRT (v1) (20 trials per task)

**Accuracy**:

| Suite | Accuracy | Success/Total |
|-------|----------|---------------|
| **libero_spatial** | **99.00%** | 198/200 |
| **libero_goal** | **98.50%** | 197/200 |
| **libero_object** | **99.50%** | 199/200 |
| **libero_10** | **96.00%** | 192/200 |
| **Overall** | **98.25%** | **786/800** |

**Latency (episode wall time)**:

| Suite | Mean (s/episode) | Median (s) | P99 (s) |
|-------|------------------|------------|---------|
| **libero_spatial** | 8.67 | 8.41 | 10.18 |
| **libero_goal** | 7.96 | 7.45 | 12.05 |
| **libero_object** | 9.35 | 9.21 | 10.26 |
| **libero_10** | 15.73 | 15.29 | 21.21 |
| **Overall** | **10.43** | **9.21** | **21.21** |

**GPU Memory**: 4954 MiB (~4.95 GB) with the engine loaded (measured via `nvidia-smi` while `serve_trt.py` is running)

Full details: [INT8_FINAL_RESULTS.md](INT8_FINAL_RESULTS.md) and [INT8_SUMMARY.md](INT8_SUMMARY.md)
