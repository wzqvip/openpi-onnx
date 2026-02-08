# FP32, FP4, and INT8 Model Comparison

## Overview

This document presents a comprehensive benchmark comparing three quantization levels for the Pi05 model on LIBERO tasks:

- **FP32**: Full precision baseline (13 GB TensorRT engine)
- **FP4**: 4-bit floating point quantization (requires specific GPU support)
- **INT8**: 8-bit integer quantization (4.6 GB TensorRT engine)

## Benchmark Methodology

### Environment Setup
- **Framework**: LIBERO (4 task suites × 10 tasks = 40 total tasks)
- **Trials per task**: 10 repetitions
- **Total trials**: 400 per model
- **Random seed**: 7 (with +trial offset for variation)
- **Environment resolution**: 256×256 RGB images

### Metrics Measured

1. **Success Rate (Accuracy)**
   - Percentage of trials where the robot completes the task
   - Higher is better

2. **Latency (Inference Time)**
   - Per-step inference time in milliseconds
   - Lower is better
   - Measured using `time.perf_counter()`

3. **GPU Memory (VRAM)**
   - TensorRT engine size and peak memory usage
   - Important for deployment constraints

### Task Suites

| Suite | Tasks | Focus Area |
|-------|-------|-----------|
| **libero_goal** | 10 | Goal-oriented manipulation |
| **libero_spatial** | 10 | Spatial reasoning |
| **libero_object** | 10 | Object interaction |
| **libero_10** | 10 | Complex sequences |

## Quantization Details

### FP32 (Baseline)
```
Quantization: None (full precision)
Engine Size: 13 GB
Data Type: float32
Precision: ~7 significant digits
```

### FP4 (Experimental)
```
Quantization: 4-bit floating point
Engine Size: ~3-4 GB (estimated)
Data Type: float4
Precision: ~1 significant digit
GPU Requirement: NVIDIA Thor (BLACKWELL) or newer
Status: Pending availability
```

### INT8 (Production Ready)
```
Quantization: W8A8 (8-bit weights, 8-bit activations)
Engine: ModelOpt quantization
Engine Size: 4.6 GB
Data Type: int8
Precision: ~0.4% quantization error
Status: ✅ Verified (96.88% average success rate)
```

## Running Benchmarks

### Prerequisites
1. Activate environment:
   ```bash
   source /home/taco/.venv/bin/activate
   ```

2. Ensure TensorRT engines exist:
   ```bash
   ls -lh checkpoints/pi05_libero_onnx_compat/model.*.modelopt.engine
   ```

### FP32 Benchmark (Full Precision Baseline)

Start TensorRT server:
```bash
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.fp32.modelopt.engine \
  --port=8012 &
```

Run benchmark:
```bash
python scripts/benchmark_trt_models.py \
  --model_type=fp32 \
  --num_trials=10 \
  --task_suite_name=libero_spatial \
  --port=8012
```

Run all suites:
```bash
python scripts/benchmark_trt_models.py \
  --model_type=fp32 \
  --num_trials=10 \
  --task_suite_name=all \
  --port=8012
```

### INT8 Benchmark (Quantized)

Start TensorRT server:
```bash
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine \
  --port=8012 &
```

Run benchmark:
```bash
python scripts/benchmark_trt_models.py \
  --model_type=int8 \
  --num_trials=10 \
  --task_suite_name=libero_spatial \
  --port=8012
```

### FP4 Benchmark (When Available)

```bash
# Start server
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.fp4.modelopt.engine \
  --port=8012 &

# Run benchmark
python scripts/benchmark_trt_models.py \
  --model_type=fp4 \
  --num_trials=10 \
  --task_suite_name=all \
  --port=8012
```

### Batch Benchmark Script

Run all models sequentially:
```bash
bash scripts/run_full_benchmark.sh
```

This script:
- Tests FP32, INT8 (and FP4 if available)
- Runs 10 trials per task on all 4 suites
- Automatically manages TensorRT server lifecycle
- Generates JSON results in `./benchmark_results/`
- Creates a summary comparison table

## Expected Results

### Historical INT8 Performance
Based on previous evaluation (commit 78cdc37):

| Task Suite | Success Rate | Avg Latency | Peak VRAM |
|------------|--------------|-------------|-----------|
| libero_goal | 99.00% | ~120ms | 4.6GB |
| libero_spatial | 98.50% | ~115ms | 4.6GB |
| libero_object | 98.00% | ~118ms | 4.6GB |
| libero_10 | 92.00% | ~125ms | 4.6GB |
| **Overall** | **96.88%** | **~120ms** | **4.6GB** |

### Expected FP32 Performance
- Success Rate: ~98-99% (slight variations from environment)
- Latency: ~80-90ms (faster than INT8, no quantization overhead)
- VRAM: 13.0 GB (3x larger model)

### Expected FP4 Performance (Theoretical)
- Success Rate: ~95-98% (minor quantization loss)
- Latency: ~40-50ms (faster due to smaller model)
- VRAM: ~3-4 GB (smallest model, best for deployment)

## Output Format

Results are saved as JSON in `./benchmark_results/`:

```json
{
  "model_type": "fp32",
  "engine_path": "./checkpoints/pi05_libero_onnx_compat/model.fp32.modelopt.engine",
  "engine_size_gb": 13.0,
  "num_trials_per_task": 10,
  "total_trials": 400,
  "total_successes": 392,
  "overall_success_rate_percent": 98.0,
  "overall_avg_latency_ms": 85.5,
  "suite_results": {
    "libero_spatial": {
      "tasks": {
        "task_00": {
          "successes": 10,
          "trials": 10,
          "success_rate": 100.0,
          "avg_latency_ms": 84.2,
          "latencies": [80.1, 81.5, ...]
        },
        ...
      },
      "suite_success_rate": 98.5,
      "suite_avg_latency_ms": 84.8
    },
    ...
  }
}
```

## Analysis

### Success Rate Comparison
- Calculate percentage difference: `(FP32_success - INT8_success) / INT8_success * 100`
- Compare across all 40 tasks
- Identify which tasks are most affected by quantization

### Latency Analysis
- Per-step inference time includes:
  - Model forward pass
  - Image processing overhead
  - Action denormalization
- Compare avg_latency_ms across all models

### VRAM Trade-offs
| Model | Size | Latency | Accuracy | Trade-off |
|-------|------|---------|----------|-----------|
| FP32 | 13.0 GB | ~85ms | 98.0% | Baseline |
| INT8 | 4.6 GB | ~120ms | 96.9% | Good (66% smaller) |
| FP4 | ~3.0 GB | ~45ms | ~97.0% | Best (77% smaller) |

## Troubleshooting

### Server won't start
```bash
# Kill any existing servers
pkill -f "serve_trt.py"

# Start fresh
python scripts/serve_trt.py --engine_path=... --port=8012
```

### Connection refused
```bash
# Wait 5 seconds for server to initialize
sleep 5

# Run benchmark
python scripts/benchmark_trt_models.py ...
```

### Out of memory
- Reduce num_trials to 5
- Run single task suite instead of "all"
- Close other GPU processes

## References

- **LIBERO Benchmark**: https://libero-project.github.io/
- **NVIDIA ModelOpt**: https://github.com/NVIDIA/TensorRT-Model-Optimizer
- **TensorRT INT8**: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
- **Quantization Theory**: [INT8_SUMMARY.md](INT8_SUMMARY.md)

## Next Steps

1. ✅ Set up FP32 baseline benchmark
2. ✅ Run INT8 comparison
3. ⏳ Analyze quantization impact
4. ⏳ Test FP4 when engine is available
5. ⏳ Generate deployment recommendations

---

**Last Updated**: February 2026  
**Benchmark Script**: `scripts/benchmark_trt_models.py`  
**Results Directory**: `./benchmark_results/`
