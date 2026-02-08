# Performance Benchmark Guide - FP32 vs INT8 vs FP4

## Quick Start

### Option 1: Automated Full Benchmark (Recommended)

Run all models sequentially with automatic server management:

```bash
bash scripts/run_full_benchmark.sh
```

This will:
- Test FP32 baseline (13GB engine)
- Test INT8 quantization (4.6GB engine)
- Run 10 trials per task across all 4 LIBERO suites
- Generate JSON results in `./benchmark_results/`
- Create comparison report

**Estimated Time**: ~4-6 hours (FP32: 2-3 hours, INT8: 2-3 hours)

### Option 2: Manual Per-Model Testing

#### Test FP32 (Baseline)

```bash
# Terminal 1: Start TensorRT server
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.fp32.modelopt.engine \
  --port=8012 &

# Terminal 2: Run benchmark (wait 5 seconds after server starts)
sleep 5
python scripts/benchmark_trt_models.py \
  --model_type=fp32 \
  --num_trials=10 \
  --task_suite_name=all \
  --port=8012
```

#### Test INT8 (Quantized)

```bash
# Terminal 1: Start TensorRT server
python scripts/serve_trt.py \
  --engine_path=checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine \
  --port=8012 &

# Terminal 2: Run benchmark
sleep 5
python scripts/benchmark_trt_models.py \
  --model_type=int8 \
  --num_trials=10 \
  --task_suite_name=all \
  --port=8012
```

## Understanding Results

### JSON Output Structure

Results are saved as `benchmark_results/benchmark_{model_type}_{trials}trials.json`:

```json
{
  "model_type": "fp32",
  "engine_path": "...",
  "engine_size_gb": 13.0,
  "overall_success_rate_percent": 98.5,
  "overall_avg_latency_ms": 85.2,
  "total_trials": 400,
  "total_successes": 394,
  "suite_results": {
    "libero_spatial": {
      "tasks": {
        "task_00": {
          "success_rate": 100.0,
          "avg_latency_ms": 84.5,
          "latencies": [80.1, 82.3, ...]
        }
      },
      "suite_success_rate": 98.5,
      "suite_avg_latency_ms": 84.8
    }
  }
}
```

### Key Metrics

| Metric | Description | Impact |
|--------|-------------|--------|
| **Success Rate** | % of successful trials | Accuracy (higher=better) |
| **Avg Latency** | Average inference time in ms | Speed (lower=better) |
| **Engine Size** | TensorRT model size | Deployment (smaller=better) |

## Generating Comparison Report

After running benchmarks, generate a summary:

```bash
python scripts/generate_comparison_report.py ./benchmark_results/
```

This creates:
- Summary tables by model
- Per-suite comparison
- Task-by-task analysis
- Markdown report

## Expected Results

### FP32 Baseline (13 GB)
- **Success Rate**: ~98-99%
- **Avg Latency**: ~80-90ms
- **VRAM**: 13 GB
- **Status**: Reference point (no quantization)

### INT8 Quantization (4.6 GB)
- **Success Rate**: ~96-97% (verified: 96.88%)
- **Avg Latency**: ~115-125ms
- **VRAM**: 4.6 GB
- **Status**: ✅ Production-ready
- **Trade-off**: 65% smaller model, ~1-2% accuracy drop

### FP4 Quantization (TBD)
- **Success Rate**: ~95-97% (estimated)
- **Avg Latency**: ~40-50ms (estimated)
- **VRAM**: ~3 GB (estimated)
- **Status**: ⏳ Awaiting available engine

## Interpretation Guide

### Success Rate Comparison

Calculate accuracy drop:
```
Accuracy Drop = (FP32_Rate - Quantized_Rate) / FP32_Rate * 100%
```

- < 1%: Excellent (acceptable for deployment)
- 1-3%: Good (viable with trade-offs)
- > 3%: Significant (needs further optimization)

### Latency Analysis

Per-step inference includes:
- Model forward pass
- Batch processing
- Image normalization
- Action denormalization

Note: INT8 may have higher latency due to quantization overhead, but reduced model size enables better memory utilization.

### VRAM Trade-offs

| Model | Size | Speed Factor | Accuracy Drop |
|-------|------|-------------|---------------|
| FP32 | 1.0x | 1.0x | 0% |
| INT8 | 0.35x | 1.4x | ~1% |
| FP4 | 0.23x | ~0.5x | ~2% |

Choose based on:
- **FP32**: When accuracy is critical, memory is abundant
- **INT8**: When balance of accuracy/size/speed needed
- **FP4**: When deployment space is extremely limited

## Troubleshooting

### Server won't start
```bash
# Check for existing servers
ps aux | grep serve_trt.py

# Kill hanging processes
pkill -f serve_trt.py

# Check logs
tail -f /tmp/trt_server_*.log
```

### Connection refused
- Ensure server started: `sleep 5` after starting server
- Check server is running: `netstat -tlnp | grep 8012`
- Try different port: `--port 8013`

### Out of memory
- Reduce trials: `--num_trials 5`
- Test single suite: `--task_suite_name libero_spatial`
- Close other GPU applications

### Slow inference
- Normal for FP32 (13GB model, more compute)
- INT8 may appear slower due to overhead, but has better memory efficiency
- Latency varies by task complexity

## File Structure

```
benchmark_results/
├── benchmark_fp32_10trials.json          # FP32 raw results
├── benchmark_int8_10trials.json          # INT8 raw results
├── benchmark_fp32.log                    # FP32 execution log
├── benchmark_int8.log                    # INT8 execution log
├── COMPARISON_REPORT.md                  # Summary report
└── BENCHMARK_REPORT.md                   # Detailed markdown report
```

## Next Steps

After benchmarking:

1. **Analyze Results**
   - Compare accuracy across models
   - Identify quantization-sensitive tasks
   - Check latency improvements

2. **Deployment Decision**
   - FP32: Full accuracy baseline
   - INT8: Recommended for production (96.88% proven)
   - FP4: Consider when extreme size constraints

3. **Optimization**
   - Fine-tune INT8 calibration
   - Explore mixed-precision (FP32 encoder + INT8 decoder)
   - Custom quantization for specific tasks

## References

- Complete comparison guide: [FP32_FP4_INT8_COMPARISON.md](docs/conversion/FP32_FP4_INT8_COMPARISON.md)
- INT8 technical details: [INT8_SUMMARY.md](INT8_SUMMARY.md)
- Evaluation results: [INT8_FINAL_RESULTS.md](INT8_FINAL_RESULTS.md)

---

**Created**: February 2026  
**Updated**: February 2026
