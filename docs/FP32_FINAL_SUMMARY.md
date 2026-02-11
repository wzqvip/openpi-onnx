# FP32 PyTorch Benchmark - Final Summary

**Date:** 2026-02-11  
**Status:** 3/4 suites completed  
**Issue:** libero_10 excluded due to script auto-restart bug

---

## ✅ Completed Results

### Individual Suite Performance

| Suite | Success Rate | Mean Latency | Median Latency | P99 Latency | GPU Memory |
|-------|--------------|--------------|----------------|-------------|------------|
| **libero_spatial** | 97.0% | 264.38 ms | 259.14 ms | 491.61 ms | 8.10 GB |
| **libero_goal** | 92.0% | 260.57 ms | 259.46 ms | 282.64 ms | 8.10 GB |
| **libero_object** | 96.0% | 262.32 ms | 261.89 ms | 281.04 ms | 8.10 GB |

### Overall Statistics (3 suites)

- **Average Success Rate:** 95.00%
- **Average Mean Latency:** 262.42 ms
- **Average Median Latency:** 260.16 ms
- **GPU Memory:** 8.10 GB (consistent)

---

## ❌ Known Issue: libero_10

### Problem Description
The evaluation script for libero_10 has an **automatic restart bug**:
1. Script completes all 10 tasks (reaches 100%)
2. Script **does not output** statistics (Total Success Rate, Latency)
3. Script **automatically restarts** from 0%
4. Cycle repeats indefinitely

### Evidence
```
100%|██████████| 10/10 [16:58<00:00, 101.89s/it]
  0%|          | 0/10 [00:00<?, ?it/s]        WARNING:root:Result: success
 10%|█         | 1/10 [01:34<14:13, 94.81s/it]
```

### Root Cause Analysis
- **Single task test:** Works correctly, outputs stats and exits
- **Full suite test:** Completes but fails to output final statistics
- Likely issue in the main evaluation loop or task iteration logic
- Other suites (spatial, goal, object) work perfectly

### Impact
- Cannot get accurate libero_10 FP32 baseline
- Overall FP32 average calculated from 3 suites only
- Comparison with INT8 (4/4 suites) is incomplete

---

## 📊 FP32 vs INT8 Comparison (Valid Suites Only)

### Accuracy Comparison (3 suites)

| Suite | FP32 | INT8 | Difference |
|-------|------|------|------------|
| spatial | 97.0% | 98.5% | +1.5% ✅ |
| goal | 92.0% | 99.0% | +7.0% 🚀 |
| object | 96.0% | 98.0% | +2.0% ✅ |
| **Average** | **95.00%** | **98.5%** | **+3.5%** |

### Key Findings
- ✅ INT8 outperforms FP32 across all tested suites
- 🎯 goal suite shows biggest improvement (+7%)
- 📈 Average improvement: +3.5% in favor of INT8
- 💾 INT8 model: 4.6GB vs FP32: 13GB (64.6% reduction)

---

## 🔍 Technical Details

### Test Configuration
- **Model:** checkpoints/pi05_libero_pytorch (FP32)
- **Trials per task:** 10
- **Seed:** 42
- **Framework:** PyTorch
- **Environment:** LIBERO Benchmark (4 suites)

### Successful Suites
1. **libero_spatial** (10 tasks × 10 trials = 100 episodes)
2. **libero_goal** (10 tasks × 10 trials = 100 episodes)
3. **libero_object** (10 tasks × 10 trials = 100 episodes)

**Total tested:** 300 episodes  
**Total successful:** 285 episodes  
**Overall success rate:** 95.00%

### Failed Suite
- **libero_10:** Script bug prevents completion
- **Attempts:** Multiple (all failed with auto-restart)
- **Status:** Excluded from final results

---

## 💡 Recommendations

### For Future Testing
1. **Fix libero_10 script bug:** Investigate task loop termination logic
2. **Add debug output:** Print before/after final statistics section
3. **Test with single task first:** Validate script behavior
4. **Consider alternative:** Use INT8 libero_10 result (92.0%) as reference

### For Production Deployment
1. ✅ **Use INT8 TensorRT model**
   - Higher accuracy (98.5% vs 95.0%)
   - Smaller size (4.6GB vs 13GB)
   - Lower memory usage (~50% reduction)
   - Production-validated on all 4 suites

2. 📊 **FP32 as reference only**
   - Good for baseline comparison
   - 3/4 suites validated
   - Higher resource requirements

---

## 📁 Log Files

- `pytorch_benchmark_spatial.log` - ✅ Complete
- `pytorch_benchmark_goal.log` - ✅ Complete
- `pytorch_benchmark_object.log` - ✅ Complete
- `pytorch_benchmark_10.log` - ❌ Incomplete (auto-restart bug)

---

## 🔗 Related Documents

- [FP32 vs INT8 Detailed Comparison](FP32_INT8_COMPARISON.md)
- [INT8 Final Results](openpi-onnx/INT8_FINAL_RESULTS.md)
- [Project README](openpi-onnx/README.md)
- [Benchmark Summary](BENCHMARK_SUMMARY.txt)

---

**Conclusion:** FP32 PyTorch baseline established for 3/4 LIBERO suites with 95% average accuracy and ~262ms mean latency. INT8 TensorRT consistently outperforms FP32 with +3.5% higher accuracy and 64.6% smaller model size.
