# Standardized Benchmark Plan - FP32 vs INT8

**Date:** 2026-02-11  
**Purpose:** Fair comparison of FP32 PyTorch baseline vs INT8 TensorRT quantized model

---

## 🎯 Test Configuration

### Standard Parameters
- **Trials per task:** 20 (consistent across all tests)
- **Tasks per suite:** 10
- **Total episodes per suite:** 200 (20 trials × 10 tasks)
- **Random seed:** 42 (for reproducibility)
- **Evaluation framework:** LIBERO Benchmark

### Test Suites
1. `libero_spatial` - Spatial reasoning tasks
2. `libero_goal` - Goal-oriented tasks
3. `libero_object` - Object manipulation tasks
4. `libero_10` - Long-horizon tasks (10 steps)

**Total episodes:** 800 (4 suites × 200 episodes)

---

## 📝 Execution Plan

### Step 1: Clean Previous Results ✅
```bash
cd /home/taco/openpi-onnx
rm -rf benchmark_logs/*.log
rm -rf benchmark_results/*.json
```

### Step 2: Run FP32 Benchmark
```bash
# Estimated time: 3-4 hours for all 4 suites
./run_fp32_benchmark.sh
```

**What it measures:**
- ✓ Accuracy (success rate per suite)
- ✓ Latency (mean, median, P99)
- ✓ GPU memory usage (peak)
- ✓ Individual task performance

**Output:**
- Logs: `benchmark_logs/fp32_*_20trials.log`
- JSON: `benchmark_results/fp32_*_20trials.json`

### Step 3: Run INT8 Benchmark
```bash
# Estimated time: 2-3 hours for all 4 suites
./run_int8_benchmark.sh
```

**What it measures:**
- ✓ Accuracy (success rate per suite)
- ✓ Latency (mean, median, P99)
- ✓ GPU memory usage (peak)
- ✓ Individual task performance

**Output:**
- Logs: `benchmark_logs/int8_*_20trials.log`
- JSON: `benchmark_results/int8_*_20trials.json`

### Step 4: Analyze Results
```bash
python3 scripts/analyze_results.py
```

**Output:**
- Markdown report: `benchmark_results/BENCHMARK_COMPARISON_REPORT.md`
- JSON summary: `benchmark_results/benchmark_summary.json`

---

## 📊 Expected Measurements

### Accuracy Metrics
- Success rate per suite (%)
- Success rate per task (%)
- Overall success rate (%)
- Failed task analysis

### Latency Metrics
- Mean inference time (ms)
- Median inference time (ms)
- P99 latency (ms)
- Standard deviation

### Memory Metrics
- Peak GPU memory (GB)
- Average GPU memory (GB)
- Model size on disk (GB)

---

## 🔍 Why This Comparison is Fair

1. **Same sample size:** Both FP32 and INT8 tested with 20 trials/task
2. **Same random seed:** Ensures same initial conditions
3. **Sequential execution:** Prevents resource contention
4. **Same environment:** Same Python env, CUDA version, LIBERO version
5. **Comprehensive metrics:** Accuracy, latency, and memory all measured

---

## 📈 Expected Timeline

| Step | Task | Duration | Total Time |
|------|------|----------|------------|
| 1 | Clean previous results | 1 min | 0h 01m |
| 2 | FP32 benchmark (4 suites) | ~3-4 hours | 3h 30m |
| 3 | INT8 benchmark (4 suites) | ~2-3 hours | 6h 00m |
| 4 | Analyze results | 1 min | 6h 01m |

**Total estimated time:** ~6 hours

---

## 🛠️ Monitoring Progress

### Check FP32 Progress
```bash
# Watch latest FP32 log
tail -f benchmark_logs/fp32_spatial_20trials.log

# Check completion
ls -lh benchmark_logs/fp32_*_20trials.log
```

### Check INT8 Progress
```bash
# Watch latest INT8 log
tail -f benchmark_logs/int8_spatial_20trials.log

# Check completion
ls -lh benchmark_logs/int8_*_20trials.log
```

### Quick Status Check
```bash
# Count completed suites
echo "FP32: $(ls benchmark_logs/fp32_*_20trials.log 2>/dev/null | wc -l)/4 suites"
echo "INT8: $(ls benchmark_logs/int8_*_20trials.log 2>/dev/null | wc -l)/4 suites"
```

---

## ⚠️ Important Notes

1. **Do not interrupt:** Let each suite complete fully
2. **Check disk space:** Ensure enough space for logs (~100MB)
3. **Monitor GPU temp:** Keep GPU temperature below 85°C
4. **No parallel runs:** Only one benchmark script at a time
5. **Backup previous results:** If needed, save old logs before starting

---

## 📁 File Structure After Completion

```
openpi-onnx/
├── benchmark_logs/
│   ├── fp32_spatial_20trials.log
│   ├── fp32_goal_20trials.log
│   ├── fp32_object_20trials.log
│   ├── fp32_10_20trials.log
│   ├── int8_spatial_20trials.log
│   ├── int8_goal_20trials.log
│   ├── int8_object_20trials.log
│   └── int8_10_20trials.log
├── benchmark_results/
│   ├── BENCHMARK_COMPARISON_REPORT.md
│   └── benchmark_summary.json
├── run_fp32_benchmark.sh
├── run_int8_benchmark.sh
└── scripts/
    └── analyze_results.py
```

---

## 🎯 Success Criteria

- ✅ All 8 benchmark logs generated (4 FP32 + 4 INT8)
- ✅ All logs contain "Total Success Rate:" line
- ✅ All logs contain latency and memory statistics
- ✅ Comparison report generated successfully
- ✅ No crashes or incomplete suites

---

## 🚀 Ready to Start

When ready, execute:
```bash
cd /home/taco/openpi-onnx

# Step 1: Clean (if needed)
rm -rf benchmark_logs/*.log benchmark_results/*.json

# Step 2: Start FP32 benchmark
./run_fp32_benchmark.sh

# Wait for completion, then step 3: Start INT8 benchmark
./run_int8_benchmark.sh

# Step 4: Generate report
python3 scripts/analyze_results.py
```

---

**Good luck with the benchmarks! 🚀**
