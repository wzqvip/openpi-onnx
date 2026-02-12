# FP32 PyTorch Baseline - 20 Trials Standard Test

**Test date**: 2026-02-11 ~ 2026-02-12  
**Config**: 20 trials per task, seed 42  
**Model**: checkpoints/pi05_libero_pytorch (13GB)  
**Device**: NVIDIA Jetson (8.10GB GPU Memory)

---

## 📊 Results

| Suite | Accuracy | Success/Total | Avg Latency (ms) | Median (ms) | P99 (ms) | GPU Memory (GB) |
|------|----------|---------------|------------------|-------------|----------|-----------------|
| **libero_spatial** | **99.5%** | 199/200 | 263.23 | 261.68 | 286.90 | 8.10 |
| **libero_goal** | **91.0%** | 182/200 | 259.49 | 258.43 | 271.99 | 8.10 |
| **libero_object** | **95.0%** | 190/200 | 264.36 | 263.70 | 283.00 | 8.10 |
| **libero_10** | **89.5%** | 179/200 | 262.56 | 262.46 | 273.26 | 8.10 |

### Overall (all 4 suites)

- **Overall accuracy**: **93.75%** (750/800 episodes)
- **Mean latency**: **262.41 ms**
- **Median latency**: **261.57 ms**
- **P99 latency**: **278.79 ms**
- **Peak GPU memory**: **8.10 GB**

---

## 📈 Analysis

### Accuracy distribution

- **Best performance**: libero_spatial (99.5%)
- **Most challenging**: libero_10 (89.5%)
- **Mid-range**: libero_object (95.0%), libero_goal (91.0%)

### Latency behavior

- **Consistency**: 259–264 ms across suites
- **Tail latency**: P99 within 272–287 ms
- **Per-inference**: ~262 ms/infer

### Resource usage

- **GPU memory**: stable at 8.10 GB across suites
- **Hardware requirement**: ~9 GB GPU memory recommended

---

## 🔍 Comparison vs 10-trial run

| Metric | 10 trials | 20 trials | Change |
|--------|----------|-----------|--------|
| **Overall accuracy** | 95.0% (285/300) | 93.75% (750/800) | -1.25% |
| **Mean latency** | 262.42 ms | 262.41 ms | -0.01 ms |
| **P99 latency** | ~280 ms | 278.79 ms | slightly improved |

**Conclusion**: More trials reduced variance. Latency is stable; accuracy slightly regressed due to a larger sample size.

---

## 📁 Logs

- `benchmark_logs/fp32_spatial_20trials.log`
- `benchmark_logs/fp32_goal_20trials.log`
- `benchmark_logs/fp32_object_20trials.log`
- `benchmark_logs/fp32_10_20trials.log`

---

## ⏭️ Next steps

- [x] FP32 baseline complete
- [x] INT8 TensorRT 20-trial benchmark complete
- [ ] Update the final FP32 vs INT8 comparison report

**Command**: `./run_int8_benchmark_v1.sh`

## 📈 Analysis

### Accuracy distribution

- **Best performance**: libero_spatial (99.5%)
- **Most challenging**: libero_10 (89.5%)
- **Mid-range**: libero_object (95.0%), libero_goal (91.0%)

### Latency behavior

- **Consistency**: 259–264 ms across suites
- **Tail latency**: P99 within 272–287 ms
- **Per-inference**: ~262 ms/infer

### Resource usage

- **GPU memory**: stable at 8.10 GB across suites
- **Hardware requirement**: ~9 GB GPU memory recommended

---

## 🔍 Comparison vs 10-trial run

| Metric | 10 trials | 20 trials | Change |
|--------|----------|-----------|--------|
| **Overall accuracy** | 95.0% (285/300) | 93.75% (750/800) | -1.25% |
| **Mean latency** | 262.42 ms | 262.41 ms | -0.01 ms |
| **P99 latency** | ~280 ms | 278.79 ms | slightly improved |

**Conclusion**: More trials reduced variance. Latency is stable; accuracy slightly regressed due to a larger sample size.

---

## 📁 Logs

- `benchmark_logs/fp32_spatial_20trials.log`
- `benchmark_logs/fp32_goal_20trials.log`
- `benchmark_logs/fp32_object_20trials.log`
- `benchmark_logs/fp32_10_20trials.log`

---

## ⏭️ Next steps

- [x] FP32 baseline complete
- [x] INT8 TensorRT 20-trial benchmark complete
- [ ] Update the final FP32 vs INT8 comparison report

**Command**: `./run_int8_benchmark_v1.sh`
