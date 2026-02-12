# INT8 Summary

## Overview

INT8 TensorRT evaluation completed successfully using `eval_libero_trt_v1.py` and `serve_trt.py`. The final accuracy is **98.25% (786/800)** across all four LIBERO suites.

## Final Results (20 trials per task)

| Suite | Accuracy | Success/Total |
|-------|----------|---------------|
| libero_spatial | 99.00% | 198/200 |
| libero_goal | 98.50% | 197/200 |
| libero_object | 99.50% | 199/200 |
| libero_10 | 96.00% | 192/200 |
| **Overall** | **98.25%** | **786/800** |

## Latency & Memory

- **Episode wall time (end-to-end)**: mean 10.43s, median 9.21s, p99 21.21s
- **GPU memory (INT8)**: 4954 MiB (~4.95 GB) with engine loaded

> Episode wall time is derived from tqdm “20/20” lines in the logs. Inference-only latency is not logged by `eval_libero_trt_v1.py`.

## Root Cause & Fix (historical)

- The WebSocket evaluation path (`eval_libero_trt.py`) failed due to missing action-dimension padding (7 → 32), leading to invalid outputs and ~15% success.
- Switching back to the verified v1 implementation (`eval_libero_trt_v1.py`) restored correct normalization and action handling.

## Commands

```bash
./run_int8_benchmark_v1.sh
tail -f benchmark_logs/int8_full_v1.log
```

## Logs

- `benchmark_logs/int8_spatial_20trials_v1.log`
- `benchmark_logs/int8_goal_20trials_v1.log`
- `benchmark_logs/int8_object_20trials_v1.log`
- `benchmark_logs/int8_10_20trials_v1.log`

**Last updated**: 2026-02-12

