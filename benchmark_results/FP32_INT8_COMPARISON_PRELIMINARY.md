# FP32 vs INT8 Comparison (Final)

## Summary

- **FP32**: 93.75% (750/800), mean latency 262.41 ms, GPU memory 8.10 GB
- **INT8**: 98.25% (786/800), episode wall time mean 10.43s, GPU memory ~4.95 GB

> INT8 latency here is end-to-end episode wall time derived from tqdm logs. Inference-only latency is not logged by `eval_libero_trt_v1.py`.

## Accuracy by Suite

| Suite | FP32 Accuracy | INT8 Accuracy | Delta |
|------|---------------|---------------|-------|
| libero_spatial | 99.5% | 99.0% | -0.5% |
| libero_goal | 91.0% | 98.5% | +7.5% |
| libero_object | 95.0% | 99.5% | +4.5% |
| libero_10 | 89.5% | 96.0% | +6.5% |
| **Overall** | **93.75%** | **98.25%** | **+4.50%** |

## Notes

- INT8 results were produced using `eval_libero_trt_v1.py` and `serve_trt.py`.
- The WebSocket implementation (`eval_libero_trt.py`) was not used due to correctness issues.

**Last updated**: 2026-02-12
