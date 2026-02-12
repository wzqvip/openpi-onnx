# INT8 Evaluation Fix Report

**Date**: 2026-02-12

## Issue

The WebSocket evaluation path (`eval_libero_trt.py`) produced ~15% success, far below the expected ~97%.

## Root Cause

- Missing action-dimension padding (7 → 32) in the WebSocket path.
- Output normalization was applied to 7D stats while the model outputs 32D actions.
- Result: invalid action ranges and timeouts across tasks.

## Fix

- Switch back to the verified v1 implementation: `eval_libero_trt_v1.py`.
- Use the correct normalization pipeline and action padding.
- Start the TensorRT server explicitly (`serve_trt.py`) for v1.

## Outcome

- **Final accuracy**: 98.25% (786/800)
- **Suites**: 99.00% / 98.50% / 99.50% / 96.00%

## Commands

```bash
./run_int8_benchmark_v1.sh
tail -f benchmark_logs/int8_full_v1.log

**Last updated**: 2026-02-12

**Summary**: The failure was caused by using a different implementation that lacked critical action-padding and normalization logic. Switching back to the verified v1 path restored correct behavior and enabled the final 98.25% result.


