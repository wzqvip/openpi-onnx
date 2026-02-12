# INT8 Diagnostic Guide

This guide is kept for historical reference. The INT8 benchmark is now stable using the v1 path.

## If accuracy drops unexpectedly

1. Verify you are using `eval_libero_trt_v1.py`.
2. Confirm action padding (7 → 32) is applied.
3. Ensure `torch_norm_stats.json` is loaded.
4. Validate the TensorRT engine matches the target device.
5. Re-run a single suite with `--num_trials_per_task=1` to sanity check.

## Recommended command

```bash
./run_int8_benchmark_v1.sh
```

**Last updated**: 2026-02-12
