# Model Benchmark Results

Hardware: NVIDIA Jetson Thor (Blackwell GPU)
Date: 2026-01-09

| Precision | Latency (ms) | Throughput (QPS) | GPU Mem (MiB) | Notes |
|-----------|--------------|------------------|---------------|-------|
| FP32      | ~250         | 4.01             | ~13,000       | Weights: 12.1 GiB. |
| FP16      | Failed       | Failed           | Failed        | Segfault. Fix in progress. |
| INT8      | 118.11       | 8.47             | **4,018**     | **Measured**. Engine Size: 3.67 GiB. |
| NVFP4     | N/A          | N/A              | N/A           | Export Failed (Requires CUDA). |
| INT4      | Failed       | Failed           | Failed        | Parse Error. |

## Findings

1. **INT8 vs FP32 Memory**:
    - **Measured Runtime Memory**: 
        - **INT8**: Peak usage **4,018 MiB** (~3.92 GiB).
        - **FP32**: Peak usage **~13,000 MiB** (~12.7 GiB).
    - **Savings**: INT8 reduces VRAM usage by **3.2x**.
2. **Performance**: INT8 provides a **2.11x** speedup (8.47 QPS vs 4.01 QPS).
3. **Status**:
    - FP32: Stable.
    - INT8: Valid, Performant, and Memory Efficient.
