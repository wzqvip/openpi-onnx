# Pi0.5 Mobile Deployment Log

## Environment Setup
*   **Conda Env**: `openpi_mobile` (Python 3.11)
*   **GPU**: NVIDIA RTX 5060 Laptop (WSL2)
*   **TensorRT**: 10.14.1 (System Install) + Python Bindings

## Export Status
*   **Baseline (FP32/FP16)**: ✅ **SUCCESS**
    *   Path: `checkpoints/pi05_libero_pytorch/model.onnx`
    *   Opset: 18
    *   Fixes: `CumSum` bool cast, `GemmaRMSNorm` patch, `roPE` patch.
*   **MXFP8**: ✅ **SUCCESS (REPAIRED)**
    *   **Method**: Export (Legacy) -> Fix Weights -> Inject Quantize Nodes -> Patch CumSum.
    *   **Path**: `checkpoints/pi05_libero_pytorch/model.mxfp8.final.onnx`
*   **INT8**: ⚠️ **BLOCKED**
    *   Export failed due to Environment Instability (GPU Crash) and Memory Constraints (OOM).

## Benchmarks (RTX 5060)

| Format | Engine | Latency (Mean) | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch (FP16)** | Eager | 106.39 ms | ✅ | Baseline |
| **ONNX (FP16)** | TensorRT | **20.30 ms** | ✅ | **5.2x Speedup**. Recommended. |
| **MXFP8** | TensorRT | ~23.71 ms | ✅ | Slightly slower than FP16 (Overhead of explicit Quantization nodes). |
| **ONNX (FP32)** | TensorRT | N/A | ⚠️ | Export OOM (Requires >32GB RAM to export full model). |

## Recommendations
**Use FP16.** It is the fastest, easiest to export, and most robust format for this model on RTX 5060.
MXFP8 is viable but requires complex graph surgery and currently yields no speedup due to small batch inference overhead.
