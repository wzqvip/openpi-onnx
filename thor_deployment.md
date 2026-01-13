# Jetson Thor Deployment Log

## Environment Info
*   **Device**: Jetson Thor (Arm64 + Blackwell)
*   **Memory**: 128GB Unified
*   **System Sudo**: 20020926

## Environment Setup
*   **Venv**: `/home/taco/openpi-onnx/.venv`
*   **Index URL**: `https://pypi.jetson-ai-lab.io/sbsa/cu130`

## Progress Log

### 1. Environment Configuration
- [x] Install Dependencies
    - Used `uv` for faster installation: `uv pip install -e .`
    - Modified `pyproject.toml` to remove `fsspec[gcs]` to avoid dependency conflicts.
    - Installed `nvidia-modelopt` and `packaging`.

### 2. Model Export Scripts
- Created `create_full_dummy_checkpoint.py`: Generates a full-sized dummy checkpoint (safetensors).
- Created `export_nvfp4_onnx.py`: Exports NVFP4 optimized ONNX model (Blackwell compatible).
- Created `export_int4_onnx.py`: Exports INT4 Blockwise ONNX model.
### 3. Export Results
Models are organized in `checkpoints/pi05_libero_pytorch/` with separate folders:
- **`fp32/`**: `model.onnx` + `model.onnx.data` (13GB)
- **`fp16/`**: `model.onnx` + `model.onnx.data` (6.5GB)
- **`int8/`**: `model.onnx` + `model.onnx.data` (13GB, FakeQuant)
- **`int4/`**: `model.onnx` + `model.onnx.data` (13GB, FakeQuant)
- **`nvfp4/`**: `model.onnx` + `model.onnx.data` (13GB, FakeQuant)
- **`nvfp8/`**: `model.onnx` + `model.onnx.data` (13GB, FakeQuant)
- **`w8a16/`**: `model.w8a16.onnx` + `.data` (~3-4GB, Weight-Only INT8)

**Note**: Files are large (13GB) because they contain full-sized tensor data (Fake Quantization or uncompressed export).

### 4. Cloud Storage
Models are being uploaded to Hugging Face Hub:
- **Repository**: [Tacoin/openpi-pi05-libero-thor-onnx](https://huggingface.co/Tacoin/openpi-pi05-libero-thor-onnx)
- **Content**: Organized ONNX exports (FP32, FP16, NVFP8, NVFP4, INT8, INT4, W8A16).

### 5. W8A16 Export
- Created `export_w8a16_onnx.py`.
- Configured `modelopt` for Weight-Only INT8 (Inputs/Activations in FP16).
- Addressed `AttributeError: module 'ml_dtypes' has no attribute 'float4_e2m1fn'` by upgrading `ml_dtypes`.

### 6. Benchmarks

see /home/taco/openpi-onnx/BENCHMARK_RESULTS.md


Hardware: NVIDIA Jetson Thor (Blackwell GPU)
Date: 2026-01-09

| Precision | Latency (ms) | Throughput (QPS) | GPU Mem (MiB) | Notes |
|-----------|--------------|------------------|---------------|-------|
| FP32      | ~250         | 4.01             | ~13,000       | Weights: 12.1 GiB. |
| FP16 (W16A16) | 184.54       | 6.26             | 6,314         | Exported as FP32, Run as FP16. Valid. |
| INT8      | 118.11       | 8.47             | **4,018**     | **Measured**. Engine Size: 3.67 GiB. |
| W8A16     | 181.81       | 6.37             | 6,313         | QDQ Export (13GB). Weights INT8, Compute Mixed. |
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

e-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md