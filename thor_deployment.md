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


### Benchmark Results (Sorted by Performance)

| Variant | Precision Label | Latency (ms) | Throughput (QPS) | GPU Mem | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **W8A8 (Sim)** | **WINT8AINT8** | **128.37** | **8.38** | **~3.7 GiB*** | **Fastest**. Implicit INT8. Real W8A8 requires calibration. |
| **W8A16 (QDQ)** | **WINT8AFP16** | 181.81 | 6.37 | 6.3 GiB | **Recommended**. Smallest verified model (13GB). Parity speed. |
| **INT4 (Sim)** | **WINT4AFP16** | 183.15 | 6.33 | ~6.3 GiB* | Run on FP32 model with `--int4`. Parity speed. |
| **FP16** | **WFP16AFP16** | 184.54 | 6.26 | 6.3 GiB | Baseline (25GB model). Native FP16 execution. |
| **BF16** | **WBF16ABF16** | 190.21 | 6.14 | ~6.3 GiB | Parity with FP16. |
| **FP8 (Sim)** | **WFP8AFP16** | 310.90 | 3.63 | ~6.3 GiB* | Slower. Requires optimized QAT model for gains. |

*\*Estimated memory for simulated runs.*

### Accuracy Verification (MSE vs PyTorch)
| Variant | MSE | Max Error | Status |
| :--- | :--- | :--- | :--- |
| **W8A16 (QDQ)** | **0.0061** | **0.203** | **Pass**. Negligible diff vs FP16. |
| **FP16** | 0.0061 | 0.205 | **Pass**. Baseline ONNX export error. |

*Note: The identical MSE indicates that W8A16 quantization introduced no additional accuracy loss compared to the validation-ready FP16 export.*

### Definitions
*   **WINT8AINT8**: Weights and Activations in INT8 (Full INT8).
*   **WINT8AFP16**: Weights in INT8, Activations in FP16. (Achieved via QDQ).
*   **WFP16AFP16**: Weights and Activations in FP16.
*   **WFP8AFP16**: Weights in FP8, Activations in FP16 (Simulated).
*   **WINT4AFP16**: Weights in INT4, Activations in FP16 (Simulated).

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

## Update: JAX -> PyTorch Pipeline (Jan 2026)
This section tracks the deployment of the *converted* JAX model (`pi05_libero`).

### 1. Accuracy (Offline Eval)
*   **Static MSE**: 1.25 (vs PyTorch). Flagged for review.
*   **Offline Validation**: `scripts/eval_libero_offline.py` provided. (Blocked by Hardware).
*   **Sim Evaluation**: Blocked by Thor SM 11.0 Incompatibility.

### 2. New Quantization Formats
*   **W4A4 (INT4/INT4)**:
    *   **Method**: `scripts/export_w4a4.py` (modelopt custom config).
    *   **Model**: `dist/final_w4a4/model.w4a4.onnx`.
    *   **Status**: Exported. Verified `trt.DequantizeLinear` ops present.
    *   **Requirement**: Needs TensorRT backend (CPU execution not supported).

### 3. Environment Warning
*   **Configuration**: CUDA 13.0 Driver. PyTorch 2.11 (`cu128`).
*   **Issue**: `onnxruntime` and `torch` binaries do not fully support Thor (SM 11.0) yet.
*   **Recommendation**: Use NVIDIA NGC PyTorch container or build from source for SM 11.0.