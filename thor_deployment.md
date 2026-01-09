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

**Note**: Files are large (13GB) because they contain full-sized tensor data (Fake Quantization or uncompressed export).

### 4. Cloud Storage
Models are being uploaded to Hugging Face Hub:
- **Repository**: [Tacoin/openpi-pi05-libero-thor-onnx](https://huggingface.co/Tacoin/openpi-pi05-libero-thor-onnx)
- **Content**: Organized ONNX exports (FP32, FP16, NVFP8, NVFP4, INT8, INT4).

### 4. Benchmarks
- **Environment**: GPU usage enabled (Torch 2.9.0 + CUDA).
- **Note**: PyTorch benchmark will run on GPU. ONNX Runtime uses CPU provider (GPU provider unavailable for this version).




### 3. Benchmarks
| Format | Precision | Latency (ms) | Notes |
| :--- | :--- | :--- | :--- |
| PyTorch | FP32 | 417.89 | GPU Enabled (Torch 2.9.0) |
| ONNX | FP32 | Failed | ORT-GPU Missing |
| ONNX | FP16 | Failed | ORT-GPU Missing |
| ONNX | BF16 | - | Not Exported |
| ONNX | INT8 | Failed | ORT-GPU Missing |
| ONNX | NVFP4| Failed | ORT-GPU Missing |
| ONNX | INT4 | Failed | ORT-GPU Missing |
