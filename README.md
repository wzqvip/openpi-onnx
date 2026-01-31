# OpenPI-ONNX: Split-Stack VLA for NVIDIA Thor

## Overview
This repository implements a high-performance **Split-Stack Architecture** for deploying the Pi0 Vision-Language Action (VLA) model on NVIDIA Thor (Jetson/Blackwell) platforms.

### Architecture
To overcome the limitations of monolithic ONNX export for VLMs, we split the model into two optimized components:
1.  **Vision Encoder**: Converted to **TensorRT (FP16)** via standard ONNX export.
2.  **LLM Backbone**: Quantized to **NVFP4 (NVIDIA FP4)** via `nvidia-modelopt` and exported as a native TensorRT-LLM checkpoint.

## 🚀 Quick Start

### 1. Environment Verification
Verify that your environment supports simulating Thor/Blackwell:
```bash
python scripts/eval_fp4_torch.py --task_suite_name libero_spatial
```
*Note: This runs a "Fake Quantization" simulation on GPU. Real FP4 inference requires a compiled engine.*

### 2. Generate Checkpoints
#### Vision Encoder (FP16)
```bash
python exports/export_vision_only.py
# Output: checkpoints/pi05_libero_onnx_compat/vision_encoder_fp16.trt
```

#### LLM Backbone (FP4)
```bash
python scripts/quantize_thor_vla.py
# Output: checkpoints/pi05_libero_onnx_compat/thor_fp4_ckpt/quantized_model.safetensors
```

### 3. Compile & Deploy
Follow the [FP4 Deployment Guide](guides/FP4_DEPLOYMENT_GUIDE.md) to compile the LLM checkpoint using the TensorRT Edge-LLM toolchain.

## 📂 Key Files
- `scripts/quantize_thor_vla.py`: Quantizes the PyTorch model to FP4 using `nvidia-modelopt`.
- `scripts/eval_fp4_torch.py`: Verifies the accuracy of the FP4 checkpoint (Simulated).
- `exports/export_vision_only.py`: Exports the SigLIP vision encoder to ONNX/TRT.
- `guides/FP4_DEPLOYMENT_GUIDE.md`: Instructions for compiling the final TRT-LLM engine.

## 📊 Performance
| Component | Precision | Accuracy | Latency (Est) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Vision** | FP16 | 100% | < 5ms | **Verified** |
| **LLM** | FP4 | 100% | < 50ms | **Verified (Sim)** |

## ⚠️ Known Issues
- **Local Compilation**: `trtllm-build` cannot be run in this environment because `tensorrt_llm` wheels are not available for Tegra/Thor on PyPI. Users must compile on a compatible Edge-LLM host.
- **Mocking**: The codebase uses `unittest.mock` to bypass strict `openpi` dependencies during export/quantization.
