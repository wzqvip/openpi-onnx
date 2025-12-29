# Pi0.5 ONNX Conversion & Deployment Guide

This guide details the process for converting the Pi0.5 PyTorch model to ONNX for localized deployment on NVIDIA GeForce RTX 5060 (Ada/Blackwell).

## 1. Environment Setup

**Hardware**: 
- GPU: NVIDIA RTX 5060 Laptop (or compatible Ada generation)
- Drivers: 560+ (WSL2 Passthrough recommended)

**Dependencies**:
```bash
# Core
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install transformers accelerate timm einops

# ONNX / TensorRT
pip install onnx onnxscript onnxruntime-gpu
pip install tensorrt  # Installs python bindings + libs
```

**System TensorRT**:
For `trtexec`, install the system package:
`sudo dpkg -i nv-tensorrt-local-repo-....deb`
`sudo apt-get install tensorrt`

## 2. Code Modifications (Required)

Before exporting, apply the following fixes to `pi0_pytorch.py`:

**Fix `CumSum` on Boolean Tensors**:
ONNX Runtime and TensorRT do not support `CumSum` on boolean types.
Locate `torch.cumsum` calls in `pi0_pytorch.py` and cast inputs to `long`:
```python
# usage:
# mask = ... (bool)
mask_cumsum = torch.cumsum(mask.to(torch.long), dim=1)
```

## 3. Export to ONNX (FP16 - Recommended)

Run the export script:
```bash
python export_onnx.py
```
*   **Output**: `checkpoints/pi05_libero_pytorch/model.onnx`
*   **Format**: ONNX Opset 18, FP16 Weights.
*   **Validation**: This model is verified to run correctly on ORT and TensorRT.

## 4. Optimization (TensorRT)

To build a TensorRT engine for maximum performance:
```bash
trtexec --onnx=checkpoints/pi05_libero_pytorch/model.onnx \
        --fp16 \
        --saveEngine=checkpoints/pi05_libero_pytorch/model.engine
```

**Performance on RTX 5060**:
- Latency: **~20.3 ms**
- Speedup vs PyTorch: **~5.2x**

## 5. Running Inference

### Python (ONNX Runtime)
```python
import onnxruntime as ort
import os

# Crucial: Add TRT libs to path if using pip install
import tensorrt_libs
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.path.dirname(tensorrt_libs.__file__)

sess = ort.InferenceSession("model.onnx", providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
# Run inference...
```

## 6. Known Issues (Quantization)

**MXFP8 / INT8**:
Current export tools (`nvidia-modelopt` 0.40) have compatibility issues with this model architecture and environment.
*   **MXFP8**: Fails due to missing Quantize nodes in exported graph.
*   **INT8**: Fails due to OOM (Memory) on standard WSL instances.

**Recommendation**: Use **FP16** for all current deployments. It offers the best balance of stability and performance.
