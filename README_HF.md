---
license: mit
tags:
- onnx
- robotics
- openpi
- jetson-thor
- quantization
- w8a16
platform: onnx
pipeline_tag: robotics
library_name: openpi
---

# OpenPi VLA Models (ONNX Export)

This repository contains **ONNX-exported** versions of the [OpenPi](https://github.com/Physical-Intelligence/openpi) VLA (Vision-Language-Action) models, specifically tuned for **NVIDIA Jetson Thor** (Blackwell) and **GeForce RTX Ada/Blackwell** GPUs.

These models are derived from the [Pi0.5 Libero](https://www.physicalintelligence.company/blog/pi05) baseline (OpenVLA architecture).

## 📊 Performance Benchmarks (Jetson Thor)

| Variant | Latency (ms) | Throughput (QPS) | GPU Mem (Est) | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **W8A16 (QDQ)** | **181.81** | **6.37** | **~6.3 GiB** | ✅ **Recommended** | **Production Ready**. Verified accuracy parity (MSE 0.0061). |
| **FP16** | 184.54 | 6.26 | ~13.0 GiB | ✅ Verified | Baseline. Native FP16 execution. |
| **W8A8 (Sim)** | 128.37 | 8.38 | ~3.7 GiB | 🚧 Benchmark | Fastest (1.4x speedup), requires calibration. |
| **INT4 (Sim)** | 183.15 | 6.33 | ~6.3 GiB | 🚧 Experimental | Parity speed with W8A16. |

*Memory Note: INT8 quantization reduces runtime VRAM usage by approximately **3.2x** compared to FP32 (4GB vs 13GB measured on runtime).*

---

## 📂 Repository Structure & "Why is this 231GB?"

We preserve **all** precision variants and intermediate export states for research reproducibility. 

| Folder | Precision | Description | Size | Contents |
| :--- | :--- | :--- | :--- | :--- |
| **`final_w8a16/`** | **W8A16** | **[USE THIS]** Clean, collapsed ONNX model. | ~12GB | `model.w8a16.onnx`, `.data` |
| `final_fp16/` | FP16 | Baseline high-precision export. | ~24GB | `model.fp32.onnx` |
| `final_w4a4/` | W4A4 | Experimental INT4/INT4 export. | ~12GB | *Granular (Many files)* |
| `final_w8a16_new/` | W8A16 | Uncollapsed dev export (Duplicate state). | ~12GB | *Granular (Many files)* |
| `checkpoints/` | PyTorch | **Raw Archive**. Original JAX->PyTorch weights. | ~100GB | `.pt` files, config, raw layers |

**Recommendation**: Users should download **ONLY** `final_w8a16/` for deployment.

---

## ⚠️ Environment & Compatibility Notes

**1. NVIDIA Jetson Thor (SM 11.0)**
*   **Driver Requirement**: CUDA 13.0+ / Driver 560+
*   **Issue**: Standard `onnxruntime` and PyTorch binaries may not fully support SM 11.0 (Blackwell) instructions yet.
*   **Fix**: Use the NVIDIA NGC containers or compile from source.

**2. Quantization Support**
*   **W4A4**: Requires a custom TensorRT backend to execute `trt.DequantizeLinear` ops efficiently. CPU execution is not supported.
*   **W8A16**: Uses standard ONNX QDQ nodes (`QuantizeLinear`/`DequantizeLinear`). Compatible with standard ONNX Runtime (GPU) and TensorRT.

---

## 🛠️ Deployment Code

### Python (ONNX Runtime)

```python
import onnxruntime as ort
import os
import tensorrt_libs

# 1. Register TensorRT Plugins (Critical for QDQ models)
os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.path.dirname(tensorrt_libs.__file__)

# 2. Load Model
model_path = "final_w8a16/model.w8a16.onnx"
providers = [
    ("TensorrtExecutionProvider", {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_int8_enable": True,    # Enable INT8 Tensor Cores
        "trt_engine_cache_enable": True,
    }),
    "CUDAExecutionProvider"
]

session = ort.InferenceSession(model_path, providers=providers)
print("Model loaded successfully!")
```

### TensorRT CLI (`trtexec`)

```bash
# Compile W8A16 model to Engine
trtexec --onnx=final_w8a16/model.w8a16.onnx         --fp16 --int8         --saveEngine=model.engine         --verbose
```

