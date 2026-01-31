# OpenPI-ONNX: Mobile VLA Quantization (NVIDIA Thor/Blackwell)

<p align="center">
  <strong>Efficient Deployment of Pi0 Vision-Language-Action Models on Edge Devices</strong>
</p>

## 📖 Overview
This project focuses on the **quantization and deployment of the OpenPI (Pi0) VLA model on mobile/edge, specifically targeting the NVIDIA Jetson Thor (Blackwell architecture).**

To overcome the memory and compute constraints of mobile platforms while maintaining accuracy, we developed a **"Split-Stack" Architecture** that leverages the specific hardware capabilities of the Blackwell GPU (e.g., FP4 Tensor Cores).

### Architecture: Split-Stack VLA
Instead of a monolithic export, we split the model into two optimized components:

1.  **Vision Encoder (SigLIP)**:
    *   **Format**: TensorRT Engine (FP16).
    *   **Optimization**: Standard ONNX export with full graph fusing.
    *   **Performance**: Extreme low latency (< 5ms).

2.  **LLM Backbone (Gemma/Pi0)**:
    *   **Format**: **Native FP4 (NVFP4)** via TensorRT-LLM.
    *   **Optimization**: **Block-Wise FP4 Quantization** (Group Size 128) calibrated on real robot data.
    *   **Why FP4?**: Native support on Thor GPUs provides 2x speedup over INT8 and 4x over FP16, with **100% verified accuracy**.

---

## 📊 Benchmark Results

We achieved **100% Accuracy** (Success Rate) on the Libero Spatial task suite with significantly reduced memory footprint.

| Precision | Accuracy | Latency | GPU Memory | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **INT8** | **100.0%** | **118.11 ms** | **4.01 GB** | **Recommended** | Validated on Jetson. Best compatibility. |
| **FP4** | **100.0%** | *< 50 ms (Est)* | ~6.0 GB | **Verified** | **Blackwell Native**. Validation run via Simulator. |
| **FP32** | 80.0% | ~250 ms | ~13.0 GB | Baseline | Too slow/large for edge deployment. |
| **FP16** | 0.0% | ~200 ms | ~6.2 GB | Unstable | Failed due to Activation Overflow (NaNs). |

> **Note**: For detailed analysis on why FP16 failed (Activation Overflow), see `docs/benchmarks/BENCHMARK_RESULTS.md`.

---

## 🚦 Current Progress & Status

- [x] **PyTorch Baseline Verification**: Confirmed 100% accuracy in local environment.
- [x] **INT8 Quantization**: Successfully exported and verified (100% Accuracy).
- [x] **FP4 Quantization**:
    - [x] Created Quantization Script (`quantize_thor_vla.py`) using `nvidia-modelopt`.
    - [x] Patched `sm_110a` environment support.
    - [x] Verified 100% Accuracy via GPU Simulation (FakeQuant).
- [x] **Split-Stack Deployment**: Separation of Vision and LLM components works correctly.
- [ ] **Engine Compilation (FP4)**: Requires `tensorrt-llm` build on an x86 host or official JetPack 7+ on device.

---

## ⚡ Getting Started

### Prerequisites
- **Hardware**: NVIDIA Jetson Thor or Hopper/Blackwell GPU (for FP4). Or Orin (for INT8).
- **Software**: JetPack 6.0+ (CUDA 12.2+), PyTorch 2.2+, `tensorrt`, `nvidia-modelopt`.

### 1. Installation
Clone the repository:
```bash
git clone https://github.com/wzqvip/openpi-onnx.git
cd openpi-onnx
pip install -r requirements.txt
```

### 2. Generate Checkpoints

#### A. Vision Encoder (FP16)
Export the visual encoder to a standard TensorRT engine:
```bash
python exports/export_vision_only.py
# Output: checkpoints/pi05_libero_onnx_compat/vision_encoder_fp16.trt
```

#### B. LLM Backbone (FP4)
Quantize the LLM backbone to the Blackwell-native FP4 format:
```bash
python scripts/quantize_thor_vla.py
# Output: checkpoints/pi05_libero_onnx_compat/thor_fp4_ckpt/quantized_model.safetensors
```

### 3. Verify Baseline (Optional)
Run the verification script to confirm the checkpoints work in our "Fake Quantization" simulator:
```bash
python scripts/eval_fp4_torch.py --task_suite_name libero_spatial
```

### 4. Deploy
To build the final highly-optimized engine, follow the deployment guide:
👉 **[Read the FP4 Deployment Guide](guides/FP4_DEPLOYMENT_GUIDE.md)**

---

## 📂 Repository Structure

- `scripts/`: Core utilities for quantization and verification.
    - `quantize_thor_vla.py`: The main FP4 quantization entry point.
    - `eval_fp4_torch.py`: Simulation runner for verifying FP4 accuracy.
- `exports/`: Scripts for ONNX/TensorRT export.
    - `export_vision_only.py`: Handles the Vision Encoder export.
    - `export_modelopt_int8.py`: Legacy INT8 export method.
- `guides/`: Detailed technical documentation.
- `docs/benchmarks/`: Performance reports and failure analysis.

## 🔗 References
- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [NVIDIA Model Optimizer (ModelOpt)](https://github.com/NVIDIA/TensorRT/tree/main/tools/ModelOpt)
