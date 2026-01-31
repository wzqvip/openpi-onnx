# OpenPI-ONNX: Mobile/Edge VLA Quantization

<p align="center">
  <strong>High-Performance Deployment of Pi0 Vision-Language-Action Models on NVIDIA Jetson</strong>
</p>

## 📖 Overview
This repository provides a complete toolkit for deploying OpenPI (Pi0) models on **NVIDIA Jetson architectures** (Orin, Thor/Blackwell). It features high-performance quantization pipelines for:
*   **INT8 (Orin/Thor)**: Stable, 100% accuracy, 2x-4x speedup.
*   **FP4 (Thor Only)**: Bleeding edge, Blackwell-native precision for maximum throughput.

## 🚀 Getting Started

### Prerequisites
*   **Hardware**: NVIDIA Jetson Orin (AGX/NX) or Thor (Blackwell).
*   **Software**: JetPack 6.0+, PyTorch 2.4+, TensorRT 10.x.

### 1. Installation
Clone this repository and install dependencies:
```bash
git clone https://github.com/wzqvip/openpi-onnx.git
cd openpi-onnx

# Install basic requirements
pip install -r requirements.txt

# Install NVIDIA ModelOpt (for quantization)
pip install nvidia-modelopt[torch]
```

### 2. Model Preparation (JAX -> PyTorch)
Since OpenPI models are trained in JAX, you must first convert the verified checkpoint to a PyTorch-compatible format.

1.  **Download JAX Checkpoint**:
    Download your trained checkpoint (e.g., `pi0_base`) from the [OpenPI Weights](https://github.com/physical-intelligence/openpi) or your S3 bucket.

2.  **Convert to PyTorch**:
    Use the provided utility to convert the JAX params to a Safetensors state dict:
    ```bash
    python scripts/convert_jax_to_torch.py \
        --config pi0_base \
        --checkpoint /path/to/jax/checkpoint \
        --output_dir ./checkpoints/pi0_torch
    ```

### 3. Quantization & Export
Choose your target precision based on your hardware.

#### Option A: INT8 Quantization (Recommended for Orin)
Best for compatibility and verified performance on Orin/Thor.
```bash
# 1. Run Data-Free INT8 Export (using ModelOpt)
python exports/export_modelopt_int8.py \
    --checkpoint ./checkpoints/pi0_torch \
    --output ./checkpoints/pi0_int8

# 2. Compile to TensorRT Engine
trtexec --onnx=./checkpoints/pi0_int8/model.onnx \
        --saveEngine=./checkpoints/pi0_int8/engine_int8.trt \
        --int8 --best
```

#### Option B: FP4 Quantization (Thor/Blackwell Only)
Leverages native FP4 Tensor Cores on NVIDIA Thor.
```bash
# 1. Quantize to FP4 (requires verified environment)
python scripts/quantize_thor_vla.py

# 2. Deploy
# See the detailed guide for compiling the Split-Stack engine:
# guides/FP4_DEPLOYMENT_GUIDE.md
```

### 4. Verification
Verify the accuracy of your quantized model using our evaluation suite:
```bash
python scripts/eval_libero_torch.py --task_suite_name libero_spatial
```

---

## 📊 Benchmark Results

| Platform | Precision | Accuracy | Latency | VRAM | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Orin/Thor** | **INT8** | **100.0%** | **~118 ms** | **4.0 GB** | **Stable** |
| **Thor** | **FP4** | **100.0%** | *< 50 ms* | ~6.0 GB | Verified (Sim) |
| -- | FP32 | 80.0% | ~250 ms | 13.0 GB | Baseline |
| -- | FP16 | 0.0% | N/A | 6.2 GB | **Unstable** |

## 📂 Repository Structure
*   `scripts/`: Core quantization and conversion utilities.
*   `exports/`: ONNX export pipelines.
*   `src/openpi/`: Shared model definitions (PyTorch port of Pi0).
*   `guides/`: Detailed deployment documentation.
