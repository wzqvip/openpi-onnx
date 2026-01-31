# PyTorch Installation Guide for NVIDIA Thor / Jetson

## Overview
This guide covers the installation of PyTorch 2.9.1 (CUDA 13.0) for NVIDIA Thor (SBSA / ARM64) to enable native FP8 and FP4 quantization support.

## Prerequisites
- **Hardware**: NVIDIA Thor or Jetson Orin (ARM64/SBSA)
- **OS**: Ubuntu 22.04 / JetPack 6.x
- **CUDA**: 13.0
- **System Dependencies**: **NVIDIA Performance Libraries (NVPL)**
    - *Critical*: The PyTorch wheel depends on `libnvpl_lapack_lp64_gomp.so.0`.
    - This library is **NOT** bundled with the pip wheel.
    - It must be installed via the system package manager or NVIDIA HPC SDK.

## Installation Steps

### 1. Install System Dependencies (JetPack / NVPL)
> [!IMPORTANT]
> The `nvidia-jetpack` packages in some repositories **DO NOT** include the following libraries:
> - `libnvpl_lapack` (NVPL)
> - `libcudss.so.0` (cuDSS)
> You typically need to install them manually via `.deb` files from the NVIDIA Developer site.

**Verified working versions for Ubuntu 24.04 (ARM64):**
- NVPL: `nvpl-local-repo-ubuntu2404-25.11_1.0-1_arm64.deb`
- cuDSS: `cudss-local-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb`

Check if available:
```bash
sudo apt-get install nvidia-jetpack-runtime
# OR
sudo apt-get install libnvpl-dev libcudss-dev
```
If `apt` fails, download and install the respective `.deb` installers.

*Verification:*
```bash
ldconfig -p | grep libnvpl_lapack
```

### 2. Install PyTorch Wheel
Install the custom wheel from the Jetson AI Lab index:
```bash
pip install "https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/e65/21b6628938478/torch-2.9.1-cp312-cp312-linux_aarch64.whl"
```

### 3. Verify Installation
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

## Troubleshooting
**Error**: `ImportError: libnvpl_lapack_lp64_gomp.so.0: cannot open shared object file`
**Solution**: This confirms NVPL is missing. Install `nvidia-jetpack-runtime`.
