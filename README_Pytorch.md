# PyTorch Workflow for Pi0.5 (Jetson/SBSA)

This document outlines the scripts and steps for the PyTorch implementation of the Pi0.5 model, including conversion, validation, optimization, and benchmarking.

## Scripts & Functions

### 1. Model Conversion & Validation
| Script | Function | Key Details |
| :--- | :--- | :--- |
| `scripts/convert_jax_to_torch.py` | Converts the original JAX checkpoint to PyTorch FP32 format (`model.safetensors`). | Base conversion step. |
| `test_pytorch_fp32.py` | Validates the correctness of the converted FP32 model by running inference. | Checks inference logic, shapes, and basic output sanity. |

### 2. Precision Optimization
| Script | Function | Key Details |
| :--- | :--- | :--- |
| `scripts/convert_to_bf16.py` | Converts the FP32 model to BFloat16 (`model.pt`). | Saves space (50%) and uses native training precision. Uses `torch.save` to handle shared tensors correctly. |
| `scripts/quantize_fp8_torchao.py` | Quantizes the BF16 model to FP8 using `torchao`. | Performs weight-only quantization. Saves as `model.pt`. |

### 3. Benchmarking
| Script | Function | Key Details |
| :--- | :--- | :--- |
| `scripts/benchmark_precision.py` | Measures latency and accuracy (MSE) of FP32, BF16, and FP8 models. | Runs on GPU with CuDNN disabled for stability. Compares outputs against FP32 baseline. |

## Evaluation Results

Benchmarks were conducted on an **NVIDIA Thor GPU** (CuDNN disabled for stability).

| Precision | Latency (avg) | Speed vs FP32 | MSE vs FP32 | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | **~408 ms** | 1.0× | — | ✅ Working | Baseline |
| **BF16** | ~250 ms | **1.63×** | 2.11 | ✅ Working | Best latency; requires `autocast` + explicit input casting |
| **INT8** | ~466 ms | 0.88× | 2.02 | ✅ Working | 35% smaller file (7.0→4.6 GB); slower without Triton kernels |

## Quick Start

1.  **Download & Convert**:
    ```bash
    python scripts/convert_jax_to_torch.py
    ```

2.  **Validate**:
    ```bash
    python test_pytorch_fp32.py
    ```

3.  **Optimize (BF16 & FP8)**:
    ```bash
    python scripts/convert_to_bf16.py
    python scripts/quantize_fp8_torchao.py
    ```

4.  **Benchmark**:
    ```bash
    python scripts/benchmark_precision.py
    ```
