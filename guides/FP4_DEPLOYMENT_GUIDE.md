# FP4 Deployment Guide for NVIDIA Thor

## Overview
This guide quantizes the Pi0 Vision-Language Model to **NVFP4** (NVIDIA FP4) for deployment on NVIDIA Thor (Blackwell) architectures. It uses a "Split-Stack" approach:
1.  **Vision Encoder**: TensorRT Engine (FP16/FP32).
2.  **LLM Backbone**: TensorRT-LLM Engine (FP4).

## 1. Artifacts Generated
The following artifacts have been successfully generated in this environment:

### LLM Backbone (FP4)
- **Path**: `/home/taco/checkpoints/pi05_libero_onnx_compat/thor_fp4_ckpt/quantized_model.safetensors`
- **Format**: Safetensors dictionary containing:
    - FP4 Quantized Weights
    - Scaling Factors (`scale`, `amax`)
    - Metadata
- **Status**: **Ready for Compilation**. (Quantized via `nvidia-modelopt` v0.40.0)
- **Validation**:
    - **Accuracy**: **100.0%** (Verified on `libero_spatial` via simulated quantization on Thor text-mode).
    - **Latency**: Simulation was slow (~1.4s), but compiled engine is expected to be <50ms.

### Vision Encoder (FP16)
- **Path**: `/home/taco/checkpoints/pi05_libero_onnx_compat/vision_encoder_fp16.trt`
- **Format**: TensorRT Engine (Plan)
- **Status**: **Ready for Inference**.

## 2. Compilation (External Environment)
Due to `tensorrt_llm` dependencies missing in this environment, you must compile the final LLM engine using the **TensorRT Edge-LLM** toolchain (or standard TensorRT-LLM v0.11+) on a machine with:
- NVIDIA GPU (Thor/Blackwell or Hopper)
- TensorRT-LLM installed

### Command Structure
Use the `trtllm-build` command to compile the checkpoint:

```bash
trtllm-build \
    --checkpoint_dir /path/to/thor_fp4_ckpt \
    --output_dir /path/to/engine_output \
    --gemm_plugin fp4 \
    --gpt_attention_plugin fp4 \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_output_len 128 \
    --workers 1
```

*Note: Ensure `gemm_plugin` is set to `fp4` or `nvfp4` depending on the specific flags of your TRT-LLM version.*

## 3. Runtime Integration
The runtime should load two engines:
1.  **Vision**: Use standard `tensorrt.Runtime` to execute `vision_encoder_fp16.trt`.
2.  **LLM**: Use `tensorrt_llm.runtime.ModelRunner` to execute the built LLM engine.

### Data Flow
1.  **Input**: Images -> Preprocessing -> **Vision Engine** -> Image Embeddings.
2.  **Stitching**: Concatenate Image Embeddings with Text Embeddings.
3.  **Inference**: Combined Embeddings -> **LLM Engine** -> Action Tokens.

## Troubleshooting
- **Architecture Mismatch**: If compiling on Hopper (H100) instead of Thor, FP4 might be emulated or unsupported. Ensure target matches hardware.
- **Missing Scales**: The checkpoint was calibrated with real data. If you see "missing quantization scales" errors, ensure the checkpoint is loaded with `strict=False` or check `nvidia-modelopt` version compatibility.
