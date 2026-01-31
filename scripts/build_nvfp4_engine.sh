#!/bin/bash
set -e

ONNX_PATH="checkpoints/pi05_libero_onnx_compat/model.nvfp4.modelopt.cleaned.onnx"
ENGINE_PATH="checkpoints/pi05_libero_onnx_compat/engine_nvfp4.trt"

mkdir -p $(dirname $ENGINE_PATH)

echo "Building NVFP4 TensorRT Engine..."
echo "Input ONNX: $ONNX_PATH"
echo "Output Engine: $ENGINE_PATH"

# Run trtexec
# --stronglyTyped is required for explicit quantization (QDQ) models from ModelOpt
# --fp4 enables FP4 precision support on Thor (if available in trtexec, otherwise --int4 or rely on QDQ)
# Checking logs: trtexec flags for FP4 might be specific.
# If --fp4 is not valid, we rely on QDQ + stronglyTyped.
# Note: TensorRT 10.0+ supports FP4.

/usr/src/tensorrt/bin/trtexec \
    --onnx=$ONNX_PATH \
    --saveEngine=$ENGINE_PATH \
    --fp4 \
    --stronglyTyped \
    --verbose \
    --warmUp=2000 \
    --duration=10
