#!/bin/bash
set -e

# Define paths
ONNX_PATH="checkpoints/pi05_libero_onnx_compat/model.fp32.modelopt.cleaned.onnx"
ENGINE_PATH="checkpoints/pi05_libero_onnx_compat/engine_fp16.trt"

echo "Building FP16 TensorRT Engine..."
echo "Input ONNX: $ONNX_PATH"
echo "Output Engine: $ENGINE_PATH"

/usr/src/tensorrt/bin/trtexec \
    --onnx=$ONNX_PATH \
    --saveEngine=$ENGINE_PATH \
    --fp16 \
    --warmUp=100 \
    --duration=0 \
    --verbose \
    > logs/build_fp16.log 2>&1

echo "Build complete. Check logs/build_fp16.log"
