#!/bin/bash
# 
# Build Native FP4/FP8/INT4 TensorRT Engine for Thor/Blackwell
# This script uses trtexec with --best flag to leverage Blackwell's
# native low-precision Tensor Cores (FP4/FP8/INT4)
#

set -e

echo "================================="
echo "TensorRT Engine Builder for Thor"
echo "================================="
echo ""

# Configuration
ONNX_MODEL="/home/taco/checkpoints/pi05_libero_onnx_compat/model.fp32.modelopt.cleaned.onnx"
OUTPUT_DIR="/home/taco/checkpoints/pi05_libero_onnx_compat"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

# Check if ONNX model exists
if [ ! -f "$ONNX_MODEL" ]; then
    echo "ERROR: ONNX model not found at $ONNX_MODEL"
    exit 1
fi

echo "Source ONNX: $ONNX_MODEL"
echo "Output Dir: $OUTPUT_DIR"
echo ""

# Function to build engine
build_engine() {
    local name=$1
    local flags=$2
    local output="${OUTPUT_DIR}/engine_${name}.trt"
    
    echo "============================================"
    echo "Building $name engine..."
    echo "Flags: $flags"
    echo "Output: $output"
    echo "============================================"
    
    $TRTEXEC \
        --onnx=$ONNX_MODEL \
        --saveEngine=$output \
        $flags \
        --verbose \
        --warmUp=2000 \
        --duration=10 \
        --avgRuns=10 \
        2>&1 | tee "${OUTPUT_DIR}/trtexec_${name}.log"
    
    if [ -f "$output" ]; then
        local size=$(ls -lh "$output" | awk '{print $5}')
        echo "✅ Success! Engine size: $size"
        echo ""
    else
        echo "❌ Failed to build $name engine"
        echo ""
        return 1
    fi
}

# Build engines with different precisions
echo "Thor/Blackwell supports: FP32, FP16, BF16, FP8, INT8, INT4"
echo ""

# Option 1: Best Performance (auto-select optimal precision)
echo "[1/4] Building BEST engine (auto-selects FP4/FP8/INT4/INT8/FP16)..."
build_engine "thor_best" "--best"

# Option 2: FP8 (Blackwell native)
echo "[2/4] Building FP8 engine (Blackwell native)..."
build_engine "thor_fp8" "--fp8 --best"

# Option 3: INT4 (closest to FP4, uses Blackwell INT4 Tensor Cores)
echo "[3/4] Building INT4 engine (uses Blackwell INT4 Tensor Cores)..."
build_engine "thor_int4" "--int4 --best"

# Option 4: INT4 + FP8 combined
echo "[4/4] Building INT4+FP8 engine (hybrid precision)..."
build_engine "thor_int4_fp8" "--int4 --fp8 --best"

echo ""
echo "================================="
echo "Build Summary"
echo "================================="
ls -lh ${OUTPUT_DIR}/engine_thor*.trt 2>/dev/null || echo "No engines built"
echo ""
echo "Next steps:"
echo "1. Test each engine with eval script"
echo "2. Compare latency and accuracy on libero_spatial"
echo "3. Select the best precision/performance tradeoff"
echo ""
echo "Recommended for Thor/Blackwell:"
echo "  - engine_thor_best.trt: Auto-optimized"
echo "  - engine_thor_fp8.trt: FP8 native Tensor Cores"
echo "  - engine_thor_int4.trt: INT4 for maximum throughput"
