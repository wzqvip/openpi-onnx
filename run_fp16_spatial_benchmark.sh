#!/bin/bash
# FP16 TensorRT Benchmark - Spatial Suite Only
set -e

NUM_TRIALS=20
SEED=42
CHECKPOINT="checkpoints/pi05_libero_pytorch"
CONFIG="pi05_libero"
LIBERO_PATH="/home/taco/openpi-onnx/third_party/libero"
SUITE="libero_spatial"
TRT_PORT=8001
ENGINE_PATH="checkpoints/pi05_libero_onnx_compat/engine_fp16.trt"

echo "======================================================================"
echo "  FP16 TensorRT Spatial Benchmark (20 trials per task)"
echo "======================================================================"

source /home/taco/.venv/bin/activate
mkdir -p benchmark_logs benchmark_results

# Cleanup function
cleanup() {
    echo "Cleaning up TensorRT server..."
    pkill -f "serve_trt.py.*$TRT_PORT" || true
    sleep 2
}
trap cleanup EXIT

# Start TensorRT server
echo "Starting FP16 TensorRT server on port $TRT_PORT..."
python3 scripts/serve_trt.py \
    --engine_path "$ENGINE_PATH" \
    --port $TRT_PORT \
    > benchmark_logs/trt_server_fp16.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait for server to be ready
sleep 5
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: TensorRT server failed to start"
    cat benchmark_logs/trt_server_fp16.log
    exit 1
fi

echo "======================================================================"
echo "Starting: $SUITE"
echo "======================================================================"
log_file="benchmark_logs/fp16_spatial_20trials.log"

START=$(date +%s)
PYTHONPATH="$LIBERO_PATH:$PYTHONPATH" \
python3 scripts/eval_libero_trt_v1.py \
    --checkpoint-dir="$CHECKPOINT" \
    --config-name="$CONFIG" \
    --task-suite-name="$SUITE" \
    --num-trials-per-task="$NUM_TRIALS" \
    --seed="$SEED" \
    --host="localhost" \
    --port="$TRT_PORT" \
    2>&1 | tee "$log_file"

END=$(date +%s)
DURATION=$((END - START))

echo ""
echo "✓ Completed: $SUITE (took ${DURATION}s)"
echo "Results:"
grep "Total Success Rate\|Latency (ms)\|Total episodes" "$log_file" | tail -3
echo ""

echo "======================================================================"
echo "FP16 Spatial Benchmark Complete!"
echo "======================================================================"
