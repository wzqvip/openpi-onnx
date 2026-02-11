#!/bin/bash
# INT8 TensorRT Benchmark - Using eval_libero_trt_v1.py
# This uses the proven working method from previous successful INT8 tests
set -e

NUM_TRIALS=20
SEED=42
CHECKPOINT="checkpoints/pi05_libero_pytorch"
CONFIG="pi05_libero"
LIBERO_PATH="/home/taco/openpi-onnx/third_party/libero"
SUITES=("libero_spatial" "libero_goal" "libero_object" "libero_10")
TRT_PORT=8000
ENGINE_PATH="checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine"

echo "======================================================================"
echo "  INT8 TensorRT Benchmark (20 trials per task) - Using v1 method"
echo "======================================================================"

# Use the main venv with all dependencies installed
source /home/taco/.venv/bin/activate
mkdir -p benchmark_logs benchmark_results

# Cleanup function
cleanup() {
    echo "Cleaning up TensorRT server..."
    pkill -f "serve_trt.py" || true
    sleep 2
}
trap cleanup EXIT

# Start TensorRT server
echo "Starting TensorRT server on port $TRT_PORT..."
python3 scripts/serve_trt.py \
    --engine_path "$ENGINE_PATH" \
    --port $TRT_PORT \
    > benchmark_logs/trt_server.log 2>&1 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

# Wait for server to be ready
sleep 5
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: TensorRT server failed to start"
    cat benchmark_logs/trt_server.log
    exit 1
fi

suite_num=0
total_suites=${#SUITES[@]}

for suite in "${SUITES[@]}"; do
    suite_num=$((suite_num + 1))
    echo ""
    echo "======================================================================"
    echo "[$suite_num/$total_suites] Starting: $suite"
    echo "======================================================================"
    log_file="benchmark_logs/int8_${suite##*_}_20trials_v1.log"
    
    START=$(date +%s)
    PYTHONPATH="$LIBERO_PATH:$PYTHONPATH" \
    python3 scripts/eval_libero_trt_v1.py \
        --checkpoint-dir="$CHECKPOINT" \
        --config-name="$CONFIG" \
        --task-suite-name="$suite" \
        --num-trials-per-task="$NUM_TRIALS" \
        --seed="$SEED" \
        2>&1 | tee "$log_file"
    
    END=$(date +%s)
    DURATION=$((END - START))
    
    echo ""
    echo "✓ Completed: $suite (took ${DURATION}s)"
    echo "Results:"
    grep "Total Success Rate\|Latency (ms)\|Total episodes" "$log_file" | tail -3
    echo ""
    sleep 5
done

echo ""
echo "======================================================================"
echo "  INT8 Benchmark Complete!"
echo "======================================================================"
