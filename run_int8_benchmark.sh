#!/bin/bash
# INT8 TensorRT Benchmark - Standardized Testing
set -e

NUM_TRIALS=20
SEED=42
ENGINE_PATH="checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine"
WS_PORT=8016
WS_URL="ws://localhost:${WS_PORT}"
LIBERO_PATH="/home/taco/openpi-onnx/third_party/libero"
SUITES=("libero_spatial" "libero_goal" "libero_object" "libero_10")

echo "======================================================================"
echo "  INT8 TensorRT Benchmark (20 trials per task)"
echo "======================================================================"

# Use the main venv with all dependencies installed
source /home/taco/.venv/bin/activate
mkdir -p benchmark_logs benchmark_results

# Start TensorRT inference server
echo ""
echo "Starting TensorRT inference server on port $WS_PORT..."
python3 scripts/serve_trt.py \
    --engine_path="$ENGINE_PATH" \
    --port="$WS_PORT" > benchmark_logs/trt_server.log 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Waiting for server to initialize (10 seconds)..."
sleep 10

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: TensorRT server failed to start!"
    cat benchmark_logs/trt_server.log
    exit 1
fi

echo "✓ Server started successfully"

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping TensorRT server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "✓ Server stopped"
}

# Register cleanup on exit
trap cleanup EXIT INT TERM

suite_num=0
total_suites=${#SUITES[@]}

for suite in "${SUITES[@]}"; do
    suite_num=$((suite_num + 1))
    echo ""
    echo "======================================================================"
    echo "[$suite_num/$total_suites] Starting: $suite"
    echo "======================================================================"
    log_file="benchmark_logs/int8_${suite##*_}_20trials.log"
    
    START=$(date +%s)
    PYTHONPATH="$LIBERO_PATH:$PYTHONPATH" \
    python3 scripts/eval_libero_trt.py \
        --task_suite_name="$suite" \
        --num_trials_per_task="$NUM_TRIALS" \
        --ws_url="$WS_URL" \
        --seed="$SEED" \
        2>&1 | tee "$log_file"
    
    END=$(date +%s)
    DURATION=$((END - START))
    
    echo ""
    echo "✓ Completed: $suite (took ${DURATION}s)"
    echo "Results:"
    grep "Total Success Rate\|Latency (ms)" "$log_file" | tail -2
    echo ""
    sleep 5
done

echo ""
echo "======================================================================"
echo "  INT8 Benchmark Complete!"
echo "======================================================================"
