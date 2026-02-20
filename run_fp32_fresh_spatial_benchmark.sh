#!/bin/bash
# Run FP32 Fresh evaluation on LIBERO spatial task

set -e

export PYTHONUNBUFFERED=1

# Virtual environment
source ~/.venv/bin/activate

cd /home/taco/openpi-onnx

# Engine path
ENGINE_PATH="checkpoints/pi05_libero_onnx_compat/engine_fp32_fresh.trt"

# Check if engine exists
if [ ! -f "$ENGINE_PATH" ]; then
    echo "ERROR: Engine not found at $ENGINE_PATH"
    exit 1
fi

echo "Starting FP32 Fresh TensorRT Server..."
echo "Engine: $ENGINE_PATH"

# Start server
python3 scripts/serve_trt.py --engine_path "$ENGINE_PATH" --port 8005 > /tmp/serve_fp32_fresh.log 2>&1 &
SERVER_PID=$!

sleep 5

# Check if server is running
if ! ps -p $SERVER_PID > /dev/null; then
    echo "ERROR: Server failed to start"
    cat /tmp/serve_fp32_fresh.log
    exit 1
fi

echo "✓ Server started (PID: $SERVER_PID, Port: 8005)"

# Run evaluation
echo "Running evaluation..."
python3 scripts/eval_libero_trt_v1.py \
    --task-suite-name libero_spatial \
    --port 8005 \
    --num-trials-per-task 20 \
    2>&1 | tee benchmark_logs/fp32_fresh_spatial_20trials.log

# Kill server
kill $SERVER_PID || true
wait $SERVER_PID 2>/dev/null || true

echo "✓ Evaluation complete"
