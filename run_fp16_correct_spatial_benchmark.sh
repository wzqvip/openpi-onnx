#!/bin/bash

# FP16 Correct spatial evaluation

source /home/taco/.venv/bin/activate

cd /home/taco/openpi-onnx

# Start TensorRT server
python3 scripts/serve_trt.py \
    --engine_path checkpoints/pi05_libero_onnx_compat/engine_fp16_correct.trt \
    --port 8003 &

SERVER_PID=$!
sleep 15

# Run LIBERO evaluation
python3 scripts/eval_libero_trt_v1.py \
    --task-suite-name libero_spatial \
    --port 8003 \
    --num-trials-per-task 20

# Kill server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
