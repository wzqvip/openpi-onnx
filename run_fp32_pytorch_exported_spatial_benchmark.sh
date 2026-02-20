#!/bin/bash
# 评估PyTorch导出的FP32模型

set -e

export PYTHONUNBUFFERED=1
source ~/.venv/bin/activate

cd /home/taco/openpi-onnx

ENGINE_PATH="checkpoints/pi05_libero_onnx_compat/engine_fp32_pytorch_exported.trt"

if [ ! -f "$ENGINE_PATH" ]; then
    echo "ERROR: Engine not found at $ENGINE_PATH"
    exit 1
fi

echo "启动PyTorch导出FP32 TensorRT服务器..."
echo "引擎: $ENGINE_PATH"

# Start server on port 8006
python3 scripts/serve_trt.py --engine_path "$ENGINE_PATH" --port 8006 > /tmp/serve_fp32_pytorch_exported.log 2>&1 &
SERVER_PID=$!

sleep 5

if ! ps -p $SERVER_PID > /dev/null; then
    echo "ERROR: Server failed to start"
    cat /tmp/serve_fp32_pytorch_exported.log
    exit 1
fi

echo "✓ 服务器已启动 (PID: $SERVER_PID, Port: 8006)"

# Run evaluation
echo "运行评估..."
python3 scripts/eval_libero_trt_v1.py \
    --task-suite-name libero_spatial \
    --port 8006 \
    --num-trials-per-task 20 \
    2>&1 | tee benchmark_logs/fp32_pytorch_exported_spatial_20trials.log

# Kill server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "✓ 评估完成"
