#!/bin/bash
# FP8 Fixed Spatial 评估（20 trials）

set -e

# 激活虚拟环境
source /home/taco/.venv/bin/activate

cd /home/taco/openpi-onnx

# 配置
ENGINE_PATH="checkpoints/pi05_libero_onnx_compat/engine_fp8_fixed.trt"
CHECKPOINT_DIR="checkpoints/pi05_libero_pytorch"
PORT=8002
SUITE="libero_spatial"
TRIALS=20
LOG_FILE="benchmark_logs/fp8_fixed_spatial_${TRIALS}trials.log"

echo "启动 FP8 Fixed TensorRT 服务器（端口 ${PORT}）..."
python3 scripts/serve_trt.py \
    --engine_path "${ENGINE_PATH}" \
    --port ${PORT} \
    > benchmark_logs/trt_server_fp8_fixed.log 2>&1 &

SERVER_PID=$!
echo "服务器已启动（PID: ${SERVER_PID}）"
sleep 5

echo "开始评估 ${SUITE} 套件（${TRIALS} trials）..."
python3 scripts/eval_libero_trt_v1.py \
    --checkpoint-dir="${CHECKPOINT_DIR}" \
    --config-name=pi05_libero \
    --task-suite-name="${SUITE}" \
    --num-trials-per-task=${TRIALS} \
    --seed=42 \
    --host=localhost \
    --port=${PORT}

echo "评估完成！"
kill ${SERVER_PID} 2>/dev/null || true
