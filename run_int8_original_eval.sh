#!/bin/bash
#
# 运行原版INT8评估脚本（100%成功率版本）
# 基于 commit 68672fe

set -e

# 激活虚拟环境
source /home/taco/.venv/bin/activate

cd /home/taco/openpi-onnx

echo "================================"
echo "INT8模型完整评估 (原版脚本)"
echo "================================"

# 配置
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-20}"
LOG_FILE="/tmp/eval_int8_original_${TASK_SUITE}_$(date +%Y%m%d_%H%M%S).log"

# 1. 检查引擎文件
ENGINE="/home/taco/checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine"
if [ ! -f "$ENGINE" ]; then
    echo "❌ INT8 引擎文件不存在: $ENGINE"
    echo "请运行: python exports/export_modelopt_int8.py"
    exit 1
fi

echo "✅ INT8 引擎文件存在 ($(ls -lh $ENGINE | awk '{print $5}'))"

# 2. 启动TensorRT服务器 (后台)
echo ""
echo "🚀 启动 TensorRT 服务器..."
python scripts/serve_trt.py \
    --engine_path="$ENGINE" \
    --port=8012 \
    > /tmp/serve_trt_int8_original.log 2>&1 &

SERVER_PID=$!
echo "服务器 PID: $SERVER_PID"

# 等待服务器启动
sleep 5

# 检查服务器是否在运行
if ! ps -p $SERVER_PID > /dev/null; then
    echo "❌ 服务器启动失败，查看日志:"
    tail -20 /tmp/serve_trt_int8_original.log
    exit 1
fi

echo "✅ 服务器已启动"

# 3. 运行评估 (原版脚本)
echo ""
echo "📊 开始评估 (${TASK_SUITE}, ${TRIALS_PER_TASK} trials per task)..."
echo ""

python scripts/eval_libero_trt_v1.py \
    --task_suite_name="${TASK_SUITE}" \
    --num_trials_per_task="${TRIALS_PER_TASK}" \
    --port=8012 \
    --seed=7 \
    2>&1 | tee "${LOG_FILE}"

EVAL_STATUS=$?

# 4. 清理
echo ""
echo "🛑 停止服务器..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

if [ $EVAL_STATUS -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✅ 评估完成！"
    echo "================================"
    echo "日志文件: ${LOG_FILE}"
else
    echo ""
    echo "================================"
    echo "❌ 评估失败 (退出码: $EVAL_STATUS)"
    echo "================================"
    exit $EVAL_STATUS
fi
