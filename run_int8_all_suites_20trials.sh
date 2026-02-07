#!/bin/bash
#
# 运行INT8模型在全部4个LIBERO任务套件上的完整评估
# 每个任务20次试验取平均值

set -e

# 激活虚拟环境
source /home/taco/.venv/bin/activate

cd /home/taco/openpi-onnx

ENGINE="/home/taco/checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine"
TRIALS=20
PORT=8012

echo "========================================"
echo "INT8 模型完整评估 - 全部4个任务套件"
echo "========================================"
echo "试验次数: ${TRIALS} per task"
echo "开始时间: $(date)"
echo ""

# 启动TensorRT服务器
echo "🚀 启动 TensorRT 服务器 (端口 ${PORT})..."
python scripts/serve_trt.py \
    --engine_path="${ENGINE}" \
    --port=${PORT} \
    > /tmp/serve_trt_int8_all_suites.log 2>&1 &

SERVER_PID=$!
echo "服务器 PID: ${SERVER_PID}"
sleep 5

# 检查服务器
if ! ps -p ${SERVER_PID} > /dev/null; then
    echo "❌ 服务器启动失败"
    tail -20 /tmp/serve_trt_int8_all_suites.log
    exit 1
fi

echo "✅ 服务器已启动"
echo ""

# 任务套件列表
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

# 运行每个套件
for SUITE in "${SUITES[@]}"; do
    echo "========================================"
    echo "📊 评估: ${SUITE}"
    echo "========================================"
    
    LOG_FILE="/tmp/eval_int8_${SUITE}_$(date +%Y%m%d_%H%M%S).log"
    
    python scripts/eval_libero_trt_v1.py \
        --task_suite_name="${SUITE}" \
        --num_trials_per_task=${TRIALS} \
        --port=${PORT} \
        --seed=7 \
        2>&1 | tee "${LOG_FILE}"
    
    STATUS=$?
    
    if [ $STATUS -eq 0 ]; then
        echo "✅ ${SUITE} 完成"
        echo "日志: ${LOG_FILE}"
    else
        echo "❌ ${SUITE} 失败 (退出码: ${STATUS})"
    fi
    echo ""
done

# 停止服务器
echo "🛑 停止 TensorRT 服务器..."
kill ${SERVER_PID} 2>/dev/null || true
wait ${SERVER_PID} 2>/dev/null || true

echo ""
echo "========================================"
echo "✅ 全部评估完成！"
echo "结束时间: $(date)"
echo "========================================"
echo ""
echo "日志文件位置:"
ls -lh /tmp/eval_int8_*.log 2>/dev/null | tail -4
