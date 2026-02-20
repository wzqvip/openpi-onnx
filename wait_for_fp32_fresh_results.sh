#!/bin/bash
# 等待Fresh FP32评估完成并收集结果

LOG_FILE="/home/taco/openpi-onnx/benchmark_logs/fp32_fresh_spatial_20trials.log"
MASTER_LOG="/home/taco/openpi-onnx/benchmark_logs/fp32_fresh_eval_v3.log"

echo "等待FP32 Fresh评估完成..."
echo "日志文件: $LOG_FILE"

# 等待日志文件出现
timeout 3600 bash -c "while [ ! -f '$LOG_FILE' ]; do sleep 10; done" || {
    echo "超时：评估日志未生成"
    tail -50 "$MASTER_LOG"
    exit 1
}

# 等待评估完成（查找完成标记）
timeout 7200 bash -c "while ! grep -q 'TASK.*COMPLETE' '$LOG_FILE'; do sleep 30; done" || {
    echo "超时：评估未完成"
    tail -100 "$LOG_FILE"
    exit 1
}

# 提取结果
echo ""
echo "================================"
echo "✓ FP32 Fresh 评估完成"
echo "================================"
echo ""

# 收集所有task的完成情况
echo "任务完成统计："
grep "TASK.*COMPLETE" "$LOG_FILE" | tee /tmp/fp32_fresh_results.txt

echo ""
echo "详细日志："
tail -100 "$LOG_FILE"
