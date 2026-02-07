#!/bin/bash
#
# 监控INT8评估进度并在完成后生成README

CHECK_INTERVAL=300  # 5分钟检查一次

echo "开始监控INT8评估进度..."
echo "检查间隔: ${CHECK_INTERVAL}秒"
echo ""

while true; do
    # 检查评估进程是否还在运行
    if ! ps aux | grep -q "[e]val_libero_trt_v1.py"; then
        echo ""
        echo "========================================"
        echo "✅ 评估已完成！"
        echo "完成时间: $(date)"
        echo "========================================"
        
        # 汇总结果
        echo ""
        echo "正在汇总结果..."
        
        # 查找所有评估日志
        SPATIAL_LOG=$(ls -t /tmp/eval_int8_libero_spatial_*.log 2>/dev/null | head -1)
        OBJECT_LOG=$(ls -t /tmp/eval_int8_libero_object_*.log 2>/dev/null | head -1)
        GOAL_LOG=$(ls -t /tmp/eval_int8_libero_goal_*.log 2>/dev/null | head -1)
        TEN_LOG=$(ls -t /tmp/eval_int8_libero_10_*.log 2>/dev/null | head -1)
        
        echo ""
        echo "找到的日志文件:"
        echo "  libero_spatial: ${SPATIAL_LOG}"
        echo "  libero_object:  ${OBJECT_LOG}"
        echo "  libero_goal:    ${GOAL_LOG}"
        echo "  libero_10:      ${TEN_LOG}"
        echo ""
        
        # 生成汇总报告
        SUMMARY_FILE="/tmp/int8_all_suites_summary.txt"
        
        cat > ${SUMMARY_FILE} << 'EOF'
========================================
INT8 模型完整评估结果汇总
========================================
评估日期: $(date '+%Y年%m月%d日')
试验次数: 每任务20次

EOF
        
        for SUITE in spatial object goal 10; do
            LOG_VAR="${SUITE^^}_LOG"
            LOG_FILE="${!LOG_VAR}"
            
            if [ -n "${LOG_FILE}" ] && [ -f "${LOG_FILE}" ]; then
                echo "【libero_${SUITE}】" >> ${SUMMARY_FILE}
                echo "----------------------------------------" >> ${SUMMARY_FILE}
                grep "TASK.*COMPLETE" "${LOG_FILE}" >> ${SUMMARY_FILE} 2>/dev/null || echo "未找到完成记录" >> ${SUMMARY_FILE}
                
                # 计算总成功率
                TOTAL=$(grep "TASK.*COMPLETE" "${LOG_FILE}" 2>/dev/null | awk -F'[:/]' '{sum+=$2; total+=$3} END {printf "%d/%d (%.2f%%)", sum, total, 100*sum/total}')
                echo "总成功率: ${TOTAL}" >> ${SUMMARY_FILE}
                echo "" >> ${SUMMARY_FILE}
            else
                echo "【libero_${SUITE}】" >> ${SUMMARY_FILE}
                echo "----------------------------------------" >> ${SUMMARY_FILE}
                echo "❌ 日志文件未找到或评估未完成" >> ${SUMMARY_FILE}
                echo "" >> ${SUMMARY_FILE}
            fi
        done
        
        cat ${SUMMARY_FILE}
        
        echo ""
        echo "汇总报告已保存: ${SUMMARY_FILE}"
        echo ""
        echo "现在可以生成README文件了！"
        
        break
    fi
    
    # 显示当前进度
    CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    COMPLETED_TASKS=$(grep -c "TASK.*COMPLETE" /tmp/eval_int8_libero_*.log 2>/dev/null || echo "0")
    
    echo "[${CURRENT_TIME}] 进度: ${COMPLETED_TASKS} 个任务完成..."
    
    # 等待下次检查
    sleep ${CHECK_INTERVAL}
done
