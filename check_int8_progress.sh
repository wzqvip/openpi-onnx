#!/bin/bash
# 快速查看INT8评估进度

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         INT8 模型评估进度监控                              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 检查进程
echo "📊 进程状态:"
if ps aux | grep -q "[e]val_libero_trt_v1.py"; then
    echo "  ✅ 评估正在运行"
    EVAL_PID=$(ps aux | grep "[e]val_libero_trt_v1.py" | awk '{print $2}')
    echo "  PID: ${EVAL_PID}"
else
    echo "  ❌ 评估未运行"
fi

if ps aux | grep -q "[s]erve_trt.py"; then
    echo "  ✅ TensorRT服务运行中"
else
    echo "  ❌ TensorRT服务未运行"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 统计各套件进度
echo "📈 任务套件完成情况:"
echo ""

TOTAL_COMPLETED=0
TOTAL_TASKS=40

for suite in spatial object goal 10; do
    LOG=$(ls -t /tmp/eval_int8_libero_${suite}_*.log 2>/dev/null | head -1)
    
    if [ -n "$LOG" ]; then
        # 统计已完成任务
        COMPLETED=$(grep -c "TASK.*COMPLETE" "$LOG" 2>/dev/null || echo "0")
        TOTAL_COMPLETED=$((TOTAL_COMPLETED + COMPLETED))
        
        # 显示进度条
        PERCENT=$((COMPLETED * 10))
        BAR=""
        for i in {1..10}; do
            if [ $i -le $COMPLETED ]; then
                BAR="${BAR}█"
            else
                BAR="${BAR}░"
            fi
        done
        
        if [ $COMPLETED -eq 10 ]; then
            STATUS="✅"
        elif [ $COMPLETED -gt 0 ]; then
            STATUS="🔄"
        else
            STATUS="⏳"
        fi
        
        printf "  %-20s %s [%s] %2d/10\n" "libero_${suite}" "${STATUS}" "${BAR}" "${COMPLETED}"
    else
        printf "  %-20s ⏳ [░░░░░░░░░░]  0/10\n" "libero_${suite}"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 总进度
TOTAL_PERCENT=$((TOTAL_COMPLETED * 100 / TOTAL_TASKS))
echo "🎯 总体进度: ${TOTAL_COMPLETED}/${TOTAL_TASKS} 任务 (${TOTAL_PERCENT}%)"

# 时间估算
if [ $TOTAL_COMPLETED -gt 0 ]; then
    # 假设每个任务约12分钟 (10任务×20次试验/10 = 20次试验，约240秒/试验)
    REMAINING=$((TOTAL_TASKS - TOTAL_COMPLETED))
    EST_MINUTES=$((REMAINING * 12))
    EST_HOURS=$((EST_MINUTES / 60))
    EST_MIN_LEFT=$((EST_MINUTES % 60))
    
    if [ $EST_HOURS -gt 0 ]; then
        echo "⏱️  预计剩余: ~${EST_HOURS}小时${EST_MIN_LEFT}分钟"
    else
        echo "⏱️  预计剩余: ~${EST_MIN_LEFT}分钟"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 最新活动
echo "📝 最新日志 (最后5行):"
if [ -f /tmp/int8_all_suites_master.log ]; then
    tail -5 /tmp/int8_all_suites_master.log | grep -v "DEBUG" | head -3 || echo "  (调试信息...)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 提示:"
echo "  • 持续监控: watch -n 30 '$0'"
echo "  • 查看详细日志: tail -f /tmp/int8_all_suites_master.log"
echo "  • 查看当前任务: tail -f /tmp/eval_int8_libero_*.log | grep 'TASK.*COMPLETE'"
echo ""
