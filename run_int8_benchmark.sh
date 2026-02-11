#!/bin/bash
# INT8 TensorRT Benchmark - Standardized Testing
set -e

NUM_TRIALS=20
SEED=42
ENGINE="model.int8.modelopt.engine"
CONFIG="pi05_libero"
LIBERO_PATH="/home/taco/openpi-onnx/third_party/libero"
SUITES=("libero_spatial" "libero_goal" "libero_object" "libero_10")

echo "======================================================================"
echo "  INT8 TensorRT Benchmark (20 trials per task)"
echo "======================================================================"

source .venv/bin/activate
mkdir -p benchmark_logs benchmark_results

suite_num=0
total_suites=${#SUITES[@]}

for suite in "${SUITES[@]}"; do
    ((suite_num++))
    echo ""
    echo "======================================================================"
    echo "[$suite_num/$total_suites] Starting: $suite"
    echo "======================================================================"
    log_file="benchmark_logs/int8_${suite##*_}_20trials.log"
    
    START=$(date +%s)
    PYTHONPATH="$LIBERO_PATH:$PYTHONPATH" \
    python3 scripts/eval_libero_trt.py \
        --engine="$ENGINE" \
        --config="$CONFIG" \
        --task_suite_name="$suite" \
        --num_trials_per_task="$NUM_TRIALS" \
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
