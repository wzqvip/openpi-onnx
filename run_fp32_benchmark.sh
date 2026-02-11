#!/bin/bash
# FP32 PyTorch Benchmark - Standardized Testing
set -e

NUM_TRIALS=20
SEED=42
CHECKPOINT="checkpoints/pi05_libero_pytorch"
CONFIG="pi05_libero"
LIBERO_PATH="/home/taco/openpi-onnx/third_party/libero"
SUITES=("libero_spatial" "libero_goal" "libero_object" "libero_10")

echo "======================================================================"
echo "  FP32 PyTorch Benchmark (20 trials per task)"
echo "======================================================================"

source .venv/bin/activate
mkdir -p benchmark_logs benchmark_results

for suite in "${SUITES[@]}"; do
    echo ""
    echo ">>> Starting: $suite"
    log_file="benchmark_logs/fp32_${suite##*_}_20trials.log"
    
    PYTHONPATH="$LIBERO_PATH:$PYTHONPATH" \
    python3 scripts/eval_libero_torch.py \
        --checkpoint="$CHECKPOINT" \
        --config="$CONFIG" \
        --task_suite_name="$suite" \
        --num_trials_per_task="$NUM_TRIALS" \
        --seed="$SEED" \
        > "$log_file" 2>&1
    
    echo "✓ Completed: $suite"
    grep "Total Success Rate\|Latency (ms)" "$log_file" | tail -2
    sleep 10
done

echo ""
echo "======================================================================"
echo "  FP32 Benchmark Complete!"
echo "======================================================================"
