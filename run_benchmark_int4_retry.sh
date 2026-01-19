
#!/bin/bash
MODEL="dist/final_fp16/model.fp32.onnx"
LOG_INT4_RETRY="benchmark_int4_retry.log"

echo "========================================"
echo "Benchmarking INT4 (Retry with Aggressive Flags)..."
echo "========================================"
# Adding --int8 to allow intermediate fallback (often required for INT4 mixed precision)
# Adding --useCudaGraph for throughput
# Adding --verbose to inspect layer decisions
/usr/src/tensorrt/bin/trtexec \
    --onnx=$MODEL \
    --int4 --int8 --fp16 \
    --avgRuns=20 \
    --duration=0 \
    --iterations=10 \
    --useCudaGraph \
    --verbose \
    > $LOG_INT4_RETRY 2>&1

echo "Benchmark Complete. Log: $LOG_INT4_RETRY"

# Check if any layer actually ran in INT4
echo "Checking for INT4 layers..."
grep "Precision: INT4" $LOG_INT4_RETRY | head -n 10
grep "Layer details:" -A 20 $LOG_INT4_RETRY
