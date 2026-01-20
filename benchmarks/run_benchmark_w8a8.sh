
#!/bin/bash
MODEL="dist/final_fp16/model.fp32.onnx"
LOG_W8A8="benchmark_w8a8.log"

echo "========================================"
echo "Benchmarking Implicit W8A8 (Weights=INT8, Activations=INT8)..."
echo "========================================"
# --int8 enables INT8 precision
# Without explicit QDQ nodes or a calibration cache, TRT might complain or fallback.
# We'll see if it runs.
/usr/src/tensorrt/bin/trtexec \
    --onnx=$MODEL \
    --int8 \
    --avgRuns=50 \
    --duration=0 \
    --iterations=10 \
    --verbose \
    > $LOG_W8A8 2>&1

echo "Benchmark Complete. Log: $LOG_W8A8"
tail -n 20 $LOG_W8A8
