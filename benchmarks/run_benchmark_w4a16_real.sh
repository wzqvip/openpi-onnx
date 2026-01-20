
#!/bin/bash
MODEL="checkpoints/pi05_libero_pytorch/model.w4a16.onnx"
LOG_W4="benchmark_w4a16_real.log"

echo "========================================"
echo "Benchmarking Real W4A16 Model..."
echo "========================================"
# Run with --fp16 to allow activation fallback, but weights are INT4
# trtexec should pick up MatMulNBits nodes if supported
/usr/src/tensorrt/bin/trtexec \
    --onnx=$MODEL \
    --fp16 \
    --avgRuns=50 \
    --duration=0 \
    --iterations=10 \
    --verbose \
    > $LOG_W4 2>&1

echo "Benchmark Complete. Log: $LOG_W4"
tail -n 15 $LOG_W4
