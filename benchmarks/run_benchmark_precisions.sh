
#!/bin/bash
MODEL="dist/final_fp16/model.fp32.onnx"
LOG_BF16="benchmark_bf16.log"
LOG_FP8="benchmark_fp8.log"

echo "========================================"
echo "Benchmarking BF16..."
echo "========================================"
/usr/src/tensorrt/bin/trtexec --onnx=$MODEL --bf16 --avgRuns=50 --duration=0 --iterations=10 > $LOG_BF16 2>&1
echo "BF16 Benchmark Complete. Log: $LOG_BF16"
tail -n 15 $LOG_BF16

echo ""
echo "========================================"
echo "Benchmarking FP8..."
echo "========================================"
/usr/src/tensorrt/bin/trtexec --onnx=$MODEL --fp8 --avgRuns=50 --duration=0 --iterations=10 > $LOG_FP8 2>&1
echo "FP8 Benchmark Complete. Log: $LOG_FP8"
tail -n 15 $LOG_FP8
