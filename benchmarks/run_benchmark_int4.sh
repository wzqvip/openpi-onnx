
#!/bin/bash
MODEL="dist/final_fp16/model.fp32.onnx"
LOG_INT4="benchmark_int4.log"

echo "========================================"
echo "Benchmarking INT4..."
echo "========================================"
# Note: --int4 usually requires --fp16 or --fp32 as fallback
/usr/src/tensorrt/bin/trtexec --onnx=$MODEL --int4 --fp16 --avgRuns=50 --duration=0 --iterations=10 > $LOG_INT4 2>&1
echo "INT4 Benchmark Complete. Log: $LOG_INT4"
tail -n 15 $LOG_INT4
