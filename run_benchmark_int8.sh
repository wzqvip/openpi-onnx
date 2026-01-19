
#!/bin/bash
MODEL_INT8="./checkpoints/pi05_libero_pytorch/int8_final/model.onnx"

echo "Benchmarking INT8 Dynamic Model..."
# --int8 --fp16 to allow all kernels.
# Dynamic quantization often uses INT8 compute with FP32 scales.
/usr/src/tensorrt/bin/trtexec --onnx=$MODEL_INT8 --int8 --fp16 --avgRuns=50 --duration=0 --iterations=10 > benchmark_int8_dynamic.log 2>&1

tail -n 20 benchmark_int8_dynamic.log
