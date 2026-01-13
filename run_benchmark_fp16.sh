
#!/bin/bash
MODEL_FP32="./checkpoints/pi05_libero_pytorch/model.fp32.onnx"
MODEL_FP16="./checkpoints/pi05_libero_pytorch/model.fp16.onnx"

if [ -f "$MODEL_FP16" ]; then
    echo "Benchmarking FP16 model..."
    /usr/src/tensorrt/bin/trtexec --onnx=$MODEL_FP16 --fp16 --avgRuns=50 --duration=0 --iterations=10 > benchmark_fp16.log 2>&1
elif [ -f "$MODEL_FP32" ]; then
    echo "Benchmarking FP32 model (simulating FP16)..."
    /usr/src/tensorrt/bin/trtexec --onnx=$MODEL_FP32 --fp16 --avgRuns=50 --duration=0 --iterations=10 > benchmark_fp16.log 2>&1
else
    echo "No model found."
fi

tail -n 20 benchmark_fp16.log
