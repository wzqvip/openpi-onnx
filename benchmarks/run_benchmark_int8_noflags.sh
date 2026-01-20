
#!/bin/bash
MODEL_INT8="./checkpoints/pi05_libero_pytorch/int8_final/model.onnx"

echo "Benchmarking INT8 Dynamic Model (No Flags)..."
/usr/src/tensorrt/bin/trtexec --onnx=$MODEL_INT8 --avgRuns=50 --duration=0 --iterations=10 > benchmark_int8_noflags.log 2>&1

tail -n 20 benchmark_int8_noflags.log
