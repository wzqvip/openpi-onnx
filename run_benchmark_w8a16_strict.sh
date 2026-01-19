
#!/bin/bash
MODEL_W8A16="./checkpoints/pi05_libero_pytorch/model.w8a16_qdq.onnx"

echo "Benchmarking W8A16 Model (QDQ) with Strict Precision..."
# --precisionConstraints=obey forces TRT to use the precision specified in the network (for layers with cast/quantize).
# --int8 --fp16 enables the kernels.
/usr/src/tensorrt/bin/trtexec --onnx=$MODEL_W8A16 --int8 --fp16 --precisionConstraints=obey --avgRuns=50 --duration=0 --iterations=10 > benchmark_w8a16_strict.log 2>&1

tail -n 20 benchmark_w8a16_strict.log
