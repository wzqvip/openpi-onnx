
#!/bin/bash
MODEL_W4A16="./checkpoints/pi05_libero_pytorch/model.w4a16.onnx"

echo "Benchmarking W4A16 Model..."
# Try to enable INT8 kernels if TRT maps INT4->INT8 or Decompress->FP16
/usr/src/tensorrt/bin/trtexec --onnx=$MODEL_W4A16 --fp16 --avgRuns=50 --duration=0 --iterations=10 > benchmark_w4a16.log 2>&1

tail -n 20 benchmark_w4a16.log
