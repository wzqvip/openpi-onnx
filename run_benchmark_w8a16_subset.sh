
#!/bin/bash
MODEL_SUBSET="./checkpoints/pi05_libero_pytorch/w8a16_subset/model.onnx"

echo "Benchmarking W8A16 Subset model..."
/usr/src/tensorrt/bin/trtexec --onnx=$MODEL_SUBSET --fp16 --avgRuns=50 --duration=0 --iterations=10 > benchmark_w8a16_subset.log 2>&1

tail -n 20 benchmark_w8a16_subset.log
