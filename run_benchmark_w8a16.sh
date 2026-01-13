
#!/bin/bash
MODEL_W8A16="./checkpoints/pi05_libero_pytorch/model.w8a16.onnx"

echo "Benchmarking W8A16 model..."
# We use --fp16 to ensure activations are FP16. Weights are INT8 in the model (hopefully).
# If correct, TensorRT typically handles Mixed Precision.
/usr/src/tensorrt/bin/trtexec --onnx=$MODEL_W8A16 --fp16 --avgRuns=50 --duration=0 --iterations=10 > benchmark_w8a16.log 2>&1

tail -n 20 benchmark_w8a16.log
