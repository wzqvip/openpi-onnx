
#!/bin/bash
MODEL_W8A16="./checkpoints/pi05_libero_pytorch/model.w8a16_qdq.onnx"

echo "Benchmarking W8A16 Model (QDQ Final)..."
/usr/src/tensorrt/bin/trtexec --onnx=$MODEL_W8A16 --int8 --fp16 --avgRuns=50 --duration=0 --iterations=10 > benchmark_w8a16_qdq_final.log 2>&1

tail -n 20 benchmark_w8a16_qdq_final.log
