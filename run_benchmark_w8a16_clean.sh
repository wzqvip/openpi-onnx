
#!/bin/bash
MODEL_CLEAN="./checkpoints/pi05_libero_pytorch/w8a16_clean/model.onnx"

echo "Benchmarking W8A16 Clean model..."
# Use --fp16 to simulate W8A16 (INT8 weights from file, FP16 compute via flag)
# Also --int8 might be needed if DynamicQuantizeLinear nodes require Int8 IO?
# But typically --fp16 covers mixed precision if nodes are explicit.
# Let's try just --fp16 first.
/usr/src/tensorrt/bin/trtexec --onnx=$MODEL_CLEAN --fp16 --avgRuns=50 --duration=0 --iterations=10 > benchmark_w8a16_clean.log 2>&1

tail -n 20 benchmark_w8a16_clean.log
