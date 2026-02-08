#!/bin/bash
# Automated workflow for FP4 model evaluation

set -e

WORKSPACE="/home/taco/openpi-onnx"
VENV="/home/taco/.venv"
CHECKPOINTS="/home/taco/checkpoints/pi05_libero_pytorch"

echo "========================================"
echo "FP4 Model Evaluation Automation"
echo "========================================"
echo ""

# Wait for INT4 evaluation to complete
echo "[1/5] Waiting for INT4 evaluation to complete..."
while ps aux | grep -E "eval_libero_trt.*8017" | grep -v grep > /dev/null; do
    sleep 30
    echo "  Still running... ($(date +%H:%M:%S))"
done

echo "  ✓ INT4 evaluation completed!"
echo ""

# Display INT4 results
echo "[2/5] INT4 Results:"
if [ -f "$WORKSPACE/results/benchmark_results/libero_spatial_int4_run.log" ]; then
    grep -A 5 "Final Results" "$WORKSPACE/results/benchmark_results/libero_spatial_int4_run.log" | sed 's/^/  /'
fi
echo ""

# Wait for FP8 engine build
echo "[3/5] Waiting for FP8 engine build to complete..."
while ps aux | grep "trtexec.*engine_fp8" | grep -v grep > /dev/null; do
    sleep 30
    if [ -f "$CHECKPOINTS/engine_fp8.trt" ]; then
        SIZE=$(du -h "$CHECKPOINTS/engine_fp8.trt" | cut -f1)
        echo "  Building... Current size: $SIZE ($(date +%H:%M:%S))"
    fi
done

echo "  ✓ FP8 engine build completed!"
if [ -f "$CHECKPOINTS/engine_fp8.trt" ]; then
    SIZE=$(du -h "$CHECKPOINTS/engine_fp8.trt" | cut -f1)
    echo "  Engine size: $SIZE"
fi
echo ""

# Kill INT4 server and start FP8 server
echo "[4/5] Starting FP8 TensorRT server..."
if ps aux | grep "serve_trt.py.*8017" | grep -v grep > /dev/null; then
    echo "  Stopping INT4 server..."
    pkill -f "serve_trt.py.*8017" || true
    sleep 3
fi

cd "$WORKSPACE"
nohup "$VENV/bin/python" scripts/serve_trt.py \
    --engine_path "$CHECKPOINTS/engine_fp8.trt" \
    --port 8018 \
    > logs/serve_fp8.log 2>&1 &
    
echo "  Waiting for server to start..."
sleep 15
echo "  ✓ FP8 server started on port 8018"
echo ""

# Run FP8 evaluation
echo "[5/5] Running FP8 evaluation..."
cd "$WORKSPACE"
"$VENV/bin/python" scripts/eval_libero_trt.py \
    --task_suite_name libero_spatial \
    --num_trials_per_task 3 \
    --ws_url ws://localhost:8018 \
    > results/benchmark_results/libero_spatial_fp8_run.log 2>&1

echo "  ✓ FP8 evaluation completed!"
echo ""

# Display FP8 results
echo "FP8 Results:"
grep -A 5 "Final Results" "$WORKSPACE/results/benchmark_results/libero_spatial_fp8_run.log" | sed 's/^/  /'
echo ""

# Compare all results
echo "========================================"
echo "Summary of All Results"
echo "========================================"
echo ""

echo "PyTorch FP32:"
echo "  Latency: 483ms | Accuracy: 100% | Memory: 8.10GB"
echo ""

echo "TensorRT INT4:"
if [ -f "$WORKSPACE/results/benchmark_results/libero_spatial_int4_run.log" ]; then
    grep -E "Success Rate:|Latency" "$WORKSPACE/results/benchmark_results/libero_spatial_int4_run.log" | sed 's/^/  /'
fi
echo ""

echo "TensorRT FP8:"
if [ -f "$WORKSPACE/results/benchmark_results/libero_spatial_fp8_run.log" ]; then
    grep -E "Success Rate:|Latency" "$WORKSPACE/results/benchmark_results/libero_spatial_fp8_run.log" | sed 's/^/  /'
fi
echo ""

echo "========================================"
echo "Automation Complete!"
echo "========================================"
