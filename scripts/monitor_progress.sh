#!/bin/bash
# Monitor evaluation progress and prepare next engines

echo "==================================="
echo "FP4 Deployment Progress Monitor"
echo "==================================="
echo ""

# Check INT4 evaluation status
if ps aux | grep -E "eval_libero_trt.*8017" | grep -v grep > /dev/null; then
    echo "✓ INT4 evaluation is running"
    INT4_PID=$(ps aux | grep -E "eval_libero_trt.*8017" | grep -v grep | awk '{print $2}')
    echo "  PID: $INT4_PID"
    
    # Check log file
    INT4_LOG="/home/taco/openpi-onnx/results/benchmark_results/libero_spatial_int4_run.log"
    if [ -f "$INT4_LOG" ]; then
        echo "  Log size: $(du -h $INT4_LOG | cut -f1)"
        echo ""
        echo "  Last 5 lines:"
        tail -n 5 "$INT4_LOG" | sed 's/^/    /'
    fi
else
    echo "✗ INT4 evaluation is not running"
    
    # Check if it completed
    INT4_LOG="/home/taco/openpi-onnx/results/benchmark_results/libero_spatial_int4_run.log"
    if [ -f "$INT4_LOG" ] && grep -q "Final Results" "$INT4_LOG"; then
        echo ""
        echo "✓ INT4 evaluation COMPLETED!"
        echo ""
        grep -A 5 "Final Results" "$INT4_LOG" | sed 's/^/    /'
    fi
fi

echo ""
echo "-----------------------------------"
echo ""

# Check server status
echo "Server Status:"
if ps aux | grep "serve_trt.py.*8017" | grep -v grep > /dev/null; then
    echo "  ✓ INT4 TensorRT server (port 8017) is running"
else
    echo "  ✗ INT4 TensorRT server is not running"
fi

echo ""

# Check available engines
echo "Available TensorRT Engines:"
find /home/taco/checkpoints -name "*.trt" -o -name "*.engine" 2>/dev/null | while read engine; do
    size=$(du -h "$engine" | cut -f1)
    name=$(basename "$engine")
    echo "  - $name ($size)"
done

echo ""
echo "==================================="
