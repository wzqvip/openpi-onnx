#!/bin/bash
# Comprehensive benchmark script for FP32, FP4, and INT8 models
# Runs 10 trials per task on all LIBERO suites, generates results and comparison

set -e

cd "$(dirname "$0")/.."

# Configuration
BENCHMARK_OUTPUT="./benchmark_results"
NUM_TRIALS=10
PORT=8012
TIMEOUT=300
MODELS=("fp32" "int8")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Ensure environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Activating Python environment...${NC}"
    source /home/taco/.venv/bin/activate
fi

# Create output directory
mkdir -p "$BENCHMARK_OUTPUT"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FP32/FP4/INT8 Model Benchmark Suite${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to start TensorRT server
start_server() {
    local model_type=$1
    local engine_path="./checkpoints/pi05_libero_onnx_compat/model.${model_type}.modelopt.engine"
    
    if [ ! -f "$engine_path" ]; then
        echo -e "${RED}Error: Engine not found at $engine_path${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Starting TensorRT server for ${model_type}...${NC}"
    echo "  Engine: $engine_path"
    echo "  Size: $(du -h "$engine_path" | cut -f1)"
    
    # Kill any existing servers
    pkill -f "serve_trt.py" || true
    sleep 2
    
    # Start new server
    python scripts/serve_trt.py \
        --engine_path="$engine_path" \
        --port=$PORT > /tmp/trt_server_${model_type}.log 2>&1 &
    
    local server_pid=$!
    echo $server_pid > /tmp/trt_server_${model_type}.pid
    
    # Wait for server to start
    echo -e "${YELLOW}Waiting for server to initialize...${NC}"
    sleep 5
    
    # Check if server is running
    if ! kill -0 $server_pid 2>/dev/null; then
        echo -e "${RED}Server failed to start!${NC}"
        cat /tmp/trt_server_${model_type}.log
        return 1
    fi
    
    echo -e "${GREEN}Server started (PID: $server_pid)${NC}"
}

# Function to stop TensorRT server
stop_server() {
    local model_type=$1
    
    if [ -f "/tmp/trt_server_${model_type}.pid" ]; then
        local pid=$(cat /tmp/trt_server_${model_type}.pid)
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            sleep 2
            echo -e "${GREEN}Server stopped${NC}"
        fi
    fi
}

# Function to run benchmark
run_benchmark() {
    local model_type=$1
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Benchmarking ${model_type^^}...${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Start server
    if ! start_server "$model_type"; then
        echo -e "${RED}Failed to start server for $model_type${NC}"
        return 1
    fi
    
    # Run benchmark
    echo -e "${YELLOW}Running benchmark (${NUM_TRIALS} trials per task)...${NC}"
    python scripts/benchmark_trt_models.py \
        --model_type="$model_type" \
        --num_trials=$NUM_TRIALS \
        --task_suite_name="all" \
        --port=$PORT \
        --benchmark_output="$BENCHMARK_OUTPUT" | tee "$BENCHMARK_OUTPUT/benchmark_${model_type}.log"
    
    # Stop server
    stop_server "$model_type"
    
    echo -e "${GREEN}${model_type^^} benchmark complete${NC}"
}

# Function to generate comparison report
generate_comparison() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Generating Comparison Report...${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    cat > "$BENCHMARK_OUTPUT/COMPARISON_REPORT.md" << 'EOF'
# Benchmark Comparison Report

## Summary
EOF
    
    # Add model results
    for model in "${MODELS[@]}"; do
        result_file="$BENCHMARK_OUTPUT/benchmark_${model}_${NUM_TRIALS}trials.json"
        if [ -f "$result_file" ]; then
            echo "### $model" >> "$BENCHMARK_OUTPUT/COMPARISON_REPORT.md"
            python3 << PYTHON
import json
with open('$result_file') as f:
    data = json.load(f)
print(f"- Success Rate: {data['overall_success_rate_percent']:.2f}%")
print(f"- Avg Latency: {data['overall_avg_latency_ms']:.2f}ms")
print(f"- Engine Size: {data['engine_size_gb']:.2f}GB")
PYTHON
            echo "" >> "$BENCHMARK_OUTPUT/COMPARISON_REPORT.md"
        fi
    done
    
    echo -e "${GREEN}Comparison report generated: $BENCHMARK_OUTPUT/COMPARISON_REPORT.md${NC}"
}

# Main execution
main() {
    echo "Models to test: ${MODELS[@]}"
    echo "Trials per task: $NUM_TRIALS"
    echo "Output directory: $BENCHMARK_OUTPUT"
    echo ""
    
    # Run benchmarks for each model
    for model in "${MODELS[@]}"; do
        if run_benchmark "$model"; then
            echo -e "${GREEN}✓ $model benchmark successful${NC}"
        else
            echo -e "${RED}✗ $model benchmark failed${NC}"
        fi
        echo ""
    done
    
    # Generate comparison
    generate_comparison
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}All benchmarks complete!${NC}"
    echo -e "${GREEN}Results saved to: $BENCHMARK_OUTPUT${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # List results
    echo ""
    echo "Generated files:"
    ls -lh "$BENCHMARK_OUTPUT"/ 2>/dev/null | tail -n +2 || true
}

# Run main
main
