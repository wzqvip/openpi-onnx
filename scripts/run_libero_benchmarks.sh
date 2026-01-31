#!/bin/bash
set -e

# Benchmark Configuration
TASKS=("libero_spatial" "libero_object" "libero_goal" "libero_100")
MODELS=("torch" "int8" "fp4")
RESULTS_DIR="results/benchmark_results"
mkdir -p $RESULTS_DIR

echo "========================================================"
echo "Starting Comprehensive Libero Benchmark"
echo "Tasks: ${TASKS[@]}"
echo "Models: ${MODELS[@]}"
echo "========================================================"

for task in "${TASKS[@]}"; do
    echo ""
    echo "--------------------------------------------------------"
    echo "Benchmarking Task Suite: $task"
    echo "--------------------------------------------------------"
    
    for model in "${MODELS[@]}"; do
        echo "Running Model: $model"
        
        if [ "$model" == "torch" ]; then
            # PyTorch Baseline
            /home/taco/.venv/bin/python scripts/eval_libero_torch.py \
                --task_suite_name $task \
                --num_trials_per_task 3
                
        elif [ "$model" == "int8" ]; then
            # INT8 TensorRT
            # Ensure engine exists
            ENGINE_PATH="checkpoints/pi05_libero_onnx_compat/engine_int8.trt"
            if [ ! -f "$ENGINE_PATH" ]; then
                echo "Skipping INT8: Engine not found at $ENGINE_PATH"
                continue
            fi
            
            echo "Starting Inference Server..."
            # Start TRT server in background
            /home/taco/.venv/bin/python scripts/serve_trt.py \
                --engine_path $ENGINE_PATH \
                --port 8015 &
            SERVER_PID=$!
            
            # Wait for server to start (heuristic)
            echo "Waiting for server (PID $SERVER_PID) to initialize..."
            sleep 15
            
            echo "Running Benchmark Client..."
            /home/taco/.venv/bin/python scripts/eval_libero_trt.py \
                --task_suite_name $task \
                --num_trials_per_task 3 \
                --port 8015 || true # Continue even if client fails
                
            # Kill server
            echo "Stopping Inference Server..."
            kill $SERVER_PID

                
        elif [ "$model" == "fp4" ]; then
             # FP4 Simulated
             /home/taco/.venv/bin/python scripts/eval_fp4_torch.py \
                --task_suite_name $task \
                --num_trials_per_task 3
        fi
        
        echo "Finished $model on $task"
    done
done

echo ""
echo "========================================================"
echo "Benchmark Complete."
echo "========================================================"
