<!-- Last Updated: 2026-01-29 -->
# Libero Benchmark on Jetson Thor (TensorRT INT8)

This guide explains how to run the **Libero Spatial** benchmark on NVIDIA Jetson Thor using the custom TensorRT Python inference pipeline with the **verified INT8 engine**.

## Architecture Overview

**Split-Process Architecture**:
1.  **Inference Server (Python 3.12)**:
    -   Hosts the TensorRT engine (`model.int8.modelopt.engine`).
    -   Script: `scripts/serve_trt.py`
2.  **Benchmark Client (Python 3.11)**:
    -   Runs the Libero simulation.
    -   Script: `scripts/eval_libero_trt.py`

---

## 1. Prerequisites
Ensure you have converted the model to INT8 using the [Conversion Guide](pi05_onnx_conversion_guide.md).

## 2. Running the Benchmark

### Terminal 1: Start the Inference Server
```bash
cd /home/taco/openpi-onnx

# Run server (ensure LD_LIBRARY_PATH includes libcudart if needed)
export LD_LIBRARY_PATH=/home/taco/.venv/lib/python3.12/site-packages/nvpl/lib:$LD_LIBRARY_PATH
python scripts/serve_trt.py \
    --engine_path checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.engine \
    --port 8012
```

**Verify**: Wait for "Server started on port 8012".

### Terminal 2: Run the Benchmark Client
```bash
cd /home/taco/openpi-onnx

python scripts/eval_libero_trt.py \
    --task_suite_name libero_spatial \
    --num_trials_per_task 10 \
    --port 8012
```

## 3. Verified Results

We have confirmed **100% Success Rate** (10/10 tasks) on the `libero_spatial` suite for the following configuration:
-   **Model**: Pi0.5
-   **Precision**: INT8 (ModelOpt Calibrated)
-   **Engine**: `model.int8.modelopt.engine`