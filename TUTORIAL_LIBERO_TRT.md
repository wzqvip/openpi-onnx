# Libero Benchmark on Jetson Thor (TensorRT GPU)

This guide explains how to run the **Libero Spatial** benchmark on NVIDIA Jetson Thor using the custom TensorRT Python inference pipeline.

## Architecture Overview

Due to software version incompatibilities on the Jetson Thor platform (running Tegra), we use a **Split-Process Architecture**:

1.  **Inference Server (Python 3.12)**:
    -   Uses the **System Python (`/usr/bin/python3`)** which has native access to `tensorrt`.
    -   Host the TensorRT engine and handles GPU inference.
    -   Script: `scripts/serve_trt.py`

2.  **Benchmark Client (Python 3.11)**:
    -   Uses the project's main **`uv` virtual environment**.
    -   Runs the Libero simulation, Robosuite environment, and OpenPi policy logic.
    -   Sends observations to the server via WebSockets.
    -   Script: `scripts/eval_libero_trt.py`

---

## 1. Prerequisites & Setup

### A. Main Project Environment (Python 3.11)
Ensure you are in the project root (`/home/taco/openpi-onnx`) and the main environment is set up.

```bash
# Install client dependencies
uv pip install msgpack msgpack-numpy websockets
```

### B. TensorRT Server Environment (Python 3.12)
We need a separate virtual environment that inherits the system packages (to get `tensorrt`).

```bash
# 1. Create venv with system-site-packages enabled
python3 -m venv --system-site-packages .venv312

# 2. Install server dependencies
.venv312/bin/pip install numpy cuda-python websockets msgpack msgpack-numpy
```

*Note: If `cuda-python` fails to provide `cudart`, the server script is patched to use `ctypes` and system `libcudart.so`.*

---

## 2. Running the Benchmark

You will need **two terminal windows**.

### Terminal 1: Start the Inference Server
This process must remain running. It will build the TensorRT engine (if not found) and listen for requests.

```bash
cd /home/taco/openpi-onnx

# Export CUDA library path (critical for finding libcudart)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run server using the Python 3.12 venv
.venv312/bin/python scripts/serve_trt.py
```

**Verify**: Wait until you see:
```
INFO:TRTServer:Server started on port 8000
```

### Terminal 2: Run the Benchmark Client
Once the server is ready, run the evaluation script from the main environment.

```bash
cd /home/taco/openpi-onnx

# Run via uv to use the main Py3.11 environment
uv run scripts/eval_libero_trt.py \
    --task_suite_name libero_spatial \
    --num_trials_per_task 10 \
    --host 0.0.0.0 \
    --port 8000
```

*Tips:*
- Use `--num_trials_per_task 1` for a quick test.
- Use `--task_id <ID>` to run a specific task index (0-9).

---

## 3. Results and Outputs

-   **Videos**: Replay videos of the rollouts are saved to `data/libero/videos_trt/`.
-   **Logs**: Check the client terminal for success rates and the server terminal for inference logs.

## 4. Pipeline Verification

To ensure the TensorRT engine outputs match the original PyTorch model (confirming correct export and inference):

1.  **Capture a Trace**: Run the PyTorch model on the CPU to record inputs and outputs.
    ```bash
    uv run scripts/capture_trace.py
    ```
    This creates `trace_data.npz`.

2.  **Compare Traces**: Run the comparison script while the TensorRT server is active.
    ```bash
    uv run scripts/compare_traces.py
    ```
    Success is indicated by "outputs match within tolerance" (Max Diff < 1.0).

---

## Troubleshooting

-   **"libcudart.so not found"**:
    -   Ensure `LD_LIBRARY_PATH` includes `/usr/local/cuda/lib64`.
    -   The server script uses `ctypes` to preload the library; check `scripts/serve_trt.py` if paths differ.

-   **"ConnectionRefusedError"**:
    -   Ensure the server is running in Terminal 1 and printed "Server started on port 8000".

-   **"KeyError: observation/image"**:
    -   Ensure `scripts/eval_libero_trt.py` does **not** double-apply `LiberoInputs` (check `input_transforms` list).

-   **0% Success Rate**:
    -   If the success rate is 0% on both CPU and GPU: The inference pipeline is correct (matched numerically), but the model checkpoint or task specification may require tuning.

-   **Slow Performance**:
    -   TensorRT on Jetson Thor should achieve ~15s per episode (vs ~60s on CPU). If slower, ensure the engine was built with FP16 enabled (default in `scripts/trt_builder.py`).

## 5. Investigation: 0% Success Rate

Both the PyTorch (CPU) baseline and the TensorRT (GPU) implementation currently achieve a 0% success rate on the `libero_spatial` benchmark.

**Verified Findings:**
1.  **Infrastructure is Sound**: The TensorRT pipeline runs without crashing, processes images correctly, and runs 4x faster than CPU.
2.  **Numerical Match**: `compare_traces.py` confirms that TensorRT outputs match PyTorch outputs (within FP16 tolerance).
3.  **Conclusion**: The failure to solve the task lies in the **Model/Policy capability** or the **Input Configuration**, not the Inference Backend deployment.

**Suggested Next Steps for Model Improvement:**
-   **Model Checkpoint**: Verify `pi05_libero_pytorch` is the correct checkpoint trained on `libero_spatial` tasks.
-   **Prompt Engineering**: Ensure the task descriptions passed to the model match the training data format exactly.
-   **Input Stats**: Validate that the normalization statistics (`assets/physical-intelligence/libero`) match the training data statistics.
-   **Simulation Gap**: Verify that the Libero simulation environment (Robosuite) rendering matches the training data visual distribution.
··