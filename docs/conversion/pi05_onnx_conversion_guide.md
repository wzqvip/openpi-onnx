<!-- Last Updated: 2026-02-12 -->
# Pi0.5 ONNX Export & INT8 Quantization Guide

This guide details the complete workflow for converting the Pi0.5 PyTorch model to a high-accuracy, quantized TensorRT INT8 engine.

## 1. Overview
We utilize **NVIDIA ModelOpt** for INT8 calibration and export, specifically addressing:
-   **Calibration Data Collection**: Capture real inference data from LIBERO environment
-   **Opset 19**: Required for modern operators
-   **Graph Patching**: Manual injection of `Cast(Int32)` -> `CumSum` to fix TensorRT compatibility
-   **Calibration**: Using real captured inference data for high accuracy

## 2. Environment Setup

**Dependencies**:
```bash
# Core
pip install torch torchvision
pip install onnx onnx-graphsurgeon onnxruntime-gpu
pip install nvidia-modelopt[torch]

# TensorRT (via system or pip)
pip install tensorrt
```

## 3. Complete INT8 Quantization Workflow

### Step 1: Collect Calibration Data

Collect real inference data from LIBERO environment for calibration:

**Command:**
```bash
python scripts/collect_calibration_data.py --output calibration_data.pt
```

**Output:**
-   `calibration_data.pt` - Contains input tensors from real LIBERO tasks

### Step 2: Export & Calibrate to INT8 ONNX

We use the custom script `exports/export_modelopt_int8.py`. This script:
1.  Loads the PyTorch model
2.  Loads real calibration data from `calibration_data.pt`
3.  Inserts INT8 quantizers and calibrates them
4.  Exports the model with **Opset 19**
5.  Applies GraphSurgeon cleanup and the `CumSum` patch

**Command:**
```bash
python exports/export_modelopt_int8.py
```

**Output:**
-   `checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.cleaned.onnx`

### Step 3: Compile TensorRT Engine

Use `trtexec` to build the engine. The `--int8` flag is crucial, even though the ONNX model is already quantized (it enables INT8 kernels).

**Command:**
```bash
trtexec --onnx=checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.cleaned.onnx \
        --saveEngine=checkpoints/pi05_libero_onnx_compat/engine_int8.trt \
        --int8
```

**Verify Output:**
Ensure the log shows `Precision: FP32+INT8`.

### Step 4: Evaluate INT8 Engine

#### Option A: Full LIBERO Benchmark (All 4 Suites)

Use the v1 evaluation script for comprehensive benchmarking:

```bash
python scripts/eval_libero_trt_v1.py \
    --engine-path checkpoints/pi05_libero_onnx_compat/engine_int8.trt \
    --suite libero_spatial \
    --num-trials 20
```

#### Option B: LIBERO-10 Suite Only (WebSocket-based)

For testing with the LIBERO-10 suite via WebSocket inference:

**Start inference server:**
```bash
python scripts/serve_trt.py \
    --engine-path checkpoints/pi05_libero_onnx_compat/engine_int8.trt \
    --port 8000
```

**Run evaluation:**
```bash
python scripts/eval_libero_10.py \
    --host localhost \
    --port 8000 \
    --num-trials 20 \
    --output libero10_results.json
```

The `eval_libero_10.py` script outputs:
- Per-task accuracy and latency statistics
- Overall accuracy across all 10 tasks
- Inference latency (mean, median, P99)
- Optional JSON output with detailed results

## 4. Performance

**Latest Results (20 trials per task, all 4 LIBERO suites, 800 episodes total)**:

| Metric | FP32 PyTorch | INT8 TensorRT | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 93.75% (750/800) | **98.25% (786/800)** | **+4.5%** |
| **Inference Latency** | 262.41ms mean (P99 278.79ms) | **~162ms mean (P99 ~167ms)** | **~1.6x Faster** |
| **GPU Memory** | 8.10 GB | **4.95 GB** | **~1.6x Smaller** |
| **Model Size** | ~12.2 GB | **~4.6 GB** | **~2.7x Smaller** |

**Per-Suite Accuracy:**

| Suite | FP32 | INT8 |
| :--- | :--- | :--- |
| libero_spatial | 99.5% (199/200) | 99.0% (198/200) |
| libero_goal | 91.0% (182/200) | 98.5% (197/200) |
| libero_object | 95.0% (190/200) | 99.5% (199/200) |
| libero_10 | 89.5% (179/200) | 96.0% (192/200) |

See [INT8_FINAL_RESULTS.md](../../INT8_FINAL_RESULTS.md) for complete details.

**"Quantizer not calibrated" error during export**:
-   Ensure `calibration_data.pt` exists and contains valid tensors.
-   The script `export_modelopt_int8.py` is configured to use a subset (e.g., 1 sample) if full calibration is too slow, which is sufficient for this architecture.

**TensorRT: "CumSum validation failed"**:
-   This indicates the ONNX model was not patched. Ensure you are using the `.cleaned.onnx` file produced by the export script, which contains the `Cast(Int32)` fix.
