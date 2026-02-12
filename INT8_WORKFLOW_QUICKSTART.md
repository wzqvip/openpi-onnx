# INT8 Quantization Workflow - Quick Start

## Overview

This document provides a step-by-step quick reference for the complete INT8 quantization workflow from PyTorch to TensorRT.

**Result Summary**: INT8 achieves **98.25% accuracy** with **~1.6x faster** inference and **~1.6x less** GPU memory vs FP32.

## Prerequisites

```bash
# Ensure environment is set up
pip install torch torchvision
pip install onnx onnx-graphsurgeon onnxruntime-gpu
pip install nvidia-modelopt[torch]
pip install tensorrt

# Clone and setup LIBERO
git submodule update --init --recursive
```

## Complete Workflow

### Step 1: Collect Calibration Data

Collect real inference data from LIBERO environment:

```bash
python scripts/collect_calibration_data.py --output calibration_data.pt
```

**Output**: `calibration_data.pt` (~100MB, contains input tensors)

---

### Step 2: Export to INT8 ONNX

Use ModelOpt to quantize and export:

```bash
python exports/export_modelopt_int8.py
```

**What happens**:
- Loads PyTorch model from `checkpoints/pi05_libero_pytorch/`
- Loads `calibration_data.pt`
- Inserts INT8 quantizers and calibrates
- Exports to ONNX (Opset 19) with CumSum patch
- Applies GraphSurgeon cleanup

**Output**: `checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.cleaned.onnx`

---

### Step 3: Build TensorRT Engine

Compile ONNX to TensorRT engine:

```bash
trtexec --onnx=checkpoints/pi05_libero_onnx_compat/model.int8.modelopt.cleaned.onnx \
        --saveEngine=checkpoints/pi05_libero_onnx_compat/engine_int8.trt \
        --int8
```

**Expected output**: Log shows `Precision: FP32+INT8`

**Output**: `checkpoints/pi05_libero_onnx_compat/engine_int8.trt` (~4.6GB)

---

### Step 4: Evaluate

#### Option A: Full Benchmark (All 4 Suites)

```bash
# Run all 4 suites (libero_spatial, libero_goal, libero_object, libero_10)
python scripts/eval_libero_trt_v1.py \
    --engine-path checkpoints/pi05_libero_onnx_compat/engine_int8.trt \
    --suite libero_spatial \
    --num-trials 20

# Repeat for other suites
```

#### Option B: Quick Test (LIBERO-10 Only)

Start inference server:

```bash
python scripts/serve_trt.py \
    --engine-path checkpoints/pi05_libero_onnx_compat/engine_int8.trt \
    --port 8000
```

In another terminal, run evaluation:

```bash
python scripts/eval_libero_10.py \
    --host localhost \
    --port 8000 \
    --num-trials 20 \
    --output results.json
```

---

## Expected Results

### Accuracy (20 trials/task, 800 episodes)

| Suite | FP32 PyTorch | INT8 TensorRT |
|-------|--------------|---------------|
| libero_spatial | 99.5% (199/200) | 99.0% (198/200) |
| libero_goal | 91.0% (182/200) | 98.5% (197/200) |
| libero_object | 95.0% (190/200) | 99.5% (199/200) |
| libero_10 | 89.5% (179/200) | 96.0% (192/200) |
| **Overall** | **93.75% (750/800)** | **98.25% (786/800)** |

### Performance

| Metric | FP32 PyTorch | INT8 TensorRT | Improvement |
|--------|--------------|---------------|-------------|
| **Accuracy** | 93.75% | **98.25%** | **+4.5%** |
| **Inference Latency** | 262.41ms mean | **~162ms mean** | **~1.6x faster** |
| **GPU Memory** | 8.10 GB | **4.95 GB** | **~1.6x smaller** |
| **Model Size** | ~12.2 GB | **~4.6 GB** | **~2.7x smaller** |

---

## File Locations

```
openpi-onnx/
├── calibration_data.pt                          # Step 1 output
├── checkpoints/
│   ├── pi05_libero_pytorch/                     # Source PyTorch model
│   └── pi05_libero_onnx_compat/
│       ├── model.int8.modelopt.cleaned.onnx     # Step 2 output
│       └── engine_int8.trt                      # Step 3 output
├── scripts/
│   ├── collect_calibration_data.py              # Step 1 script
│   ├── eval_libero_trt_v1.py                    # Step 4A script
│   ├── eval_libero_10.py                        # Step 4B script
│   └── serve_trt.py                             # Inference server
└── exports/
    └── export_modelopt_int8.py                  # Step 2 script
```

---

## Troubleshooting

### "Quantizer not calibrated" error
- Ensure `calibration_data.pt` exists
- Check file size (should be ~100MB)

### "CumSum validation failed" in TensorRT
- Use the `.cleaned.onnx` file (has CumSum patch)
- Verify export script completed successfully

### Low accuracy (<90%)
- Check normalization stats in model
- Verify calibration data was collected correctly
- Use `eval_libero_trt_v1.py` (not the WebSocket version for benchmarking)

---

## References

- Complete guide: [docs/conversion/pi05_onnx_conversion_guide.md](docs/conversion/pi05_onnx_conversion_guide.md)
- Full results: [INT8_FINAL_RESULTS.md](INT8_FINAL_RESULTS.md)
- Summary: [INT8_SUMMARY.md](INT8_SUMMARY.md)
