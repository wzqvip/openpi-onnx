<!-- Last Updated: 2026-01-29 -->
# Updating ONNX Export to ModelOpt Style

## Problem
The previous ONNX export method resulted in accuracy degradation (success rate drop) due to:
1.  **Opset Version**: Older opsets didn't support necessary operators for modern transformer architectures correctly.
2.  **Normalization Stats**: Discrepancies in how normalization statistics were applied or exported.
3.  **Graph Optimization**: Aggressive constant folding by `onnx-graphsurgeon` corrupted the `CumSum` operations used in rotary positional embeddings.

## Solution: ModelOpt-Style Export
We adopted a robust "ModelOpt-style" workflow:

1.  **Opset 19**: Exporting with `opset_version=19` ensures compatibility with modern operators.
2.  **Monkey Patching**:
    - Patched `onnx.helper.float32_to_bfloat16` to fix compatibility with `onnx-graphsurgeon`.
    - Mocked `jax`, `jaxtyping`, and `typeguard` to bypass runtime checks and dependencies not needed for inference.
3.  **Manual Graph Patching**:
    - Instead of relying on automatic constant folding (which broke `CumSum`), we manually injected a patch: `Cast(Bool -> Int32)` -> `CumSum` -> `Cast(Int32 -> Int64)`. This works around TensorRT's limitation with boolean `CumSum` inputs.
4.  **INT8 Quantization with Calibration**:
    - Used `modelopt.torch.quantization` to insert quantizers.
    - Collected **real calibration data** from the inference server (saving inputs from actual Libero evaluation runs) instead of using random noise.
    - Calibrated the model on a subset of real data to ensure accurate quantization scales.

## Results
- **FP32 Accuracy**: 100% Success Rate (matched PyTorch baseline).
- **INT8 Accuracy**: 100% Success Rate (no degradation).
- **Model Size**: Reduced from ~12GB (FP32) to ~4.6GB (INT8).
- **Latency**: Reduced inference latency (exact ms improvement depends on hardware, typically 2-3x speedup on Thor).

## How to Reproduce
1.  **Export INT8 Model**:
    ```bash
    python exports/export_modelopt_int8.py
    ```
    This script loads `calibration_data.pt`, applies quantization, calibrates, and exports `model.int8.modelopt.cleaned.onnx`.

2.  **Compile to TensorRT**:
    ```bash
    trtexec --onnx=model.int8.modelopt.cleaned.onnx --saveEngine=model.int8.modelopt.engine --int8
    ```

3.  **Run Inference**:
    ```bash
    python scripts/serve_trt.py --engine_path model.int8.modelopt.engine --port 8012
    ```
