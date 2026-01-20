#!/bin/bash
set -e

echo "Starting Remaining Exports..."
source .venv/bin/activate

echo "3. Exporting INT8 ONNX..."
python export_int8_onnx.py || echo "INT8 Export Failed"

echo "4. Exporting NVFP4 ONNX..."
python export_nvfp4_onnx.py || echo "NVFP4 Export Failed"

echo "5. Exporting INT4 ONNX..."
python export_int4_onnx.py || echo "INT4 Export Failed"

echo "6. Exporting FP8 ONNX..."
python export_fp8_onnx.py || echo "FP8 Export Failed"

echo "All remaining exports attempted."
