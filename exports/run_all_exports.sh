#!/bin/bash
set -e

echo "Starting Exports..."
source .venv/bin/activate

# Ensure checkpoint exists
if [ ! -f checkpoints/pi05_libero_pytorch/model.safetensors ]; then
    echo "Checkpoint not found! Running creation script..."
    python create_full_dummy_checkpoint.py
fi

echo "1. Exporting FP32 ONNX..."
python export_onnx.py --dtype fp32 || echo "FP32 Export Failed"

echo "2. Exporting FP16 ONNX..."
python export_onnx.py --dtype fp16 || echo "FP16 Export Failed"

echo "3. Exporting INT8 ONNX..."
python export_int8_onnx.py || echo "INT8 Export Failed"

echo "4. Exporting NVFP4 ONNX..."
python export_nvfp4_onnx.py || echo "NVFP4 Export Failed"

echo "5. Exporting INT4 ONNX..."
python export_int4_onnx.py || echo "INT4 Export Failed"

echo "6. Exporting FP8 ONNX..."
python export_fp8_onnx.py || echo "FP8 Export Failed"

echo "All exports attempted."
