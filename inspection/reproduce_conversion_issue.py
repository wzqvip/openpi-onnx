import onnx
from onnxconverter_common import float16
import os

# Path to the valid FP32 model (assuming it was saved as model.fp32.onnx by the export script)
# Wait, the export script saved it as model.onnx (because I ran it with default args first?)
# Let's check checks/pi05_libero_pytorch/
FP32_MODEL_PATH = "checkpoints/pi05_libero_pytorch/model.fp32.onnx"
if not os.path.exists(FP32_MODEL_PATH):
    # Fallback to the organized path if available, or just use the model.onnx if it's fp32
    FP32_MODEL_PATH = "checkpoints/pi05_libero_pytorch/model.onnx"

print(f"Loading FP32 model from: {FP32_MODEL_PATH}")
model_fp32 = onnx.load(FP32_MODEL_PATH)

if len(model_fp32.opset_import) > 0:
    print(f"FP32 Opset: {model_fp32.opset_import[0].version}")
else:
    print("FP32 Opset: None (Empty list)")

print("Converting to FP16...")
try:
    model_fp16 = float16.convert_float_to_float16(model_fp32)
except Exception as e:
    print(f"Conversion failed: {e}")
    exit(1)

if len(model_fp16.opset_import) > 0:
    print(f"FP16 Opset: {model_fp16.opset_import[0].version}")
else:
    print("FP16 Opset: None (Empty list) - FAILURE DETECTED")
