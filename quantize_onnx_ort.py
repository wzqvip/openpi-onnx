import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

input_model_path = "./checkpoints/pi05_libero_pytorch/model.fp32.onnx"
output_model_path = "./checkpoints/pi05_libero_pytorch/model.int8.onnx"

print(f"Quantizing {input_model_path} to {output_model_path}...")

# Use dynamic quantization for CPU speedup
quantize_dynamic(
    input_model_path,
    output_model_path,
    weight_type=QuantType.QUInt8
)

print("Quantization complete.")
