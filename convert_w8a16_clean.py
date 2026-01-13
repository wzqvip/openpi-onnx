
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

INPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.fp32.onnx"
OUTPUT_DIR = "./checkpoints/pi05_libero_pytorch/w8a16_clean"
OUTPUT_MODEL = os.path.join(OUTPUT_DIR, "model.onnx")

def quantize_w8a16_clean():
    print(f"Quantizing {INPUT_MODEL} to {OUTPUT_MODEL}...")
    
    # Ensure input exists
    if not os.path.exists(INPUT_MODEL):
        print(f"Error: {INPUT_MODEL} does not exist")
        return

    # Use quantize_dynamic. It quantizes weights to Int8.
    # It also inserts DynamicQuantizeLinear for activations.
    # This is "Dynamic Quantization".
    # For GPU execution with TensorRT, TRT might fuse these or just run them.
    # But weights will be Int8 (small file).
    
    quantize_dynamic(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        weight_type=QuantType.QInt8,
        use_external_data_format=True,
        extra_options={
            "save_as_external_data": True,
            "all_tensors_to_one_file": True,
            "external_data_location": "model.onnx.data",
            "external_data_size_threshold": 1024
        }
    )
    
    print(f"Saved to {OUTPUT_MODEL}")
    print(f"Checking file size...")
    print(f"Size: {os.path.getsize(OUTPUT_MODEL)} bytes")
    data_file = os.path.join(OUTPUT_DIR, "model.onnx.data")
    if os.path.exists(data_file):
        print(f"Data Size: {os.path.getsize(data_file)} bytes")

if __name__ == "__main__":
    quantize_w8a16_clean()
