
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

INPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.fp32.onnx"
OUTPUT_DIR = "./checkpoints/pi05_libero_pytorch/w8a16_subset"
OUTPUT_MODEL = os.path.join(OUTPUT_DIR, "model.onnx")

def quantize_w8a16_subset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Quantizing {INPUT_MODEL} to {OUTPUT_MODEL} with subset ops...")
    
    # Only quantize MatMul and Gemm (Linear layers)
    op_types = ['MatMul', 'Gemm']
    
    quantize_dynamic(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types,
        use_external_data_format=True,
        extra_options={
            "save_as_external_data": True,
            "all_tensors_to_one_file": True,
            "external_data_location": "model.onnx.data",
            "external_data_size_threshold": 1024
        }
    )
    
    print(f"Saved to {OUTPUT_MODEL}")
    if os.path.exists(OUTPUT_MODEL):
       print(f"Size: {os.path.getsize(OUTPUT_MODEL)}")
    
if __name__ == "__main__":
    quantize_w8a16_subset()
