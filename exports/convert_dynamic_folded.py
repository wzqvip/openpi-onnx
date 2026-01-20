
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.folded.onnx"
OUTPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.int8.onnx"

def run_dynamic_quant():
    print(f"Quantizing {INPUT_MODEL} to Int8 Dynamic...")
    
    quantize_dynamic(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        weight_type=QuantType.QUInt8,
        use_external_data_format=True
    )
    
    print(f"Saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    run_dynamic_quant()
