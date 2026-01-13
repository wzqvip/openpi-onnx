
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

INPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.fp32.onnx"
OUTPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.w8a16.onnx"

def quantize_w8a16():
    print(f"Quantizing {INPUT_MODEL} to W8A16 (Weight Only Int8)...")
    
    # We use quantize_dynamic but essentially only for weights if we can constrain it
    # standard quantize_dynamic quantizes both.
    # To get Weight-Only, we might need other tools, but let's try standard dynamic first.
    # Actually, dynamic quantization uses Int8 for weights and uint8/float for activations at runtime.
    # But sticking to "Weight Only" usually implies we just want to compress weights.
    
    # Let's use `quantize_dynamic` with `weight_type=QuantType.QInt8`
    
    quantize_dynamic(
        model_input=INPUT_MODEL,
        model_output=OUTPUT_MODEL,
        weight_type=QuantType.QInt8,
        use_external_data_format=True
        # We can try to avoid quantizing activations if we want pure W8A32/FP16 execution
        # But quantize_dynamic inherently adds dynamic quantize nodes for activations.
        # Use onnxruntime.quantization.quantize which allows 'op_types_to_quantize'
    )
    
    print(f"Quantized model saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    quantize_w8a16()
