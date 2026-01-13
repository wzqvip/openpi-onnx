
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

INPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.fp32.onnx"
OUTPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.w8a16.onnx"

def quantize_w8_weight_only():
    print(f"Quantizing {INPUT_MODEL} to W8 (WeightOnly Int8) using MatMulNBitsQuantizer...")
    
    # n_bits=8 means 8-bit weights.
    # block_size=128 is standard for weight only, or -1 for per-channel?
    # Usually per-channel (-1) is better for accuracy if supported, or per-block.
    # TensorRT supports blocked or per-channel.
    # We'll try block_size=128 for now as it's common.
    
    quantizer = MatMulNBitsQuantizer(
        INPUT_MODEL,
        n_bits=8,
        block_size=128,
        symmetric=True, # INT8 is usually symmetric [-127, 127]
        accuracy_level=None
    )
    
    quantizer.process(
        model_output=OUTPUT_MODEL,
        use_external_data_format=True
    )
    
    print(f"Quantized W8 model saved to {OUTPUT_MODEL}")

if __name__ == "__main__":
    quantize_w8_weight_only()
