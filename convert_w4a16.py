
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from onnxruntime.quantization import QuantFormat

import logging
logging.basicConfig(level=logging.INFO)
import shutil

INPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.folded.onnx"
OUTPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.w4a16.onnx"

def quantize_w4_weight_only():
    print(f"Quantizing {INPUT_MODEL} to W4...")
    
    # Initialize quantizer with QDQ format for TensorRT compatibility
    quantizer = MatMulNBitsQuantizer(
        model=INPUT_MODEL,
        block_size=128,
        is_symmetric=True,
        algo_config=None,
        quant_format=QuantFormat.QDQ
    )
    
    print("Executing quantization process...")
    quantizer.process()
    
    print(f"Quantizer attributes: {dir(quantizer)}")

    # Try to find the proto
    model_proto = None
    if hasattr(quantizer.model, 'model'):
        model_proto = quantizer.model.model
    elif hasattr(quantizer.model, 'proto'):
        model_proto = quantizer.model.proto
        
    if model_proto:
        print(f"Found proto of type: {type(model_proto)}")
        print(f"Saving W4 model to {OUTPUT_MODEL}...")
        import onnx
        onnx.save_model(
            model_proto,
            OUTPUT_MODEL,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.w4a16.onnx.data",
            size_threshold=1024,
            convert_attribute=False
        )
        print(f"Quantized W4 model saved to {OUTPUT_MODEL}")
    else:
        print("Error: Could not find ModelProto in ONNXModel wrapper.")

if __name__ == "__main__":
    quantize_w4_weight_only()
