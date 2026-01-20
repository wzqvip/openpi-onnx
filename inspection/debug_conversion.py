
import onnx
from onnxconverter_common import float16
import os

TEMP_FP32_PATH = "./checkpoints/pi05_libero_pytorch/model.fp16.temp_fp32.onnx"

def main():
    if not os.path.exists(TEMP_FP32_PATH):
        print("Temp model not found.")
        return

    print("Loading FP32 model...")
    model = onnx.load(TEMP_FP32_PATH)
    print(f"Nodes: {len(model.graph.node)}")
    
    print("Converting... (keep_io_types=True)")
    try:
        model1 = float16.convert_float_to_float16(model, keep_io_types=True)
        print(f"Result Nodes: {len(model1.graph.node)}")
    except Exception as e:
        print(f"Error: {e}")

    print("Converting... (keep_io_types=False)")
    try:
        model2 = float16.convert_float_to_float16(model, keep_io_types=False)
        print(f"Result Nodes: {len(model2.graph.node)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
