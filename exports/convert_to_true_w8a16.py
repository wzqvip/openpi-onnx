
import onnx
from onnxconverter_common import float16
import os

INPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.w8a16.patched.onnx"
OUTPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.w8a16.final.onnx"
DATA_FILE = "model.w8a16.final.onnx.data"

def main():
    if not os.path.exists(INPUT_MODEL):
        print(f"Error: {INPUT_MODEL} not found.")
        return

    print(f"Loading {INPUT_MODEL}...")
    model = onnx.load(INPUT_MODEL)
    
    print("Converting to Float16...")
    # keep_io_types=False => Inputs/Outputs become FP16
    try:
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
        print(f"Converted. Nodes: {len(model_fp16.graph.node)}, Initializers: {len(model_fp16.graph.initializer)}")
    except Exception as e:
        print(f"Conversion error: {e}")
        return
    
    print(f"Saving to {OUTPUT_MODEL}...")
    if os.path.exists(OUTPUT_MODEL):
        os.remove(OUTPUT_MODEL)
    # Cleanup data file only if it is in the same dir and we are overwriting
    data_path_full = os.path.join(os.path.dirname(OUTPUT_MODEL), DATA_FILE)
    if os.path.exists(data_path_full):
        os.remove(data_path_full)
        
    try:
        onnx.save_model(
            model_fp16,
            OUTPUT_MODEL,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=DATA_FILE,
            size_threshold=1024,
            convert_attribute=False
        )
        print("Save successful.")
    except Exception as e:
        print(f"Save error: {e}")

if __name__ == "__main__":
    main()
