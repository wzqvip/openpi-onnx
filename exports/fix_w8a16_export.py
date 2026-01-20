
import onnx
from onnxconverter_common import float16
import os
import shutil

TEMP_PATH = "./checkpoints/pi05_libero_pytorch/model.w8a16.onnx.temp.onnx"
OUTPUT_PATH = "./checkpoints/pi05_libero_pytorch/model.w8a16.onnx"
DATA_PATH = "model.w8a16.onnx.data"

def main():
    if not os.path.exists(TEMP_PATH):
        print(f"Error: {TEMP_PATH} not found.")
        return

    print(f"Loading temp model form {TEMP_PATH}...")
    # load_external_data=True is default. 
    # It will look for the thousands of files in the same dir.
    model = onnx.load(TEMP_PATH)
    
    print(f"Model loaded. Graph nodes: {len(model.graph.node)}")
    
    # print("Converting to float16...")
    # try:
    #     model_fp16 = float16.convert_float_to_float16(model)
    # except Exception as e:
    #     print(f"Conversion failed: {e}")
    #     return
    model_fp16 = model

    print(f"Saving W8A16 model to {OUTPUT_PATH} with external data in {DATA_PATH}...")
    
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    if os.path.exists(os.path.join(os.path.dirname(OUTPUT_PATH), DATA_PATH)):
        os.remove(os.path.join(os.path.dirname(OUTPUT_PATH), DATA_PATH))

    try:
        onnx.save_model(
            model_fp16,
            OUTPUT_PATH,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=DATA_PATH,
            size_threshold=1024,
            convert_attribute=False
        )
        print("Save successful.")
    except Exception as e:
        print(f"Save failed: {e}")
        return
        
    print(f"Checking file size: {os.path.getsize(OUTPUT_PATH)} bytes")
    
if __name__ == "__main__":
    main()
