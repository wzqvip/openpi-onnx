
import onnx
import os

INPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.w4a16.onnx"
OUTPUT_DIR = "./checkpoints/pi05_libero_pytorch/w4a16_final"
OUTPUT_MODEL = os.path.join(OUTPUT_DIR, "model.onnx")

def clean_model():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading {INPUT_MODEL}...")
    # Load without external data initially to save memory? No, need to traverse.
    model = onnx.load(INPUT_MODEL, load_external_data=False) 
    
    # We can't verify unused without loading everything usually.
    # But usually save_model with all_tensors_to_one_file handles it if we create a clean copy?
    # No, it copies referenced tensors.
    
    # Let's try to remove initializers that are NOT inputs to any node?
    # This is hard.
    
    # Better: Use onnx.shape_inference?
    
    # Actually, verify if graph.initializer contains the FP32 weights.
    # MatMulNBits replaces them with packed Int4 weights.
    
    print("Checking initializers...")
    used_inputs = set()
    for node in model.graph.node:
        for i in node.input:
            used_inputs.add(i)
            
    # Filter initializers
    new_inits = []
    removed_count = 0
    removed_bytes = 0
    
    for init in model.graph.initializer:
        if init.name in used_inputs:
             new_inits.append(init)
        else:
             removed_count += 1
             # We can't analyze size easily without loading data if it's external.
             
    print(f"Removing {removed_count} unused initializers.")
    
    if removed_count > 0:
        model.graph.ClearField("initializer")
        model.graph.initializer.extend(new_inits)
        
    print(f"Saving cleaned model to {OUTPUT_MODEL}...")
    if os.path.exists(OUTPUT_MODEL):
        os.remove(OUTPUT_MODEL)
    data_path = "model.onnx.data"
    full_data_path = os.path.join(OUTPUT_DIR, data_path)
    if os.path.exists(full_data_path):
        os.remove(full_data_path)

    onnx.save_model(
        model,
        OUTPUT_MODEL,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
        size_threshold=1024,
        convert_attribute=False
    )
    print("Done.")

if __name__ == "__main__":
    clean_model()
