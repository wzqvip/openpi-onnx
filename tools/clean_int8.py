
import onnx
import os

INPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.int8.onnx"
OUTPUT_DIR = "./checkpoints/pi05_libero_pytorch/int8_final"
OUTPUT_MODEL = os.path.join(OUTPUT_DIR, "model.onnx")

def clean_model():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading {INPUT_MODEL}...")
    model = onnx.load(INPUT_MODEL) # Load external data to traverse
    
    print("Checking initializers...")
    graph = model.graph
    
    used_inputs = set()
    for node in graph.node:
        for i in node.input:
            used_inputs.add(i)
            
    # Filter initializers
    new_inits = []
    removed_count = 0
    
    for init in graph.initializer:
        if init.name in used_inputs:
             new_inits.append(init)
        else:
             removed_count += 1
             
    print(f"Removing {removed_count} unused initializers.")
    
    if removed_count > 0:
        graph.ClearField("initializer")
        graph.initializer.extend(new_inits)
        
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
