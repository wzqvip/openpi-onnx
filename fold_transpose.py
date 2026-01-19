
import onnx
from onnx import numpy_helper, TensorProto
import numpy as np

INPUT_MODEL = "./dist/final_fp16/model.fp32.onnx"
OUTPUT_MODEL = "./checkpoints/pi05_libero_pytorch/model.folded.onnx"

def fold_transpose():
    print(f"Loading {INPUT_MODEL}...")
    model = onnx.load(INPUT_MODEL)
    graph = model.graph
    
    # Map initializer name to initializer object
    init_map = {init.name: init for init in graph.initializer}
    
    # Count usage of initializers
    init_usage = {}
    for node in graph.node:
        for inp in node.input:
            if inp in init_map:
                init_usage[inp] = init_usage.get(inp, 0) + 1
    
    nodes_to_remove = []
    inits_to_remove = set()
    
    print("Scanning for Transpose -> Initializer pattern...")
    folded_count = 0
    
    for node in graph.node:
        if node.op_type == "Transpose":
            input_name = node.input[0]
            if input_name in init_map:
                # Found Transpose of Initializer!
                init = init_map[input_name]
                
                # Get perm
                perm = []
                for attr in node.attribute:
                    if attr.name == "perm":
                         perm = list(attr.ints)
                         pass
                
                if not perm:
                    # Default perm? Or missing?
                    print(f"Skipping Transpose {node.name}: no perm attribute found.")
                    continue
                    
                # Load header
                try:
                    tensor = numpy_helper.to_array(init)
                    # Transpose
                    new_tensor = np.transpose(tensor, perm)
                    
                    # Create new initializer
                    new_init_name = node.output[0]
                    new_init = numpy_helper.from_array(new_tensor, name=new_init_name)
                    
                    # Add to graph
                    graph.initializer.append(new_init)
                    
                    nodes_to_remove.append(node)
                    folded_count += 1
                    
                    # Decrement usage of old init
                    if input_name in init_usage:
                        init_usage[input_name] -= 1
                        if init_usage[input_name] <= 0:
                            inits_to_remove.add(input_name)
                    
                except Exception as e:
                    print(f"Error folding {node.name}: {e}")
    
    print(f"Folded {folded_count} Transpose nodes.")
    print(f"Removing {len(inits_to_remove)} orphaned initializers.")
    
    if folded_count > 0:
        # Remove folded nodes
        new_nodes = [n for n in graph.node if n not in nodes_to_remove]
        graph.ClearField("node")
        graph.node.extend(new_nodes)
        
        # Remove orphaned inits
        new_inits = [i for i in graph.initializer if i.name not in inits_to_remove]
        graph.ClearField("initializer")
        graph.initializer.extend(new_inits)
        
        print(f"Saving folded model to {OUTPUT_MODEL}...")
        import os
        if os.path.exists(OUTPUT_MODEL):
            os.remove(OUTPUT_MODEL)
        data_path = "model.folded.onnx.data"
        full_data_path = os.path.join(os.path.dirname(OUTPUT_MODEL), data_path)
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
    fold_transpose()
