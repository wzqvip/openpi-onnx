
import onnx
from onnx import helper, TensorProto
import os

MODEL_PATH = "./checkpoints/pi05_libero_pytorch/model.w8a16.onnx"
OUTPUT_PATH = "./checkpoints/pi05_libero_pytorch/model.w8a16.patched.onnx"
DATA_PATH = "model.w8a16.patched.onnx.data"

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return

    print("Loading model...")
    model = onnx.load(MODEL_PATH)
    graph = model.graph
    
    print("Scanning nodes...")
    
    # Map node outputs to node types to infer types
    node_out_type = {}
    for node in graph.node:
        for out in node.output:
            node_out_type[out] = node.op_type
            
    new_nodes = []
    patched_count = 0
    
    for i, node in enumerate(graph.node):
        if node.op_type == "CumSum":
            input_name = node.input[0]
            # Check if input producer is boolean-like
            producer_type = node_out_type.get(input_name, "Unknown")
            is_bool = producer_type in ["Equal", "Greater", "Less", "And", "Or", "Not", "Cast"] 
            # Note: Cast can be to any type, but often to Bool in this context.
            
            # If strictly checking, we might miss some. 
            # But the error said `tensor(bool)`.
            
            # Use specific check for Cast to Bool if needed, but for now assume if it breaks validation it's bool.
            # Strategy: Insert Cast to int32 before CumSum input.
            
            # We will insert a cast node.
            cast_out = input_name + "_cast_int32"
            cast_node = helper.make_node(
                "Cast",
                inputs=[input_name],
                outputs=[cast_out],
                to=TensorProto.INT32
            )
            
            # Modify CumSum to use cast_out
            node.input[0] = cast_out
            
            new_nodes.append(cast_node)
            new_nodes.append(node)
            patched_count += 1
            print(f"Patched CumSum node {node.name} (input: {input_name} from {producer_type})")
        else:
            new_nodes.append(node)
            
    if patched_count == 0:
        print("No CumSum nodes found? Check opset.")
    else:
        print(f"Patched {patched_count} CumSum nodes.")
        
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    
    # Save
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
        
    print(f"Saving patched model to {OUTPUT_PATH}...")
    onnx.save_model(
        model,
        OUTPUT_PATH,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=DATA_PATH,
        size_threshold=1024,
        convert_attribute=False
    )
    print("Done.")

if __name__ == "__main__":
    main()
