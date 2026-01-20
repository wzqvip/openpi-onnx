
import onnx
from onnx import helper, TensorProto
import os

MODEL_PATH = "./checkpoints/pi05_libero_pytorch/w8a16_clean/model.onnx"

def patch_cumsum_nodes(model_path):
    print(f"Loading {model_path}...")
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    graph = model.graph
    
    new_nodes = []
    patched_count = 0
    
    print("Scanning for CumSum nodes to patch...")
    for node in graph.node:
        if node.op_type == "CumSum":
            # Check inputs. If input[0] is Bool, we cast to Int32
            # For simplicity, we just inject a Cast(Int32) for the first input always
            # Assuming the issue is strict typing in TRT
            
            input_name = node.input[0]
            cast_out = input_name + "_cast_int32"
            
            cast_node = helper.make_node(
                "Cast",
                inputs=[input_name],
                outputs=[cast_out],
                to=TensorProto.INT32,
                name=node.name + "_cast_patch"
            )
            
            node.input[0] = cast_out
            new_nodes.append(cast_node)
            new_nodes.append(node)
            patched_count += 1
        else:
            new_nodes.append(node)

    print(f"Patched {patched_count} CumSum nodes.")
    
    if patched_count > 0:
        graph.ClearField("node")
        graph.node.extend(new_nodes)
        
        print(f"Saving patched model to {model_path}...")
        if os.path.exists(model_path):
            os.remove(model_path)
            
        onnx.save(model, model_path)
        print("Done.")
    else:
        print("No patching needed.")

if __name__ == "__main__":
    patch_cumsum_nodes(MODEL_PATH)
