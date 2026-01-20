
import onnx
from onnx import helper, TensorProto
import os

TEMP_PATH = "./checkpoints/pi05_libero_pytorch/model.w8a16.onnx.temp.onnx"
OUTPUT_PATH = "./checkpoints/pi05_libero_pytorch/model.w8a16_qdq.onnx"

def fix_and_save():
    print(f"Loading {TEMP_PATH}...")
    model = onnx.load(TEMP_PATH)
    
    # PATCH CumSum
    print("Patching CumSum nodes...")
    new_nodes = []
    patched_count = 0
    for node in model.graph.node:
        if node.op_type == "CumSum":
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
    
    if patched_count > 0:
        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)
        print(f"Patched {patched_count} CumSum nodes.")

    print(f"Saving to {OUTPUT_PATH} with external data...")
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    data_path = os.path.basename(OUTPUT_PATH) + ".data"
    data_full_path = os.path.join(os.path.dirname(OUTPUT_PATH), data_path)
    if os.path.exists(data_full_path):
        os.remove(data_full_path)
        
    onnx.save_model(
        model,
        OUTPUT_PATH,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path,
        size_threshold=1024,
        convert_attribute=False
    )
    print("Done.")

if __name__ == "__main__":
    fix_and_save()
