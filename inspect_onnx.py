import onnx
from onnx import helper

model_path = "checkpoints/pi05_libero_pytorch/fp32/model.onnx"
print(f"Loading {model_path}...")
model = onnx.load(model_path)

graph = model.graph
print("Checking CumSum nodes...")

for node in graph.node:
    if node.op_type == "CumSum":
        print(f"\nFound CumSum node: {node.name}")
        input_name = node.input[0]
        print(f"  Input: {input_name}")
        
        # Find input type
        # Check value_info
        found = False
        for vi in graph.value_info:
            if vi.name == input_name:
                print(f"  Input Type in value_info: {vi.type.tensor_type.elem_type}")
                # 1=FLOAT, 2=UINT8, 3=INT8, 4=UINT16, 5=INT16, 6=INT32, 7=INT64, 9=BOOL
                found = True
                break
        if not found:
             # Check inputs
             for inp in graph.input:
                if inp.name == input_name:
                    print(f"  Input Type in graph input: {inp.type.tensor_type.elem_type}")
                    found = True
                    break
        if not found:
            # Check initializers? (Unlikely for data path)
            print("  Input Type not found in value_info/input.")
            
        print(f"  Outputs: {node.output}")
