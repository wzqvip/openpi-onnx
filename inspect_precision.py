
import onnx
import sys
import os

def inspect_onnx(path):
    print(f"Inspecting: {path}")
    if not os.path.exists(path):
        print("File not found")
        return

    try:
        # Load the model, but don't load external data to be fast, just check graph
        model = onnx.load(path, load_external_data=False)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    print(f"  Opset version: {model.opset_import[0].version if model.opset_import else 'Unknown'}")
    
    # Check inputs
    print("  Inputs:")
    for inp in model.graph.input:
        elem_type = inp.type.tensor_type.elem_type
        print(f"    {inp.name}: {onnx.TensorProto.DataType.Name(elem_type)}")

    # Check first few weights (initializers)
    print("  Weights (first 5):")
    for i, init in enumerate(model.graph.initializer[:5]):
        print(f"    {init.name}: {onnx.TensorProto.DataType.Name(init.data_type)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_precision.py <path_to_onnx>")
    else:
        for p in sys.argv[1:]:
            inspect_onnx(p)
