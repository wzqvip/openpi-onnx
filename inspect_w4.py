
import onnx

MODEL_PATH = "./checkpoints/pi05_libero_pytorch/model.w4a16.onnx"

def inspect():
    print(f"Loading {MODEL_PATH}...")
    model = onnx.load(MODEL_PATH, load_external_data=False)
    
    print(f"Nodes: {len(model.graph.node)}")
    print(f"Initializers: {len(model.graph.initializer)}")
    
    # Check first few initializers size
    for i, init in enumerate(model.graph.initializer[:5]):
        # data_location
        loc = "External" if init.HasField("data_location") else "Embedded"
        print(f"Init {i}: {init.name}, Loc: {loc}")

    # Check first few MatMul/Gemm nodes
    print("\nScanning nodes for quantization...")
    matmul_nbits_count = 0
    matmul_count = 0
    gemm_count = 0
    
    init_map = {init.name: init for init in model.graph.initializer}
    
    for node in model.graph.node:
        if node.op_type == "MatMulNBits":
            matmul_nbits_count += 1
        elif node.op_type == "MatMul":
            matmul_count += 1
            if matmul_count <= 2:
                 print(f"MatMul Node: {node.name}, Inputs: {node.input}")
                 # Check weight shape
                 if len(node.input) > 1:
                     weight_name = node.input[1]
                     if weight_name in init_map:
                         init = init_map[weight_name]
                         dims = list(init.dims)
                         print(f"  Weight: {weight_name}, Shape: {dims}")
                     else:
                         print(f"  Weight {weight_name} NOT in initializers.")
        elif node.op_type == "Gemm":
            gemm_count += 1
            if gemm_count <= 2:
                 print(f"Gemm Node: {node.name}, Inputs: {node.input}")

    print(f"\nSummary:")
    print(f"MatMulNBits: {matmul_nbits_count}")
    print(f"MatMul: {matmul_count}")
    print(f"Gemm: {gemm_count}")

if __name__ == "__main__":
    inspect()
