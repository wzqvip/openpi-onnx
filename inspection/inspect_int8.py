
import onnx

MODEL_PATH = "./checkpoints/pi05_libero_pytorch/int8_final/model.onnx"

def inspect():
    print(f"Loading {MODEL_PATH}...")
    try:
        model = onnx.load(MODEL_PATH, load_external_data=False)
    except:
        # Fallback to int8 root if final doesn't exist
        model = onnx.load("./checkpoints/pi05_libero_pytorch/model.int8.onnx", load_external_data=False)

    
    matmul_int_count = 0
    matmul_count = 0
    gemm_count = 0
    
    for node in model.graph.node:
        if node.op_type == "MatMulInteger":
            matmul_int_count += 1
        elif node.op_type == "MatMul":
            matmul_count += 1
        elif node.op_type == "Gemm":
            gemm_count += 1

    print(f"\nSummary:")
    print(f"MatMulInteger: {matmul_int_count}")
    print(f"MatMul: {matmul_count}")
    print(f"Gemm: {gemm_count}")

if __name__ == "__main__":
    inspect()
