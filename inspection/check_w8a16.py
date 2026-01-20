
import onnx

MODEL_PATH = "./checkpoints/pi05_libero_pytorch/model.w8a16.onnx"

def check_model():
    print(f"Loading {MODEL_PATH}...")
    try:
        model = onnx.load(MODEL_PATH, load_external_data=False)
        print("Model loaded successfully (without external data).")
        print(f"IR Version: {model.ir_version}")
        print(f"Opset Import: {model.opset_import}")
        if len(model.opset_import) == 0:
            print("WARNING: Opset is empty!")
            # Fix it
            op = model.opset_import.add()
            op.domain = ""
            op.version = 18
            print("Fixed Opset to 18.")
            onnx.save(model, MODEL_PATH)
            print("Saved with fixed opset.")
            
        print(f"Graph Nodes: {len(model.graph.node)}")
        print(f"Graph Initializers: {len(model.graph.initializer)}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_model()
