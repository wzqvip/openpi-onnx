
import onnx
import os

TEMP_PATH = "./checkpoints/pi05_libero_pytorch/model.w8a16.onnx.temp.onnx"

def inspect_temp():
    if not os.path.exists(TEMP_PATH):
        print("Temp file not found.")
        return

    print(f"Loading {TEMP_PATH}...")
    try:
        model = onnx.load(TEMP_PATH)
        print("Loaded.")
        print(f"Nodes: {len(model.graph.node)}")
        print(f"Initializers: {len(model.graph.initializer)}")
        
        # Check size of first initializer
        if len(model.graph.initializer) > 0:
            init = model.graph.initializer[0]
            print(f"First Init: {init.name}, Size: {len(init.raw_data)} bytes")
            # Check external data location
            if init.HasField("data_location"):
                print(f"External Data Location: {init.data_location}")
            else:
                print("No external data location set on initializer.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_temp()
