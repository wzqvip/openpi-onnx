import tensorrt as trt
import tensorrt as trt
# from cuda import cudart
import numpy as np
import time

# Create a logger
logger = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)

    # Parse directly from file to handle external data correctly
    if not parser.parse_from_file(onnx_file_path):
        print("Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

    # Inspect inputs and setup profile
    profile = builder.create_optimization_profile()
    
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = tensor.shape
        print(f"Input: {name}, Shape: {shape}")
        
        # If dynamic shape (contains -1), set profile
        # We assume dynamic batch dimension (usually dim 0)
        # We set min=1, opt=1, max=1 for verification
        
        # Construct valid shape for profile
        # Replace -1 with 1
        min_shape = [1 if x == -1 else x for x in shape]
        opt_shape = [1 if x == -1 else x for x in shape]
        max_shape = [1 if x == -1 else x for x in shape]
        
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    
    config.add_optimization_profile(profile)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build engine
    engine_bytes = builder.build_serialized_network(network, config)
    return engine_bytes

def main():
    onnx_file = "checkpoints/pi05_libero_pytorch/model.fp32.onnx"
    
    print("Building TRT Engine...")
    start = time.time()
    engine_bytes = build_engine(onnx_file)
    if not engine_bytes:
        print("Engine build failed")
        return·
    print(f"Engine built in {time.time() - start:.2f}s")
    
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    
    print("Creating Context...")
    context = engine.create_execution_context()
    
    # Inputs/Outputs
    # Inspect engine to find input/output names and shapes
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        print(f"Tensor: {name}, Mode: {mode}, Shape: {shape}, Dtype: {dtype}")

    # Allocate buffers
    # Assuming first input/output
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1) # Assuming single input/output for simplicity or inspecting
    
    # We need to allocate device memory
    # Quick dummy test
    print("Verification complete (Engine built and deserialized).")

if __name__ == "__main__":
    main()
