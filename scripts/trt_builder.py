import tensorrt as trt
import os

def build_engine(onnx_file_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, logger)

    # with open(onnx_file_path, "rb") as model:
    #    if not parser.parse(model.read()):
    if not parser.parse_from_file(onnx_file_path):
        print("Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None
            
    # Optimization profile
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = tensor.shape
        
        min_shape = [1 if x == -1 else x for x in shape]
        opt_shape = [1 if x == -1 else x for x in shape]
        max_shape = [1 if x == -1 else x for x in shape]
        
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    
    config.add_optimization_profile(profile)
    
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    engine_bytes = builder.build_serialized_network(network, config)
    return engine_bytes
