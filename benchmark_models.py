
import torch
import torch.nn as nn
import onnxruntime as ort
import time
import numpy as np
import os
import modelopt.torch.quantization
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models import model as _model

# Configuration
checkpoint_dir = "./checkpoints/pi05_libero_pytorch"
config_name = "pi05_libero"
onnx_path = "./checkpoints/pi05_libero_pytorch/model.onnx"
mxfp8_path = "./checkpoints/pi05_libero_pytorch/model.mxfp8.onnx"
int8_path = "./checkpoints/pi05_libero_pytorch/model.int8.onnx"

# HACK: Add TensorRT libs to LD_LIBRARY_PATH for ORT
import site
try:
    trt_libs = os.path.join(site.getsitepackages()[0], "tensorrt_libs")
    if os.path.exists(trt_libs):
        os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + trt_libs
        print(f"Added {trt_libs} to LD_LIBRARY_PATH")
except:
    pass

def create_dummy_inputs(batch_size, device, config):
    dtype = torch.float32 # Use float32 for inputs to be safe on CPU
    return (
        torch.randn(batch_size, 3, 224, 224, dtype=dtype, device=device), # base
        torch.randn(batch_size, 3, 224, 224, dtype=dtype, device=device), # left
        torch.zeros(batch_size, 3, 224, 224, dtype=dtype, device=device), # right
        torch.randn(batch_size, 32, dtype=dtype, device=device),          # state
        torch.randint(0, 100, (batch_size, config.model.max_token_len), dtype=torch.int32, device=device), # prompt
        torch.ones(batch_size, config.model.max_token_len, dtype=torch.bool, device=device),  # prompt_mask
        torch.randn(batch_size, config.model.action_horizon, config.model.action_dim, dtype=dtype, device=device) # noise
    )

def create_numpy_inputs(dummy_inputs):
    inputs = {
        "base_0_rgb": dummy_inputs[0].cpu().numpy(),
        "left_wrist_0_rgb": dummy_inputs[1].cpu().numpy(),
        "right_wrist_0_rgb": dummy_inputs[2].cpu().numpy(),
        "state": dummy_inputs[3].cpu().numpy(),
        "tokenized_prompt": dummy_inputs[4].cpu().numpy(),
        "tokenized_prompt_mask": dummy_inputs[5].cpu().numpy(),
        "noise": dummy_inputs[6].cpu().numpy()
    }
    return inputs

def reconstruct_obs_and_call(model, inputs):
    # Wrapper helper for PyTorch call
    base, left, right, state, prompt, prompt_mask, noise = inputs
    images = {"base_0_rgb": base, "left_wrist_0_rgb": left, "right_wrist_0_rgb": right}
    image_masks = {k: torch.ones(base.shape[0], dtype=torch.bool, device=base.device) for k in images}
    obs = _model.Observation(images=images, image_masks=image_masks, state=state, tokenized_prompt=prompt, tokenized_prompt_mask=prompt_mask)
    return type(model).sample_actions(model, device=base.device, observation=obs, noise=noise, num_steps=10)

def benchmark_pytorch(config, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU for PyTorch benchmark.")
        device = "cpu"
        
    print(f"\n--- Benchmarking PyTorch (Float32) on {device} ---")
    try:
        policy = policy_config.create_trained_policy(config, checkpoint_dir, pytorch_device=device)
        model = policy._model
        if hasattr(model, "gradient_checkpointing_disable"): model.gradient_checkpointing_disable()
        model.to(torch.float32)
        model.eval()

        dummy_inputs = create_dummy_inputs(1, device, config)
        
        # Warmup
        print("Warmup...")
        for _ in range(2):
            with torch.no_grad():
                reconstruct_obs_and_call(model, dummy_inputs)
        
        if device == "cuda": torch.cuda.synchronize()
        start = time.time()
        iters = 5 # Reduced iterations for CPU speed
        print(f"Running {iters} iterations...")
        for _ in range(iters):
            with torch.no_grad():
                reconstruct_obs_and_call(model, dummy_inputs)
        if device == "cuda": torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / iters * 1000
        print(f"PyTorch Average Latency: {avg_time:.2f} ms")
        del model
        if device == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print(f"PyTorch Benchmark Failed: {e}")
        import traceback
        traceback.print_exc()

def benchmark_onnx(model_path, name="Standard"):
    print(f"\n--- Benchmarking ONNX ({name}) ---")
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}. Skipping.")
        return

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        print(f"Active Providers: {session.get_providers()}")
        
        # Create Inputs (Numpy)
        conf = _config.get_config(config_name)
        dummy_tensors = create_dummy_inputs(1, "cpu", conf)
        inputs = create_numpy_inputs(dummy_tensors)
        
        # Warmup
        print("Warmup...")
        for _ in range(2):
            session.run(None, inputs)
            
        start = time.time()
        iters = 5
        print(f"Running {iters} iterations...")
        for _ in range(iters):
            session.run(None, inputs)
        end = time.time()
        
        avg_time = (end - start) / iters * 1000
        print(f"ONNX ({name}) Average Latency: {avg_time:.2f} ms")
        del session
    except Exception as e:
        print(f"ONNX ({name}) Benchmark Failed: {e}")

def main():
    # HACK: Monkey patch gemma.get_config to return tiny configs
    import openpi.models.gemma as _gemma_mod
    original_get_config = _gemma_mod.get_config
    def tiny_get_config(variant):
        c = original_get_config(variant)
        if hasattr(c, "depth"): c.depth = 1
        if hasattr(c, "num_hidden_layers"): c.num_hidden_layers = 1
        try: c.vocab_size = 1024
        except: pass
        if hasattr(c, "vision_config"): c.vision_config.num_hidden_layers = 1
        if hasattr(c, "text_config"): 
            c.text_config.num_hidden_layers = 1
            c.text_config.vocab_size = 1024
        return c
    _gemma_mod.get_config = tiny_get_config

    config = _config.get_config(config_name)
    
    # 1. PyTorch
    benchmark_pytorch(config)
    
    # 2. Standard ONNX
    benchmark_onnx(onnx_path, "FP32/FP16")
    
    # 3. Optimized ONNX
    benchmark_onnx(mxfp8_path, "MXFP8")

    # 4. INT8 ONNX
    benchmark_onnx(int8_path, "INT8")

if __name__ == "__main__":
    main()
