
import onnxruntime as ort
import torch
import time
import numpy as np
import os
import pynvml
import time

# Model Path
MODEL_PATH = "/home/taco/openpi-onnx/checkpoints/pi05_libero_pytorch/model.w8a16.onnx"

def get_gpu_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2  # MiB
    except:
        return 0

from openpi.training import config as _config

CONFIG_NAME = "pi05_libero"

def main():
    print(f"Benchmarking W8A16 Model: {MODEL_PATH}")
    
    config = _config.get_config(CONFIG_NAME)
    max_token_len = config.model.max_token_len
    action_horizon = config.model.action_horizon
    action_dim = config.model.action_dim
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    # Providers
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024, # 24GB
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ]

    print("Loading model via ONNX Runtime...")
    start_mem = get_gpu_memory()
    try:
        sess = ort.InferenceSession(MODEL_PATH, providers=providers)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    loaded_mem = get_gpu_memory()
    print(f"Model Loaded. VRAM Increase: {loaded_mem - start_mem:.2f} MiB")

    # Dummy Inputs
    B = 1
    # Match shapes from export script
    inputs = {
        "base_0_rgb": np.random.randn(B, 3, 224, 224).astype(np.float16),
        "left_wrist_0_rgb": np.random.randn(B, 3, 224, 224).astype(np.float16),
        "right_wrist_0_rgb": np.zeros((B, 3, 224, 224), dtype=np.float16),
        "state": np.random.randn(B, 32).astype(np.float16),
        "tokenized_prompt": np.random.randint(0, 100, (B, max_token_len), dtype=np.int32), 
        "tokenized_prompt_mask": np.ones((B, max_token_len), dtype=bool),
        "noise": np.random.randn(B, action_horizon, action_dim).astype(np.float16) 
    }

    # Warmup
    print("Warming up...")
    for _ in range(5):
        sess.run(None, inputs)

    # Benchmark
    print("Benchmarking...")
    latencies = []
    num_iters = 50
    
    for _ in range(num_iters):
        start_t = time.time()
        sess.run(None, inputs)
        end_t = time.time()
        latencies.append((end_t - start_t) * 1000) # ms

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    throughput = 1000 / avg_latency
    
    peak_mem = get_gpu_memory()
    
    print("\nResults:")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"P95 Latency: {p95_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} QPS")
    print(f"Peak VRAM: {peak_mem:.2f} MiB")
    print(f"Approx Weights Size (VRAM): {loaded_mem - start_mem:.2f} MiB")

if __name__ == "__main__":
    main()
