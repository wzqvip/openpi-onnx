
import torch
import onnxruntime as ort
import time
import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models import model as _model

# Configuration
checkpoint_dir = "/local/scratch1/wang.20306/openpi/checkpoints/pi05_libero_pytorch"
config_name = "pi05_libero"
int8_path = "/local/scratch1/wang.20306/openpi/checkpoints/pi05_libero_pytorch/model.int8.onnx"

def create_dummy_inputs(batch_size, device, config):
    return (
        torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device=device), # base
        torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device=device), # left
        torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32, device=device), # right
        torch.randn(batch_size, 32, dtype=torch.float32, device=device),          # state
        torch.randint(0, 100, (batch_size, config.model.max_token_len), dtype=torch.int32, device=device), # prompt
        torch.ones(batch_size, config.model.max_token_len, dtype=torch.bool, device=device),  # prompt_mask
        torch.randn(batch_size, config.model.action_horizon, config.model.action_dim, dtype=torch.float32, device=device) # noise
    )

def create_numpy_inputs(dummy_inputs):
    inputs = {
        "base_0_rgb": dummy_inputs[0].cpu().numpy(),
        "left_wrist_0_rgb": dummy_inputs[1].cpu().numpy(),
        "right_wrist_0_rgb": dummy_inputs[2].cpu().numpy(),
        "state": dummy_inputs[3].cpu().numpy(),
        "tokenized_prompt": dummy_inputs[4].cpu().numpy(),
        "tokenized_prompt_mask": dummy_inputs[5].cpu().numpy().astype(np.int32),
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
    print("\n--- Benchmarking PyTorch (Float32) ---")
    try:
        policy = policy_config.create_trained_policy(config, checkpoint_dir, pytorch_device=device)
        model = policy._model
        if hasattr(model, "gradient_checkpointing_disable"): model.gradient_checkpointing_disable()
        model.to(torch.float32)
        model.eval()

        dummy_inputs = create_dummy_inputs(1, device, config)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                reconstruct_obs_and_call(model, dummy_inputs)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(20):
            with torch.no_grad():
                reconstruct_obs_and_call(model, dummy_inputs)
        torch.cuda.synchronize()
        end = time.time()
        
        avg_time = (end - start) / 20 * 1000
        print(f"PyTorch Average Latency: {avg_time:.2f} ms")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"PyTorch Benchmark Failed: {e}")

def benchmark_onnx(model_path, name="Standard"):
    print(f"\n--- Benchmarking ONNX ({name}) ---")
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # PROVIDERS
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        print(f"Active Providers: {session.get_providers()}")
        
        # Create Inputs (Numpy)
        conf = _config.get_config(config_name)
        dummy_tensors = create_dummy_inputs(1, "cpu", conf)
        inputs = create_numpy_inputs(dummy_tensors)
        
        # Warmup
        for _ in range(5):
            session.run(None, inputs)
            
        start = time.time()
        for _ in range(20):
            session.run(None, inputs)
        end = time.time()
        
        avg_time = (end - start) / 20 * 1000
        print(f"ONNX ({name}) Average Latency: {avg_time:.2f} ms")
        del session
    except Exception as e:
        print(f"ONNX ({name}) Benchmark Failed: {e}")

def main():
    config = _config.get_config(config_name)
    
    # 1. PyTorch
    benchmark_pytorch(config)
    
    # 2. INT8 ONNX
    benchmark_onnx(int8_path, "INT8")

if __name__ == "__main__":
    main()
