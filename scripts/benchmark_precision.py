import torch
import time
import os
import json
import numpy as np
from safetensors.torch import load_file
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch
from torchao.quantization.quant_api import Float8WeightOnlyConfig, quantize_

# Constants
FP32_PATH = "checkpoints/pi05_libero_pytorch"
BF16_PATH = "checkpoints/pi05_libero_bf16"
FP8_PATH = "checkpoints/pi05_libero_fp8"

NUM_WARMUP = 5
NUM_STEPS = 20
BATCH_SIZE = 1

def load_model(path, precision="fp32", device="cuda"):
    print(f"Loading {precision} model from {path}...")
    
    config_path = os.path.join(path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Force precision in config to match loading intention/availablity
    # For pi0_pytorch, it uses config.dtype to set model precision during init
    if precision == "fp32":
        dtype = torch.float32
        config_dict["precision"] = "float32" # or whatever the config expects
    elif precision == "bf16":
        dtype = torch.bfloat16
        config_dict["precision"] = "bfloat16"
    elif precision == "fp8":
        dtype = torch.bfloat16 # Base model is BF16, then quantized
        config_dict["precision"] = "bfloat16" 
    
    config = Pi0Config(
        action_dim=config_dict.get("action_dim", 32),
        action_horizon=config_dict.get("action_horizon", 10),
        paligemma_variant=config_dict.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=config_dict.get("action_expert_variant", "gemma_300m"),
        dtype=config_dict.get("precision", "bfloat16"), # use config value
    )
    
    model = pi0_pytorch.PI0Pytorch(config)
    
    # Load weights
    if os.path.exists(os.path.join(path, "model.pt")):
        state_dict = torch.load(os.path.join(path, "model.pt"), map_location=device)
    else:
        state_dict = load_file(os.path.join(path, "model.safetensors"), device=device)
        
    # Handle FP32/BF16 loading
    if precision != "fp8":
        # Ensure state dict matches target dtype if coming from generic loader
        # But wait, load_file(device=...) loads as is.
        # We should cast model to target dtype
        pass

    model.load_state_dict(state_dict, strict=False)
    model = model.to(dtype=dtype, device=device)
    
    if precision == "fp8":
        print("Applying FP8 quantization...")
        quantize_(model, Float8WeightOnlyConfig())

    model.eval()
    return model

def create_dummy_input(batch_size, device, dtype=torch.float32):
    # Images should be correct dtype
    images = [torch.randn(batch_size, 3, 224, 224, dtype=dtype, device=device) for _ in range(3)]
    
    # Masks: usually bool/int. standard transformer implementation often casts them 
    # but let's keep them as bool unless we see issues. 
    # State projection expects float/bf16 input
    
    img_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(3)]
    tokenized_prompt = torch.randint(0, 256000, (batch_size, 200), dtype=torch.int32, device=device)
    tokenized_prompt_mask = torch.ones(batch_size, 200, dtype=torch.bool, device=device)
    
    # CRITICAL: state must be same dtype as model weights
    state = torch.zeros(batch_size, 32, dtype=dtype, device=device) 
    
    observation = type('obj', (object,), {
        'images': {
            'base_0_rgb': images[0],
            'left_wrist_0_rgb': images[1],
            'right_wrist_0_rgb': images[2],
        },
        'image_masks': {
            'base_0_rgb': img_masks[0],
            'left_wrist_0_rgb': img_masks[1],
            'right_wrist_0_rgb': img_masks[2],
        },
        'tokenized_prompt': tokenized_prompt,
        'tokenized_prompt_mask': tokenized_prompt_mask,
        'token_ar_mask': torch.ones(batch_size, 200, dtype=torch.bool, device=device),
        'token_loss_mask': torch.ones(batch_size, 200, dtype=torch.bool, device=device),
        'state': state,
    })()
    return observation

def benchmark(model, device, name="Model", dtype=torch.float32):
    print(f"\nBenchmarking {name}...")
    obs = create_dummy_input(BATCH_SIZE, device, dtype=dtype)
    
    # Warmup
    print("Warmup...")
    # Wrap in autocast for safety as requested by Phase 5
    # "Context Manager: Wrap the forward pass in torch.autocast..."
    if device == "cuda" and dtype != torch.float32:
        autocast_context = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        from contextlib import nullcontext
        autocast_context = nullcontext()

    with torch.inference_mode(), autocast_context:
        for _ in range(NUM_WARMUP):
            _ = model.sample_actions(device, obs)
    
    # Timing
    print(f"Running {NUM_STEPS} steps...")
    latencies = []
    outputs = []
    
    with torch.inference_mode(), autocast_context:
        for _ in range(NUM_STEPS):
            start = time.time()
            out = model.sample_actions(device, obs)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            latencies.append((end - start) * 1000) # ms
            outputs.append(out)
            
    avg_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    print(f"Latency: {avg_lat:.2f} ms ± {std_lat:.2f} ms")
    
    return outputs[0], avg_lat

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results = {}

    if device == "cuda":
        # Force CuDNN off to avoid segfaults on this specific env
        torch.backends.cudnn.enabled = False
        print("WARNING: CuDNN disabled for stability.")
    
    # Force CuDNN benchmark off to avoid "unable to find engine" on some edge cases
    torch.backends.cudnn.benchmark = False
    
    # 1. FP32 (Baseline)
    try:
        model_fp32 = load_model(FP32_PATH, "fp32", device)
        out_fp32, lat_fp32 = benchmark(model_fp32, device, "FP32", dtype=torch.float32)
        results["fp32"] = {"output": out_fp32, "latency": lat_fp32}
        del model_fp32
        if device == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print(f"FP32 Benchmark failed: {e}")

    # 2. BF16
    try:
        model_bf16 = load_model(BF16_PATH, "bf16", device)
        out_bf16, lat_bf16 = benchmark(model_bf16, device, "BF16", dtype=torch.bfloat16)
        results["bf16"] = {"output": out_bf16, "latency": lat_bf16}
        
        # Compare vs FP32
        if "fp32" in results:
            diff = (out_bf16.float() - results["fp32"]["output"].float()).abs()
            mse = (diff ** 2).mean().item()
            max_err = diff.max().item()
            print(f"BF16 vs FP32 -- MSE: {mse:.6f}, Max Diff: {max_err:.6f}")
            results["bf16"]["mse"] = mse
            
        del model_bf16
        if device == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        print(f"BF16 Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

    # 3. FP8
    try:
        print("\nLoading FP8 checkpoint (saved via torch.save)...")
        
        config_path = os.path.join(FP8_PATH, "config.json")
        with open(config_path, "r") as f:
             config_dict = json.load(f)
        config = Pi0Config(
            action_dim=config_dict.get("action_dim", 32),
            action_horizon=config_dict.get("action_horizon", 10),
            paligemma_variant=config_dict.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=config_dict.get("action_expert_variant", "gemma_300m"),
            dtype="bfloat16",
        )
        model_fp8 = pi0_pytorch.PI0Pytorch(config).to(device)
        # Apply quantization structure first
        quantize_(model_fp8, Float8WeightOnlyConfig()) 
        # Now load the weights
        state_dict = torch.load(os.path.join(FP8_PATH, "model.pt"), map_location=device)
        model_fp8.load_state_dict(state_dict)
        
        # FP8 typically uses BF16/FP16 activations
        out_fp8, lat_fp8 = benchmark(model_fp8, device, "FP8", dtype=torch.bfloat16)
        results["fp8"] = {"output": out_fp8, "latency": lat_fp8}
        
        if "fp32" in results:
            diff = (out_fp8.float() - results["fp32"]["output"].float()).abs()
            mse = (diff ** 2).mean().item()
            max_err = diff.max().item()
            print(f"FP8 vs FP32 -- MSE: {mse:.6f}, Max Diff: {max_err:.6f}")
        
    except Exception as e:
        print(f"FP8 Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
