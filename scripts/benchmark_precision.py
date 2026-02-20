import torch
import time
import os
import json
import numpy as np
from contextlib import nullcontext
from safetensors.torch import load_file
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch
from torchao.quantization import quantize_, Int8WeightOnlyConfig

# Constants
FP32_PATH  = "checkpoints/pi05_libero_pytorch"
BF16_PATH  = "checkpoints/pi05_libero_bf16"
INT8_PATH  = "checkpoints/pi05_libero_int8"

NUM_WARMUP  = 5
NUM_STEPS   = 20
BATCH_SIZE  = 1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(path, precision="fp32", device="cuda"):
    print(f"Loading {precision} model from {path}...")
    with open(os.path.join(path, "config.json"), "r") as f:
        config_dict = json.load(f)

    model_dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    config_dtype = "float32" if precision == "fp32" else "bfloat16"

    config = Pi0Config(
        action_dim=config_dict.get("action_dim", 32),
        action_horizon=config_dict.get("action_horizon", 10),
        paligemma_variant=config_dict.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=config_dict.get("action_expert_variant", "gemma_300m"),
        dtype=config_dtype,
    )

    model = pi0_pytorch.PI0Pytorch(config).to(dtype=model_dtype, device=device)

    # Load weights ─ prefer .pt (model.state_dict), fall back to safetensors
    pt_path = os.path.join(path, "model.pt")
    st_path = os.path.join(path, "model.safetensors")
    if os.path.exists(pt_path):
        state_dict = torch.load(pt_path, map_location=device)
    else:
        state_dict = load_file(st_path, device=str(device))

    # For INT8: apply quantize_() on a freshly-initialized model FIRST,
    # so all nn.Linear params become AffineQuantizedTensor, then load the
    # saved quantized state dict in one shot.
    if precision == "int8":
        print("  Applying INT8 quantization structure (before loading weights)...")
        quantize_(model, Int8WeightOnlyConfig())

    # Load the checkpoint — types now match for all precision variants
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Dummy inputs (Phase 5: all float tensors cast to model dtype)
# ---------------------------------------------------------------------------

def create_dummy_input(batch_size, device, dtype=torch.float32):
    """All floating-point tensors use `dtype`. Bool/int tensors left as-is."""
    images = [
        torch.randn(batch_size, 3, 224, 224, dtype=dtype, device=device)
        for _ in range(3)
    ]
    img_masks = [torch.ones(batch_size, dtype=torch.bool, device=device) for _ in range(3)]

    return type("Observation", (), {
        "images": {
            "base_0_rgb":       images[0],
            "left_wrist_0_rgb": images[1],
            "right_wrist_0_rgb": images[2],
        },
        "image_masks": {
            "base_0_rgb":       img_masks[0],
            "left_wrist_0_rgb": img_masks[1],
            "right_wrist_0_rgb": img_masks[2],
        },
        "tokenized_prompt":      torch.randint(0, 256000, (batch_size, 200), dtype=torch.int32, device=device),
        "tokenized_prompt_mask": torch.ones(batch_size, 200, dtype=torch.bool, device=device),
        "token_ar_mask":         torch.ones(batch_size, 200, dtype=torch.bool, device=device),
        "token_loss_mask":       torch.ones(batch_size, 200, dtype=torch.bool, device=device),
        "state":                 torch.zeros(batch_size, 32, dtype=dtype, device=device),
    })()


# ---------------------------------------------------------------------------
# Benchmark (Phase 5: autocast wraps every forward pass)
# ---------------------------------------------------------------------------

def benchmark(model, device, name="Model", dtype=torch.float32):
    print(f"\nBenchmarking {name}...")
    obs = create_dummy_input(BATCH_SIZE, device, dtype=dtype)

    # Autocast for BF16 / INT8; identity context for FP32
    if device == "cuda" and dtype != torch.float32:
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    print("Warmup...")
    with torch.inference_mode(), autocast_ctx:
        for _ in range(NUM_WARMUP):
            _ = model.sample_actions(device, obs)

    print(f"Running {NUM_STEPS} steps...")
    latencies, outputs = [], []
    with torch.inference_mode(), autocast_ctx:
        for _ in range(NUM_STEPS):
            start = time.time()
            out = model.sample_actions(device, obs)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000)
            outputs.append(out)

    avg_lat = np.mean(latencies)
    print(f"Latency: {avg_lat:.2f} ms ± {np.std(latencies):.2f} ms")
    return outputs[0], avg_lat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        torch.backends.cudnn.enabled = False   # stability on Thor GPU
        torch.backends.cudnn.benchmark = False
        print("WARNING: CuDNN disabled for stability.")

    results = {}

    # 1. FP32 — baseline
    try:
        m = load_model(FP32_PATH, "fp32", device)
        out, lat = benchmark(m, device, "FP32", dtype=torch.float32)
        results["fp32"] = {"output": out, "latency": lat}
        del m;  torch.cuda.empty_cache() if device == "cuda" else None
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"FP32 failed: {e}")

    # 2. BF16
    try:
        m = load_model(BF16_PATH, "bf16", device)
        out, lat = benchmark(m, device, "BF16", dtype=torch.bfloat16)
        results["bf16"] = {"output": out, "latency": lat}
        if "fp32" in results:
            diff = (out.float() - results["fp32"]["output"].float()).abs()
            mse  = (diff ** 2).mean().item()
            print(f"BF16 vs FP32 — MSE: {mse:.6f}, Max Diff: {diff.max().item():.6f}")
            results["bf16"]["mse"] = mse
        del m;  torch.cuda.empty_cache() if device == "cuda" else None
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"BF16 failed: {e}")

    # 3. INT8
    try:
        m = load_model(INT8_PATH, "int8", device)
        out, lat = benchmark(m, device, "INT8", dtype=torch.bfloat16)
        results["int8"] = {"output": out, "latency": lat}
        if "fp32" in results:
            diff = (out.float() - results["fp32"]["output"].float()).abs()
            mse  = (diff ** 2).mean().item()
            print(f"INT8 vs FP32 — MSE: {mse:.6f}, Max Diff: {diff.max().item():.6f}")
            results["int8"]["mse"] = mse
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"INT8 failed: {e}")

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for prec, v in results.items():
        mse_str = f"  |  MSE vs FP32: {v['mse']:.6f}" if "mse" in v else ""
        print(f"  {prec.upper():5s}  {v['latency']:7.1f} ms{mse_str}")

if __name__ == "__main__":
    main()
