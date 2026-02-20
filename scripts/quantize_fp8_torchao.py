import torch
import os
import json
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch
from torchao.quantization import quantize_, Float8WeightOnlyConfig

# Constants
INPUT_DIR = "checkpoints/pi05_libero_bf16"
OUTPUT_DIR = "checkpoints/pi05_libero_fp8"

def quantize_model():
    print(f"Loading BF16 model from {INPUT_DIR}...")
    
    # Load config
    config_path = os.path.join(INPUT_DIR, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        
    config = Pi0Config(
        action_dim=config_dict.get("action_dim", 32),
        action_horizon=config_dict.get("action_horizon", 10),
        paligemma_variant=config_dict.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=config_dict.get("action_expert_variant", "gemma_300m"),
        dtype="bfloat16",
    )
    
    # Phase 2: Model Loading (Memory-Safe)
    # Initialize model config and model in BF16 directly on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing model on {device} in bfloat16...")
    model = pi0_pytorch.PI0Pytorch(config).to(dtype=torch.bfloat16, device=device)
    
    print("Loading weights...")
    state_dict = torch.load(os.path.join(INPUT_DIR, "model.pt"), map_location=device)
    
    # Fix key mismatches (JAX -> Torch artifacts)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        # Fix LayerNorm/RMSNorm keys
        if "norm.dense.weight" in k:
            new_k = k.replace("norm.dense.weight", "norm.weight")
        elif "norm.dense.bias" in k:
            new_k = k.replace("norm.dense.bias", "norm.bias")
        elif "layernorm.dense.weight" in k:
            new_k = k.replace("layernorm.dense.weight", "layernorm.weight")
        elif "layernorm.dense.bias" in k:
            new_k = k.replace("layernorm.dense.bias", "layernorm.bias")
            
        # Fix MLP keys
        if "time_mlp_" in k and "action_time_mlp_" not in k:
             new_k = new_k.replace("time_mlp_", "action_time_mlp_")
             
        new_state_dict[new_k] = v
        
    state_dict = new_state_dict
    
    # Use strict=False to allow for some minor mismatches if inevitable, 
    # but print missing keys to be sure we aren't missing big chunks.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"WARNING: Missing keys: {missing[:5]} ... ({len(missing)} total)")
    if len(unexpected) > 0:
        print(f"WARNING: Unexpected keys: {unexpected[:5]} ... ({len(unexpected)} total)")
    model.eval()
    
    # Phase 3: Applying Quantization In-Place
    print("Quantizing to FP8 (weight-only)...")
    try:
        # Option B (FP8 Weight-Only)
        quantize_(model, Float8WeightOnlyConfig())
    except Exception as e:
        print(f"Quantization failed: {e}")
        if "Triton" in str(e):
             print("Triton might be missing. Ensure triton is installed or use a different quantization backend if available.")
        raise

    print("Quantization complete.")
    print(model)

    # Phase 4: Serialization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Saving quantized model to {OUTPUT_DIR}/model.pt...")
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    
    # Save config
    print(f"Saving config to {OUTPUT_DIR}/config.json...")
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    quantize_model()
