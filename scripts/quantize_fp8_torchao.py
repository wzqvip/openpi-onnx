import torch
import os
import json
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch
from torchao.quantization import quantize_, Int8WeightOnlyConfig

# Constants
INPUT_DIR = "checkpoints/pi05_libero_bf16"
OUTPUT_DIR = "checkpoints/pi05_libero_int8"

def quantize_model():
    print(f"Loading BF16 model from {INPUT_DIR}...")

    with open(os.path.join(INPUT_DIR, "config.json"), "r") as f:
        config_dict = json.load(f)

    config = Pi0Config(
        action_dim=config_dict.get("action_dim", 32),
        action_horizon=config_dict.get("action_horizon", 10),
        paligemma_variant=config_dict.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=config_dict.get("action_expert_variant", "gemma_300m"),
        dtype="bfloat16",
    )

    # Phase 2: Model Loading (Memory-Safe)
    # Initialize model in BF16 directly on CUDA/CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing model on {device} in bfloat16...")
    model = pi0_pytorch.PI0Pytorch(config).to(dtype=torch.bfloat16, device=device)

    print("Loading BF16 weights...")
    state_dict = torch.load(os.path.join(INPUT_DIR, "model.pt"), map_location=device)

    # strict=True now works since convert_to_bf16.py saves model.state_dict()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
    model.eval()

    # Phase 3: Quantize in-place with INT8 (Option A — broad GPU support)
    print("Applying INT8 weight-only quantization...")
    quantize_(model, Int8WeightOnlyConfig())
    print("Quantization complete.")

    # Phase 4: Serialization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "model.pt")
    print(f"Saving quantized model to {out_path}...")
    torch.save(model.state_dict(), out_path)

    out_cfg = os.path.join(OUTPUT_DIR, "config.json")
    config_dict["precision"] = "int8"
    with open(out_cfg, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Report size
    bf16_size = os.path.getsize(os.path.join(INPUT_DIR, "model.pt")) / 1e9
    int8_size = os.path.getsize(out_path) / 1e9
    print(f"BF16 size: {bf16_size:.2f} GB  |  INT8 size: {int8_size:.2f} GB  |  Reduction: {(1 - int8_size/bf16_size)*100:.0f}%")
    print("Done!")

if __name__ == "__main__":
    quantize_model()
