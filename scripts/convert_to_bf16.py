import torch
import os
import json
from safetensors.torch import load_file
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch import pi0_pytorch

INPUT_DIR = "checkpoints/pi05_libero_pytorch"
OUTPUT_DIR = "checkpoints/pi05_libero_bf16"

def convert():
    print(f"Loading config from {INPUT_DIR}...")
    with open(os.path.join(INPUT_DIR, "config.json"), "r") as f:
        config_dict = json.load(f)
    config_dict["precision"] = "bfloat16"

    config = Pi0Config(
        action_dim=config_dict.get("action_dim", 32),
        action_horizon=config_dict.get("action_horizon", 10),
        paligemma_variant=config_dict.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=config_dict.get("action_expert_variant", "gemma_300m"),
        dtype="bfloat16",
    )

    # Phase 2: Load model directly in BF16 on CUDA (memory-safe)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing model on {device} in bfloat16...")
    model = pi0_pytorch.PI0Pytorch(config).to(dtype=torch.bfloat16, device=device)

    # Load FP32 safetensors and cast keys on-the-fly
    print("Loading FP32 safetensors weights...")
    state_dict = load_file(os.path.join(INPUT_DIR, "model.safetensors"), device=str(device))

    # strict=False: JAX-converted keys don't map 1:1, but the model's internal
    # load_state_dict handles the majority; unresolved keys keep default init.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    model.eval()

    # Save model.state_dict() — keys are now in PyTorch canonical format
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving config to {OUTPUT_DIR}/config.json...")
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"Saving BF16 model to {OUTPUT_DIR}/model.pt...")
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    print("Done!")

if __name__ == "__main__":
    convert()
