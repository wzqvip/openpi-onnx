
import numpy as np
import onnxruntime as ort
import sys
import torch
import os
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models import model as _model

# CONFIG
PYTORCH_CHECKPOINT = "./checkpoints/pi05_libero_pytorch_new"
ONNX_MODEL_PATH = "./dist/final_w4a4/model.w4a4.onnx"
CONFIG_NAME = "pi05_libero"

def main():
    print("="*40)
    print("       W4A4 Accuracy Verification")
    print("="*40)

    # 1. Load PyTorch (Baseline)
    print(f"[PyTorch] Loading policy from: {PYTORCH_CHECKPOINT}")
    config = _config.get_config(CONFIG_NAME)
    torch.compile = lambda x, **k: x # disable compile
    
    # HACK: Disable tiny model patch
    import openpi.models.gemma as _gemma_mod
    original_get_config = _gemma_mod.get_config

    policy = policy_config.create_trained_policy(config, PYTORCH_CHECKPOINT, pytorch_device="cpu")
    model = policy._model
    model.eval()
    
    # Generate Inputs
    batch_size = 1
    device = "cpu"
    dtype = torch.float32
    
    base = torch.randn(batch_size, 3, 224, 224, dtype=dtype, device=device)
    left = torch.randn(batch_size, 3, 224, 224, dtype=dtype, device=device)
    right = torch.zeros(batch_size, 3, 224, 224, dtype=dtype, device=device)
    state = torch.randn(batch_size, 32, dtype=dtype, device=device)
    prompt = torch.randint(0, 100, (batch_size, config.model.max_token_len), dtype=torch.int32, device=device)
    mask = torch.ones(batch_size, config.model.max_token_len, dtype=torch.bool, device=device)
    noise = torch.randn(batch_size, config.model.action_horizon, config.model.action_dim, dtype=dtype, device=device)
    
    # Run PyTorch
    print("[PyTorch] Running Inference...")
    images = {"base_0_rgb": base, "left_wrist_0_rgb": left, "right_wrist_0_rgb": right}
    image_masks = {k: torch.ones(v.shape[:-3], dtype=torch.bool) for k, v in images.items()}
    
    with torch.no_grad():
        obs = _model.Observation(images, image_masks, state, prompt, mask)
        pt_out = type(model).sample_actions(model, device, obs, noise, num_steps=10)
    
    pt_numpy = pt_out.numpy()
    print(f"[PyTorch] Output Mean: {pt_numpy.mean()}")

    # 2. Run ONNX W4A4
    print(f"[W4A4] Loading session: {ONNX_MODEL_PATH}")
    try:
        sess = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    ort_inputs = {
        "observation.images.base_0_rgb": base.numpy(),
        "observation.images.left_wrist_0_rgb": left.numpy(),
        "observation.images.right_wrist_0_rgb": right.numpy(),
        "observation.state": state.numpy(),
        "observation.tokenized_prompt": prompt.numpy(),
        "observation.tokenized_prompt_mask": mask.numpy(),
        "noise": noise.numpy()
    }
    
    print("[W4A4] Running Inference...")
    onnx_out = sess.run(["actions"], ort_inputs)[0]
    
    # 3. Compare
    mse = np.mean((pt_numpy - onnx_out)**2)
    max_err = np.max(np.abs(pt_numpy - onnx_out))
    
    print(f"[W4A4] MSE: {mse}")
    print(f"[W4A4] Max Error: {max_err}")
    
if __name__ == "__main__":
    main()
