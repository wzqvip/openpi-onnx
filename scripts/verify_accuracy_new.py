
import torch
import numpy as np
import onnxruntime as ort
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.models import model as _model
import os

# --- Configuration ---
PYTORCH_CHECKPOINT = "./checkpoints/pi05_libero_pytorch_new"
CONFIG_NAME = "pi05_libero"

# Comparison Targets
ONNX_MODELS = {
    "W8A16 (New)": "./dist/final_w8a16_new/model.w8a16.onnx",
}

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    print("========================================")
    print("       Model Accuracy Verification      ")
    print("========================================")

    # --- 1. PyTorch Baseline ---
    print(f"\n[PyTorch] Loading config: {CONFIG_NAME}")
    config = _config.get_config(CONFIG_NAME)
    
    # Patch get_safe_dtype to force float32 on CPU to match export environment
    from openpi.models_pytorch import pi0_pytorch
    pi0_pytorch.get_safe_dtype = lambda target, device: torch.float32

    print(f"[PyTorch] Loading policy from: {PYTORCH_CHECKPOINT}")
    policy = policy_config.create_trained_policy(config, PYTORCH_CHECKPOINT, pytorch_device="cpu")
    model = policy._model
    model.eval()
    
    # Inputs
    batch_size = 1
    action_horizon = config.model.action_horizon
    action_dim = config.model.action_dim
    max_token_len = config.model.max_token_len
    
    # Create deterministic inputs
    torch.manual_seed(42)
    inputs_torch = {
        "base_0_rgb": torch.randn(batch_size, 3, 224, 224, dtype=torch.float32),
        "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, dtype=torch.float32),
        "right_wrist_0_rgb": torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32),
        "state": torch.randn(batch_size, 32, dtype=torch.float32),
        "tokenized_prompt": torch.randint(0, 100, (batch_size, max_token_len), dtype=torch.int32),
        "tokenized_prompt_mask": torch.ones(batch_size, max_token_len, dtype=torch.bool),
        "noise": torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32)
    }

    # PyTorch Inference
    print("[PyTorch] Running Inference...")
    images = {
        "base_0_rgb": inputs_torch["base_0_rgb"],
        "left_wrist_0_rgb": inputs_torch["left_wrist_0_rgb"],
        "right_wrist_0_rgb": inputs_torch["right_wrist_0_rgb"]
    }
    image_masks = {k: torch.ones(v.shape[:-3] if v.dim() == 4 else v.shape[:-1], dtype=torch.bool) for k, v in images.items()}
    
    obs = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=inputs_torch["state"],
        tokenized_prompt=inputs_torch["tokenized_prompt"],
        tokenized_prompt_mask=inputs_torch["tokenized_prompt_mask"]
    )
    
    with torch.no_grad():
        pytorch_output = model.sample_actions(
            device="cpu",
            observation=obs,
            noise=inputs_torch["noise"],
            num_steps=10 # Reduced steps for speed
        )
    print(f"[PyTorch] Output Shape: {pytorch_output.shape}")
    pytorch_numpy = to_numpy(pytorch_output)

    # --- 2. ONNX Verification ---
    def verify_onnx(model_path, model_name):
        print(f"\n[{model_name}] Loading session: {model_path}")
        if not os.path.exists(model_path):
            print(f"[{model_name}] File not found! Skipping.")
            return

        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        # Dynamic Input Mapping
        sess_inputs = {i.name: i for i in sess.get_inputs()}
        onnx_inputs = {}
        
        mapping = {
            "base_0_rgb": ["base_0_rgb", "observation.images.base_0_rgb"],
            "left_wrist_0_rgb": ["left_wrist_0_rgb", "observation.images.left_wrist_0_rgb"],
            "right_wrist_0_rgb": ["right_wrist_0_rgb", "observation.images.right_wrist_0_rgb"],
            "state": ["state", "observation.state"],
            "tokenized_prompt": ["tokenized_prompt", "observation.tokenized_prompt"],
            "tokenized_prompt_mask": ["tokenized_prompt_mask", "observation.tokenized_prompt_mask"],
            "noise": ["noise"]
        }
        
        for key, possible_names in mapping.items():
            found = False
            for name in possible_names:
                if name in sess_inputs:
                    onnx_inputs[name] = to_numpy(inputs_torch[key])
                    found = True
                    break
            if not found:
                 print(f"[{model_name}] WARNING: Could not find input for '{key}' in model inputs: {list(sess_inputs.keys())}")

        print(f"[{model_name}] Running Inference...")
        try:
            onnx_output = sess.run(["actions"], onnx_inputs)[0]
            
            # Metrics
            diff = np.abs(pytorch_numpy - onnx_output)
            mse = np.mean((pytorch_numpy - onnx_output)**2)
            max_err = np.max(diff)
            mean_err = np.mean(diff)
            
            print(f"[{model_name}] Accuracy Metrics vs PyTorch:")
            print(f"  > MSE:       {mse:.6f}")
            print(f"  > Max Error: {max_err:.6f}")
            print(f"  > Mean Error:{mean_err:.6f}")
            
            # Thresholds
            if mse < 1e-3:
                print(f"[{model_name}] RESULT: PASS (High Accuracy)")
            elif mse < 0.1:
                print(f"[{model_name}] RESULT: ACCEPTABLE (Quantization Noise)")
            else:
                print(f"[{model_name}] RESULT: WARNING (High Deviation)")
                
        except Exception as e:
            print(f"[{model_name}] Inference Failed: {e}")

    # Iterate models
    for name, path in ONNX_MODELS.items():
        verify_onnx(path, name)

if __name__ == "__main__":
    main()
