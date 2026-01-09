import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path

# User provided token
TOKEN = "hf_REPLACED_TOKEN"
CHECKPOINT_DIR = "checkpoints/pi05_libero_pytorch"
REPO_NAME = "openpi-pi05-libero-thor-onnx"

def main():
    api = HfApi(token=TOKEN)
    
    # 1. Login/Identify
    user_info = api.whoami()
    username = user_info['name']
    print(f"Logged in as: {username}")
    
    repo_id = f"{username}/{REPO_NAME}"
    print(f"Target Repo: {repo_id}")
    
    # 2. Create Repo
    try:
        create_repo(repo_id, token=TOKEN, private=False, exist_ok=True)
        print(f"Repository {repo_id} ready.")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # 3. Create README.md (Model Card)
    readme_content = f"""---
license: mit
tags:
- onnx
- robotics
- openpi
- jetson-thor
- quantization
---

# OpenPi Pi0.5 Libero ONNX Exports for Jetson Thor

This repository contains ONNX exports of the OpenPi Pi0.5 Libero model, optimized for Nvidia Jetson Thor (Blackwell).

## Models Included

All models are exported with their external data files consolidated.

| Format | Path | Size | Description |
| :--- | :--- | :--- | :--- |
| **FP32** | `fp32/` | ~13GB | Full precision FP32 export. |
| **FP16** | `fp16/` | ~6.5GB | True Float32 -> Float16 converted model. Recommended for general GPU use. |
| **NVFP8**| `nvfp8/`| ~13GB | **NVIDIA FP8** Quantized (FakeQuant). Optimized for Blackwell Transformer Engine. |
| **NVFP4**| `nvfp4/`| ~13GB | **NVIDIA FP4** Quantized (FakeQuant). Optimized for Blackwell Transformer Engine. |
| **INT8** | `int8/` | ~13GB | Standard INT8 Quantized (FakeQuant). |
| **INT4** | `int4/` | ~13GB | INT4 Blockwise Quantized (FakeQuant). |

**Note**: The quantized models (INT8/4, NVFP8/4) are exported using `modelopt` and currently contain FakeQuant nodes with Float32 tensors (hence the ~13GB size). They are intended to be loaded into TensorRT which will fuse these nodes into actual low-precision kernels on supported hardware (Jetson Thor).

## Directory Structure

```
.
├── fp32/
│   ├── model.onnx
│   └── model.onnx.data
├── fp16/
│   ├── model.onnx
│   └── model.onnx.data
├── nvfp8/
│   ├── model.onnx
│   └── model.onnx.data
├── nvfp4/
│   ├── model.onnx
│   └── model.onnx.data
├── int8/
│   ├── model.onnx
│   └── model.onnx.data
└── int4/
    ├── model.onnx
    └── model.onnx.data
```

## Usage

These models are designed to be consumed by **TensorRT** or **ONNX Runtime** with TensorRT Execution Provider on Nvidia Jetson Thor.

"""
    
    readme_path = os.path.join(CHECKPOINT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print("Created README.md")

    # 4. Upload Folder
    print("Uploading models... This may take a while.")
    api.upload_folder(
        folder_path=CHECKPOINT_DIR,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=".",
        ignore_patterns=["*.safetensors", "config.json", "*.pth", "trt__*"] # Upload mainly the organized folders
    )
    print("Upload complete!")

if __name__ == "__main__":
    main()
