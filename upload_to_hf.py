import os
from huggingface_hub import HfApi, create_repo
import shutil

# Config
TOKEN = ""
REPO_ID = "Tacoin/openpi-pi0.5-libero-onnx"
SOURCE_DIR = "dist"

def main():
    print(f"Preparing upload for {REPO_ID} from {SOURCE_DIR}...")
    api = HfApi(token=TOKEN)
    
    # Verify login
    try:
        user = api.whoami()
        print(f"Authenticated as: {user['name']}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        return

    # Create Repo
    try:
        create_repo(REPO_ID, token=TOKEN, private=False, exist_ok=True)
        print(f"Repository {REPO_ID} is ready.")
    except Exception as e:
        print(f"Repo creation notice: {e}")

    # Create README (Model Card)
    readme_content = """---
license: mit
tags:
- onnx
- robotics
- openpi
- jetson-thor
- quantization
- w8a16
platform: onnx
---

# OpenPi Pi0.5 Libero ONNX Models for Jetson Thor

This repository contains optimized ONNX export variants of the OpenPi Pi0.5 Libero model, specifically tuned for the **NVIDIA Jetson Thor** (Blackwell) platform.

## Benchmark Report (All Precisions)

We evaluated the model across various precisions on Jetson Thor. 

| Variant | Precision Label | Latency (ms) | Throughput (QPS) | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **W8A8 (Sim)** | **WINT8AINT8** | **128.37** | **8.38** | *Benchmark Only* | **Fastest**. Implicit INT8 measurement. |
| **W8A16 (QDQ)** | **WINT8AFP16** | **181.81** | **6.37** | **Available** | **Recommended**. Verified high accuracy and performance. |
| **INT4 (Sim)** | **WINT4AFP16** | 183.15 | 6.33 | *Benchmark Only* | Parity speed with W8A16. |
| **FP16** | **WFP16AFP16** | 184.54 | 6.26 | **Available** | Baseline high-precision export. |
| **BF16** | **WBF16ABF16** | 190.21 | 6.14 | *Benchmark Only* | Parity with FP16. |
| **FP8 (Sim)** | **WFP8AFP16** | 310.90 | 3.63 | *Benchmark Only* | Slower without explicit QAT optimization. |

### Accuracy Verification (MSE vs PyTorch)
| Variant | MSE | Max Error | Result |
| :--- | :--- | :--- | :--- |
| **W8A16 (QDQ)** | **0.0061** | **0.203** | **PASS** (Identical to FP16) |
| **FP16** | 0.0061 | 0.205 | PASS |

## Available Models

The following verified models are available for download in this repository:

| Config | Path | Size | Description |
| :--- | :--- | :--- | :--- |
| **W8A16 (QDQ)** | `final_w8a16/` | ~12GB | **Recommended Deployment Model**. INT8 Weights, FP16 Activations. |
| **FP16** | `final_fp16/` | ~12GB | Baseline FP16 export. |

## Usage
These models are "flat" ONNX files with external data. They are designed for **TensorRT** compilation.

### Deployment Guide
See `thor_deployment.md` in this repository for detailed instructions.

## Directory Structure
```
.
├── final_w8a16/        # Weight-Only INT8 Quantized Model (QDQ format)
│   ├── model.w8a16.onnx
│   └── model.w8a16.onnx.data
├── final_fp16/         # FP16/FP32 Baseline Model
│   ├── model.fp32.onnx
│   └── model.fp32.onnx.data
├── thor_deployment.md  # Detailed Documentation
└── README.md
```
"""
    
    with open(os.path.join(SOURCE_DIR, "README.md"), "w") as f:
        f.write(readme_content)
    print("Updated README.md in dist/")

    # Upload
    print("Starting upload...")
    api.upload_folder(
        folder_path=SOURCE_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        path_in_repo=".",
    )
    print("Upload successfully completed!")

if __name__ == "__main__":
    main()
