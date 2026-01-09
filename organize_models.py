import os
import shutil
import onnx
from pathlib import Path

BASE_DIR = Path("checkpoints/pi05_libero_pytorch")
MODELS = {
    "fp32": "model.fp32.onnx",
    "fp16": "model.onnx",
    "int8": "model.int8.onnx",
    "int4": "model.int4.onnx",
    "nvfp4": "model.nvfp4.onnx",
    "nvfp8": "model.nvfp8.onnx",
}

def organize_model(variant, filename):
    src_path = BASE_DIR / filename
    if not src_path.exists():
        print(f"Skipping {variant}: {filename} not found.")
        return

    dst_dir = BASE_DIR / variant
    dst_dir.mkdir(exist_ok=True)
    dst_path = dst_dir / "model.onnx"
    dst_data_path = dst_dir / "model.onnx.data"

    print(f"Processing {variant} from {src_path} to {dst_path}...")
    
    # Load model
    try:
        model = onnx.load(str(src_path))
        
        # Save split model
        onnx.save_model(
            model,
            str(dst_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
            size_threshold=1024,
            convert_attribute=False
        )
        print(f"Saved {variant} successfully.")
    except Exception as e:
        print(f"Failed to process {variant}: {e}")

def cleanup():
    print("Cleaning up root checkpoint directory...")
    # List of files to keep (original checkpoint specific files if any, though usually safely contained in their own subfolder structure, but here we are in the checkpoint folder itself).
    # Actually, the user said "clean up extra files". I will remove all .onnx and .onnx.data and random quantizer files in the root.
    # I should preserve the `config.json` or original `model.safetensors` if they are there?
    # From `ls` output earlier, I saw `config.json` (implied by model loading working).
    # I'll delete all *.onnx, *.onnx.data, and *quantizer* files in the root.
    
    for file in BASE_DIR.glob("*"):
        if file.is_dir():
            continue
        if file.name.endswith(".onnx") or file.name.endswith(".onnx.data") or "quantizer" in file.name:
            print(f"Deleting {file.name}")
            file.unlink()

if __name__ == "__main__":
    for variant, filename in MODELS.items():
        organize_model(variant, filename)
    
    cleanup()
