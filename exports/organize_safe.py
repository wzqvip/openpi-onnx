import os
import onnx
from pathlib import Path

BASE_DIR = Path("checkpoints/pi05_libero_pytorch")
MODELS = {
    "fp32": "model.fp32.onnx",
    "fp16": "model.onnx",
    # "int8": "model.int8.onnx", # Skip pending for now or let it fail gracefully
    # "nvfp8": "model.nvfp8.onnx",
    # "int4": "model.int4.onnx",
    # "nvfp4": "model.nvfp4.onnx",
}

def organize_model(variant, filename):
    src_path = BASE_DIR / filename
    if not src_path.exists():
        print(f"Skipping {variant}: {filename} not found.")
        return

    dst_dir = BASE_DIR / variant
    dst_dir.mkdir(exist_ok=True)
    dst_path = dst_dir / "model.onnx"
    # Skip if dest exists and is newer?
    # For now just overwrite to be sure
    
    print(f"Processing {variant} from {src_path} to {dst_path}...")
    
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

if __name__ == "__main__":
    for variant, filename in MODELS.items():
        organize_model(variant, filename)
    # No cleanup
