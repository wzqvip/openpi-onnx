from huggingface_hub import HfApi

TOKEN = "hf_REPLACE_WITH_YOUR_TOKEN"
REPO_ID = "Tacoin/openpi-pi0.5-libero-onnx"

api = HfApi(token=TOKEN)

print(f"Checking {REPO_ID} content...")
try:
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
    
    print(f"Total files: {len(files)}")
    
    # Check Critical Files
    has_readme = "README.md" in files
    has_zip = "openpi-onnx-code.zip" in files
    
    print(f"[Check] README.md: {'PRESENT' if has_readme else 'MISSING'}")
    print(f"[Check] openpi-onnx-code.zip: {'PRESENT (Action Needed: Delete)' if has_zip else 'ABSENT (Good)'}")
    
    # Check Model Folders
    w8a16_files = [f for f in files if f.startswith("final_w8a16/")]
    fp16_files = [f for f in files if f.startswith("final_fp16/")]
    
    print(f"[Check] final_w8a16/: {len(w8a16_files)} files")
    print(f"[Check] final_fp16/: {len(fp16_files)} files")
    
    if len(w8a16_files) > 0 and len(fp16_files) > 0:
        print("Model folders appear populate.")
    else:
        print("Warning: Model folders might be incomplete.")

except Exception as e:
    print(f"Error checking repo: {e}")
