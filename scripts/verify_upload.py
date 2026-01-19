from huggingface_hub import HfApi
import sys

TOKEN = "hf_REPLACE_WITH_YOUR_TOKEN"
REPO_ID = "Tacoin/openpi-pi0.5-libero-onnx"

api = HfApi(token=TOKEN)

print(f"Verifying {REPO_ID}...")
try:
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")
    print(f"Found {len(files)} files.")
    required = ["README.md", "openpi-onnx-code.zip"]
    for r in required:
        if r in files:
            print(f"[PASS] {r} found.")
        else:
            print(f"[FAIL] {r} MISSING.")
    
    # Check for some dist files
    if any(f.startswith("final_w8a16/") for f in files):
        print("[PASS] final_w8a16 models found.")
    else:
        print("[WARN] final_w8a16 folder seems empty/missing.")

except Exception as e:
    print(f"Verification failed: {e}")
