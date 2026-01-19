from huggingface_hub import HfApi
import collections

TOKEN = "hf_REPLACE_WITH_YOUR_TOKEN"
REPO_ID = "Tacoin/openpi-pi0.5-libero-onnx"

api = HfApi(token=TOKEN)

print(f"Analyzing storage usage for {REPO_ID}...")
files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")

# We can't easily get exact sizes from list_repo_files without looking at detailed objects,
# but we can look at the *structure* which often explains it (e.g., duplicated checkpoints).
# For exact sizes, we'd need model_info which is heavier, but let's look at counts first.

folder_counts = collections.defaultdict(int)
for f in files:
    parts = f.split("/")
    if len(parts) > 1:
        folder_counts[parts[0]] += 1
    else:
        folder_counts["root"] += 1

print("\nFile Counts by Top-Level Folder:")
for folder, count in folder_counts.items():
    print(f"- {folder}: {count} files")

# Check for 'checkpoints' which usually has heavy pth files
checkpoints = [f for f in files if f.startswith("checkpoints/")]
print(f"\nCheckpoints folder has {len(checkpoints)} files.")
if len(checkpoints) > 0:
    print("Sample checkpoint files:")
    for i, f in enumerate(checkpoints[:5]):
        print(f"  {f}")
