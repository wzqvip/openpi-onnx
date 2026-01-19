from huggingface_hub import HfApi

TOKEN = "hf_REPLACE_WITH_YOUR_TOKEN"
REPO_ID = "Tacoin/openpi-pi0.5-libero-onnx"
ZIP_FILE = "openpi-onnx-code.zip"

api = HfApi(token=TOKEN)

print(f"Checking {REPO_ID} for {ZIP_FILE}...")
files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")

if ZIP_FILE in files:
    print(f"Found {ZIP_FILE} on remote. Deleting it to comply with user request...")
    api.delete_file(
        path_in_repo=ZIP_FILE,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Remove code zip as per user request"
    )
    print("Deleted.")
else:
    print(f"{ZIP_FILE} not found on remote. Good.")
