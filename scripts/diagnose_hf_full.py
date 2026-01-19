from huggingface_hub import HfApi, create_repo
import sys

TOKEN = "hf_REPLACE_WITH_YOUR_TOKEN"
ORG = "OSU-AIoT-MLSys-Lab"
MY_USER = "Tacoin"

api = HfApi(token=TOKEN)

print(f"Checking access for {MY_USER}...")

# 1. List Org Repos
try:
    models = api.list_models(author=ORG)
    print(f"Visible models in {ORG}:")
    found = False
    for m in models:
        print(f" - {m.modelId}")
        if m.modelId == f"{ORG}/vla":
            found = True
    
    if found:
        print(f"!!! Repository {ORG}/vla EXISTS !!!")
    else:
        print(f"Repository {ORG}/vla NOT FOUND in list.")

except Exception as e:
    print(f"Failed to list models: {e}")

# 2. Check Write Token Capability (try to create a *private* repo in own namespace)
test_repo = f"{MY_USER}/write_test_123"
try:
    print(f"Testing WRITE permission by creating {test_repo}...")
    api.create_repo(test_repo, repo_type="model", exist_ok=True, private=True)
    print("WRITE capability CONFIRMED (User namespace).")
    # clean up
    api.delete_repo(test_repo, repo_type="model")
    print("Test repo deleted.")
except Exception as e:
    print(f"WRITE capability FAILED: {e}")
    if "403" in str(e):
        print("Token likely Read-only or restricted scope.")

