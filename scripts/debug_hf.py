from huggingface_hub import HfApi, create_repo
import sys

TOKEN = "hf_REPLACE_WITH_YOUR_TOKEN"
REPO_ID = "OSU-AIoT-MLSys-Lab/vla"

api = HfApi(token=TOKEN)

try:
    user = api.whoami()
    print(f"Authenticated as: {user['name']} (Orgs: {[org['name'] for org in user.get('orgs', [])]})")
except Exception as e:
    print(f"Auth failed: {e}")
    sys.exit(1)

try:
    print(f"Creating/Checking repo {REPO_ID}...")
    url = create_repo(REPO_ID, token=TOKEN, private=False, exist_ok=True, repo_type="model")
    print(f"Success! Repo URL: {url}")
except Exception as e:
    print(f"Create repo failed: {e}")
