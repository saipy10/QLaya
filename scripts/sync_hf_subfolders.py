"""Upload subfolder configs and tokenizers to saipy10/qlaya on Hugging Face.

This ensures compatibility with all clients (current and legacy) requesting:
- root tokenizer.json
- distil_qlaya_14l/tokenizer.json
- distil_qlaya_6l/tokenizer.json
- qlaya-int8/rl_agent_config.json and tokenizer.json
- other variant subfolders
"""
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv("F:/Project/laya/.env")
token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN not found in environment or .env file.")

REPO_ID = "saipy10/qlaya"
api = HfApi(token=token)

BASE_DIR = "F:/Project/laya"
BASE_CONFIG = f"{BASE_DIR}/models/base/rl_agent_config.json"
BASE_TOK_JSON = f"{BASE_DIR}/models/base/tokenizer/tokenizer.json"
BASE_TOK_CFG = f"{BASE_DIR}/models/base/tokenizer/tokenizer_config.json"

D14_CONFIG = f"{BASE_DIR}/models/distilled/distil_qlaya_14l/rl_agent_config.json"
D14_TOK_JSON = f"{BASE_DIR}/models/distilled/distil_qlaya_14l/tokenizer/tokenizer.json"
D14_TOK_CFG = f"{BASE_DIR}/models/distilled/distil_qlaya_14l/tokenizer/tokenizer_config.json"

D6_CONFIG = f"{BASE_DIR}/models/distilled/distil_qlaya_6l/rl_agent_config.json"
D6_TOK_JSON = f"{BASE_DIR}/models/distilled/distil_qlaya_6l/tokenizer/tokenizer.json"
D6_TOK_CFG = f"{BASE_DIR}/models/distilled/distil_qlaya_6l/tokenizer/tokenizer_config.json"

def upload(local_path, repo_path, existing):
    if not os.path.exists(local_path):
        print(f"  [MISSING LOCAL] {local_path}")
        return
    if repo_path in existing:
        print(f"  [EXISTS] {repo_path}")
        return
    print(f"  Uploading: {repo_path}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Add {repo_path} for client compatibility",
    )
    print(f"  ✓ Uploaded: {repo_path}")
    existing.add(repo_path)

def main():
    print(f"Connecting to Hugging Face repo: {REPO_ID}")
    existing = set(api.list_repo_files(repo_id=REPO_ID))
    print(f"Found {len(existing)} existing files in {REPO_ID}.")

    # 1. Root level tokenizer.json
    print("\n--- Ensuring root level tokenizer.json ---")
    upload(BASE_TOK_JSON, "tokenizer.json", existing)

    # 2. Student models: root tokenizer.json in subfolders
    print("\n--- Ensuring student subfolders have tokenizer.json ---")
    upload(D14_TOK_JSON, "distil_qlaya_14l/tokenizer.json", existing)
    upload(D6_TOK_JSON, "distil_qlaya_6l/tokenizer.json", existing)

    # 3. Teacher variant subfolders (for legacy & subfolder-based loaders)
    teacher_subfolders = [
        "qlaya-fp32",
        "qlaya-fp16",
        "qlaya-int8",
        "qlaya-int4-b32",
        "qlaya-int4-b64",
    ]
    print("\n--- Uploading teacher variant subfolders ---")
    for sub in teacher_subfolders:
        upload(BASE_CONFIG, f"{sub}/rl_agent_config.json", existing)
        upload(BASE_TOK_JSON, f"{sub}/tokenizer.json", existing)
        upload(BASE_TOK_JSON, f"{sub}/tokenizer/tokenizer.json", existing)
        upload(BASE_TOK_CFG, f"{sub}/tokenizer/tokenizer_config.json", existing)

    # 4. 14L student variant subfolders
    print("\n--- Uploading 14L variant subfolders ---")
    for sub in ["qlaya-distil-14l-fp32", "qlaya-distil-14l-int8"]:
        upload(D14_CONFIG, f"{sub}/rl_agent_config.json", existing)
        upload(D14_TOK_JSON, f"{sub}/tokenizer.json", existing)
        upload(D14_TOK_JSON, f"{sub}/tokenizer/tokenizer.json", existing)
        upload(D14_TOK_CFG, f"{sub}/tokenizer/tokenizer_config.json", existing)

    # 5. 6L student variant subfolders
    print("\n--- Uploading 6L variant subfolders ---")
    for sub in ["qlaya-distil-6l-fp32", "qlaya-distil-6l-int8", "qlaya-distil-6l-int4"]:
        upload(D6_CONFIG, f"{sub}/rl_agent_config.json", existing)
        upload(D6_TOK_JSON, f"{sub}/tokenizer.json", existing)
        upload(D6_TOK_JSON, f"{sub}/tokenizer/tokenizer.json", existing)
        upload(D6_TOK_CFG, f"{sub}/tokenizer/tokenizer_config.json", existing)

    # 6. Re-upload updated Model Card README
    print("\n--- Updating HF README.md ---")
    api.upload_file(
        path_or_fileobj=f"{BASE_DIR}/HF_README.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Update README.md with Node/TS and Python quickstarts",
    )
    print("  ✓ Updated README.md on saipy10/qlaya")

    print("\nAll Hugging Face sync operations completed successfully!")

if __name__ == "__main__":
    main()
