"""Upload all 10 regenerated QLaya models, checkpoints, and model card to saipy10/qlaya."""
import os
import sys
import time

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "saipy10/qlaya"
api = HfApi()

def step(msg):
    print("\n" + "=" * 70)
    print("  " + msg)
    print("=" * 70)

def ensure_gitattributes():
    step("Ensuring .gitattributes has *.data and *.onnx.data tracked by LFS")
    try:
        p = hf_hub_download(REPO_ID, ".gitattributes")
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        content = ""
    
    needed = [
        "*.data filter=lfs diff=lfs merge=lfs -text",
        "*.onnx.data filter=lfs diff=lfs merge=lfs -text",
        "*.onnx filter=lfs diff=lfs merge=lfs -text",
        "*.safetensors filter=lfs diff=lfs merge=lfs -text",
    ]
    updated = False
    for line in needed:
        if line.split()[0] not in content:
            content += ("\n" if not content.endswith("\n") else "") + line + "\n"
            updated = True
    
    if updated:
        tmp_path = "gitattributes_tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=".gitattributes",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Update .gitattributes to track *.data and *.onnx.data with LFS",
        )
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print("  ✓ Updated .gitattributes on repository")
    else:
        print("  ✓ .gitattributes is already properly configured")

def upload_file_safe(local_path, repo_path, existing_files=None):
    if not os.path.exists(local_path):
        print(f"  [MISSING] {local_path}")
        return False
    if existing_files and repo_path in existing_files:
        print(f"  [EXISTS] {repo_path} already present in {REPO_ID}, skipping.")
        return True
    sz = os.path.getsize(local_path) / (1024 * 1024)
    print(f"\nUploading: {repo_path} ({sz:.1f} MB)...")
    t0 = time.time()
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Upload {repo_path}",
        )
        print(f"  ✓ Uploaded {repo_path} in {time.time() - t0:.1f}s")
        return True
    except Exception as e:
        print(f"  ✗ Error uploading {repo_path}: {e}")
        return False

def main():
    step(f"Starting Hugging Face Upload to {REPO_ID}")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    ensure_gitattributes()
    
    existing = set(api.list_repo_files(repo_id=REPO_ID))
    print(f"Found {len(existing)} existing files in repo.")

    # 1. Model Card & Configs (always re-upload README to keep docs fresh)
    upload_file_safe("F:/Project/laya/HF_README.md", "README.md")
    upload_file_safe("F:/Project/laya/models/base/rl_agent_config.json", "rl_agent_config.json", existing)
    upload_file_safe("F:/Project/laya/models/base/encoder/config.json", "encoder/config.json", existing)
    upload_file_safe("F:/Project/laya/models/base/tokenizer/tokenizer.json", "tokenizer/tokenizer.json", existing)
    upload_file_safe("F:/Project/laya/models/base/tokenizer/tokenizer_config.json", "tokenizer/tokenizer_config.json", existing)
    upload_file_safe("F:/Project/laya/benchmark_results.json", "benchmark_results.json")

    # 2. Key Production ONNX Models (Top Picks)
    step("Uploading Production ONNX Models")
    upload_file_safe("F:/Project/laya/models/quantized/qlaya.int8.onnx", "qlaya.int8.onnx", existing)
    upload_file_safe("F:/Project/laya/models/distilled/onnx/distil_qlaya_6l.int8.onnx", "distil_qlaya_6l.int8.onnx", existing)
    upload_file_safe("F:/Project/laya/models/distilled/onnx/distil_qlaya_6l.int4.onnx", "distil_qlaya_6l.int4.onnx", existing)
    upload_file_safe("F:/Project/laya/models/distilled/onnx/distil_qlaya_14l.int8.onnx", "distil_qlaya_14l.int8.onnx", existing)

    # 3. Quantized Teacher & Half-Precision Models
    step("Uploading Balanced & Quantized Teacher Variants")
    upload_file_safe("F:/Project/laya/models/quantized/qlaya.fp16.onnx", "qlaya.fp16.onnx", existing)
    upload_file_safe("F:/Project/laya/models/quantized/qlaya.int4_b32.onnx", "qlaya.int4_b32.onnx", existing)
    upload_file_safe("F:/Project/laya/models/quantized/qlaya.int4_b64.onnx", "qlaya.int4_b64.onnx", existing)

    # 4. FP32 ONNX Models
    step("Uploading FP32 ONNX Baselines")
    upload_file_safe("F:/Project/laya/models/distilled/onnx/distil_qlaya_6l.fp32.onnx", "distil_qlaya_6l.fp32.onnx", existing)
    upload_file_safe("F:/Project/laya/models/distilled/onnx/distil_qlaya_6l.fp32.onnx.data", "distil_qlaya_6l.fp32.onnx.data", existing)
    upload_file_safe("F:/Project/laya/models/distilled/onnx/distil_qlaya_14l.fp32.onnx", "distil_qlaya_14l.fp32.onnx", existing)
    upload_file_safe("F:/Project/laya/models/distilled/onnx/distil_qlaya_14l.fp32.onnx.data", "distil_qlaya_14l.fp32.onnx.data", existing)
    upload_file_safe("F:/Project/laya/models/quantized/qlaya.fp32.onnx", "qlaya.fp32.onnx", existing)
    upload_file_safe("F:/Project/laya/models/quantized/qlaya.fp32.onnx.data", "qlaya.fp32.onnx.data", existing)

    # 5. PyTorch Checkpoints & Subfolder configs
    step("Uploading PyTorch Checkpoints and Student Tokenizers")
    upload_file_safe("F:/Project/laya/models/base/model.safetensors", "model.safetensors", existing)
    upload_file_safe("F:/Project/laya/models/distilled/distil_qlaya_14l/model.safetensors", "distil_qlaya_14l/model.safetensors", existing)
    upload_file_safe("F:/Project/laya/models/distilled/distil_qlaya_14l/rl_agent_config.json", "distil_qlaya_14l/rl_agent_config.json", existing)
    upload_file_safe("F:/Project/laya/models/distilled/distil_qlaya_14l/tokenizer/tokenizer.json", "distil_qlaya_14l/tokenizer/tokenizer.json", existing)
    upload_file_safe("F:/Project/laya/models/distilled/distil_qlaya_14l/tokenizer/tokenizer_config.json", "distil_qlaya_14l/tokenizer/tokenizer_config.json", existing)

    upload_file_safe("F:/Project/laya/models/distilled/distil_qlaya_6l/model.safetensors", "distil_qlaya_6l/model.safetensors", existing)
    upload_file_safe("F:/Project/laya/models/distilled/distil_qlaya_6l/rl_agent_config.json", "distil_qlaya_6l/rl_agent_config.json", existing)
    upload_file_safe("F:/Project/laya/models/distilled/distil_qlaya_6l/tokenizer/tokenizer.json", "distil_qlaya_6l/tokenizer/tokenizer.json", existing)
    upload_file_safe("F:/Project/laya/models/distilled/distil_qlaya_6l/tokenizer/tokenizer_config.json", "distil_qlaya_6l/tokenizer/tokenizer_config.json", existing)

    step("All Models and Checkpoints Uploaded to saipy10/qlaya Successfully!")

if __name__ == "__main__":
    main()

