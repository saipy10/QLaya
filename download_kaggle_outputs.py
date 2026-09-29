import os
import sys
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

TARGET_DIR = "F:/Project/laya/models/distilled"
os.makedirs(TARGET_DIR, exist_ok=True)

api = KaggleApi()
api.authenticate()

print("Connecting to Kaggle kernel session output...")
with api.build_kaggle_client() as kaggle:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = "sai10py"
    req.kernel_slug = "qlaya-distillation-and-quantization"
    req.page_size = 50
    res = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
    
    files = res.files or []
    print(f"Found {len(files)} output files on Kaggle.")
    
    for f in files:
        rel_path = f.file_name
        if rel_path.startswith("models/"):
            clean_rel = rel_path[len("models/"):]
        else:
            clean_rel = rel_path
            
        out_path = os.path.join(TARGET_DIR, clean_rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Check if already downloaded
        head_resp = requests.head(f.url)
        remote_sz = int(head_resp.headers.get("content-length", 0))
        
        if os.path.exists(out_path) and remote_sz > 0 and os.path.getsize(out_path) == remote_sz:
            sz_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"[SKIP ALREADY DOWNLOADED] {clean_rel} ({sz_mb:.1f} MB)")
            continue
            
        print(f"\nDownloading: {rel_path} -> {out_path}")
        resp = requests.get(f.url, stream=True)
        resp.raise_for_status()
        
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        
        with open(out_path, "wb") as out_f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        print(f"\r  Progress: {mb:.1f} / {total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Downloaded: {mb:.1f} MB", end="", flush=True)
                        
        final_sz = os.path.getsize(out_path) / (1024 * 1024)
        print(f"\n  [OK] Finished: {clean_rel} ({final_sz:.1f} MB)")

print("\n" + "=" * 60)
print(f"All files successfully downloaded to {TARGET_DIR}!")
print("=" * 60)
