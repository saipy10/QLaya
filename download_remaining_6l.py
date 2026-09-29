import os
import sys
import time
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

TARGET_DIR = "F:/Project/laya/models/distilled"

api = KaggleApi()
api.authenticate()

print("Fetching file list from Kaggle...")
with api.build_kaggle_client() as kaggle:
    req = ApiListKernelSessionOutputRequest()
    req.user_name = "sai10py"
    req.kernel_slug = "qlaya-distillation-and-quantization"
    res = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
    files = res.files or []

for f in files:
    if "distil_qlaya_6l" not in f.file_name:
        continue
        
    rel_path = f.file_name
    if rel_path.startswith("models/"):
        clean_rel = rel_path[len("models/"):]
    else:
        clean_rel = rel_path
        
    out_path = os.path.join(TARGET_DIR, clean_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Get remote size
    head = requests.get(f.url, stream=True)
    total_size = int(head.headers.get("content-length", 0))
    head.close()
    
    if os.path.exists(out_path) and os.path.getsize(out_path) == total_size and total_size > 0:
        print(f"[ALREADY DONE] {clean_rel} ({total_size / (1024*1024):.1f} MB)")
        continue
        
    print(f"\nDownloading: {clean_rel} (Total: {total_size / (1024*1024):.1f} MB)")
    
    max_retries = 10
    for attempt in range(max_retries):
        try:
            curr_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
            if curr_size >= total_size and total_size > 0:
                break
                
            headers = {}
            if curr_size > 0:
                headers["Range"] = f"bytes={curr_size}-"
                mode = "ab"
                print(f"  Resuming from {curr_size / (1024*1024):.1f} MB...")
            else:
                mode = "wb"
                
            resp = requests.get(f.url, headers=headers, stream=True, timeout=30)
            if resp.status_code not in (200, 206):
                mode = "wb"
                curr_size = 0
                resp = requests.get(f.url, stream=True, timeout=30)
                
            with open(out_path, mode) as out_f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out_f.write(chunk)
                        curr_size += len(chunk)
                        pct = (curr_size / total_size) * 100 if total_size > 0 else 0
                        print(f"\r  Progress: {curr_size/(1024*1024):.1f} / {total_size/(1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
            
            if curr_size >= total_size:
                print(f"\n  [OK] Completed {clean_rel} ({curr_size/(1024*1024):.1f} MB)")
                break
        except Exception as e:
            print(f"\n  Network retry {attempt+1}/{max_retries}: {e}")
            time.sleep(3)

print("\n" + "=" * 60)
print("All 6L student files downloaded successfully!")
print("=" * 60)
