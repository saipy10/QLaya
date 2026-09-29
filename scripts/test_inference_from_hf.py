"""Inference Test Program for QLaya models loaded directly from Hugging Face Hub (saipy10/qlaya).

Tests inference across all 10 models:
  1. qlaya.int8.onnx            (Top Production Teacher INT8 - 571.9 MB)
  2. distil_qlaya_6l.int8.onnx  (Ultra-Fast Edge 6L INT8 - 311.4 MB)
  3. distil_qlaya_6l.int4.onnx  (Ultra-Small Storage 6L INT4 - 278.4 MB)
  4. distil_qlaya_14l.int8.onnx (High-Speed Production 14L INT8 - 406.2 MB)
  5. qlaya.fp16.onnx            (Balanced Teacher FP16 - 806.6 MB)
  6. qlaya.int4_b32.onnx        (Slow CPU Teacher INT4 block-32 - 441.2 MB)
  7. qlaya.int4_b64.onnx        (Degraded Accuracy Teacher INT4 block-64 - 419.6 MB)
  8. distil_qlaya_6l.fp32.onnx  (Compact Student 6L FP32 - 578.5 MB)
  9. distil_qlaya_14l.fp32.onnx (Intermediate Student 14L FP32 - 952.6 MB)
 10. qlaya.fp32.onnx            (Original Baseline FP32 Teacher - 1607 MB)

Usage:
  python scripts/test_inference_from_hf.py
  python scripts/test_inference_from_hf.py --model distil_qlaya_6l.int8.onnx
"""
import argparse
import os
import sys
import time
from functools import partial
from typing import Dict, Any

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

print = partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import hf_hub_download
from qlaya.onnx_agent import ONNXAgent

REPO_ID = "saipy10/qlaya"

# Catalog of all 10 models
ALL_MODELS = [
    {
        "name": "QLaya Top Production (Teacher INT8)",
        "file": "qlaya.int8.onnx",
        "has_data": False,
        "recommended": True,
    },
    {
        "name": "QLaya Ultra-Fast Edge (Distil-6L INT8)",
        "file": "distil_qlaya_6l.int8.onnx",
        "has_data": False,
        "recommended": True,
    },
    {
        "name": "QLaya Ultra-Small Storage (Distil-6L INT4)",
        "file": "distil_qlaya_6l.int4.onnx",
        "has_data": False,
        "recommended": True,
    },
    {
        "name": "QLaya High-Speed Production (Distil-14L INT8)",
        "file": "distil_qlaya_14l.int8.onnx",
        "has_data": False,
        "recommended": False,
    },
    {
        "name": "QLaya Balanced (Teacher FP16)",
        "file": "qlaya.fp16.onnx",
        "has_data": False,
        "recommended": False,
    },
    {
        "name": "QLaya Slow CPU (Teacher INT4-b32)",
        "file": "qlaya.int4_b32.onnx",
        "has_data": False,
        "recommended": False,
    },
    {
        "name": "QLaya Degraded Accuracy (Teacher INT4-b64)",
        "file": "qlaya.int4_b64.onnx",
        "has_data": False,
        "recommended": False,
    },
    {
        "name": "QLaya Compact Student (Distil-6L FP32)",
        "file": "distil_qlaya_6l.fp32.onnx",
        "has_data": True,
        "recommended": False,
    },
    {
        "name": "QLaya Intermediate Student (Distil-14L FP32)",
        "file": "distil_qlaya_14l.fp32.onnx",
        "has_data": True,
        "recommended": False,
    },
    {
        "name": "QLaya Original Baseline (Teacher FP32)",
        "file": "qlaya.fp32.onnx",
        "has_data": True,
        "recommended": False,
    },
]

# Sample enterprise customer support ticket state
SAMPLE_STATE = (
    "Customer Support Ticket #89211\n"
    "Subject: Urgent: Unauthorized account changes and duplicate billing charges\n"
    "Message: I noticed two charges of $499 on my corporate credit card that I did not authorize, "
    "and someone changed the admin email on our team subscription. We are locked out of our workspace. "
    "Please reverse these charges immediately and restore access to our legitimate administrator."
)

SAMPLE_QUESTIONS = {
    "ticket_category": {
        "type": "choice",
        "instructions": "Route this ticket to the primary team responsible for resolving it.",
        "criteria": {
            "billing": "Invoice disputes, credit card charges, refund requests",
            "security": "Unauthorized access, hijacked accounts, credential issues",
            "general_support": "How-to questions, minor configuration, product feedback",
            "bug_report": "System errors, unexpected crashes, application bugs",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "Determine the severity and operational urgency of this ticket.",
        "criteria": [
            "low: non-urgent inquiry",
            "medium: standard business query",
            "high: important feature broken",
            "critical: business blocking, data/financial security risk",
        ],
    },
    "escalate_to_manager": {
        "type": "noul",
        "instructions": "Does this issue require immediate manager escalation or fraud review?",
    },
}

def test_single_model(model_info: Dict[str, Any]) -> Dict[str, Any]:
    file_name = model_info["file"]
    display_name = model_info["name"]
    print("\n" + "=" * 75)
    print(f"Testing Model: {display_name}")
    print(f"Hugging Face File: {REPO_ID}/{file_name}")
    print("=" * 75)

    # 1. Download Model from Hugging Face Hub
    print(f"[1/4] Downloading {file_name} from Hugging Face Hub ({REPO_ID})...")
    t0 = time.time()
    try:
        onnx_local = hf_hub_download(repo_id=REPO_ID, filename=file_name)
        if model_info["has_data"]:
            data_file = file_name + ".data"
            print(f"      Downloading external weights: {data_file}...")
            hf_hub_download(repo_id=REPO_ID, filename=data_file)
        dl_time = time.time() - t0
        file_size_mb = os.path.getsize(onnx_local) / (1024 * 1024)
        print(f"      ✓ Downloaded in {dl_time:.2f}s (Cache location: {onnx_local}, Size: {file_size_mb:.1f} MB)")
    except Exception as e:
        print(f"      ✗ Failed to download {file_name} from HF: {e}")
        return {"name": display_name, "file": file_name, "status": "DOWNLOAD_FAILED", "error": str(e)}

    # 2. Instantiate ONNXAgent with HF Tokenizer/Config & downloaded ONNX path
    print(f"[2/4] Initializing QLaya ONNXAgent session from Hugging Face assets...")
    t1 = time.time()
    try:
        agent = ONNXAgent(model_id_or_path=REPO_ID, onnx_path=onnx_local)
        load_time = (time.time() - t1) * 1000
        print(f"      ✓ ONNX runtime session loaded in {load_time:.1f} ms")
    except Exception as e:
        print(f"      ✗ Failed to initialize ONNXAgent: {e}")
        return {"name": display_name, "file": file_name, "status": "LOAD_FAILED", "error": str(e)}

    # 3. Warmup Run
    print("[3/4] Running warmup inference...")
    try:
        _ = agent.system_one(SAMPLE_STATE, SAMPLE_QUESTIONS)
        print("      ✓ Warmup complete")
    except Exception as e:
        print(f"      ✗ Warmup failed: {e}")
        return {"name": display_name, "file": file_name, "status": "INFERENCE_FAILED", "error": str(e)}

    # 4. Timed Inference Runs
    print("[4/4] Executing timed benchmark inference...")
    latencies = []
    decision_result = None
    for i in range(3):
        start = time.perf_counter()
        decision_result = agent.system_one(SAMPLE_STATE, SAMPLE_QUESTIONS)
        latencies.append((time.perf_counter() - start) * 1000)

    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)

    # Inspect decision results
    answers = decision_result.get("answers", {})
    confidence = decision_result.get("confidence", {})

    print(f"\nResults for {display_name}:")
    print(f"  • Latency: {min_latency:.2f} ms (min) | {avg_latency:.2f} ms (avg 3 runs)")
    print(f"  • Predictions:")
    for q_id, ans in answers.items():
        q_type = ans.get("type", "unknown")
        if q_type == "choice":
            ans_val = f"choice='{ans.get('choice')}'"
        elif q_type == "score":
            ans_val = f"score={ans.get('score'):.2f}"
        elif q_type == "noul":
            ans_val = f"noul={ans.get('noul'):.3f}"
        else:
            ans_val = str(ans)
        conf = ans.get("answer_confidence", 0.0)
        print(f"      - {q_id} ({q_type}): {ans_val} (confidence: {conf:.3f})")

    return {
        "name": display_name,
        "file": file_name,
        "status": "SUCCESS",
        "size_mb": file_size_mb,
        "min_latency_ms": min_latency,
        "avg_latency_ms": avg_latency,
        "answers": answers,
        "confidence": confidence,
    }

def print_summary_table(results):
    print("\n" + "=" * 90)
    print(f"{'Model Name':<38} | {'File':<24} | {'Size':<9} | {'Latency':<9} | {'Status'}")
    print("=" * 90)
    for r in results:
        status = r["status"]
        if status == "SUCCESS":
            sz = f"{r['size_mb']:.1f} MB"
            lat = f"{r['min_latency_ms']:.1f} ms"
            status_str = "✓ PASS"
        else:
            sz = "-"
            lat = "-"
            status_str = f"✗ {status}"
        print(f"{r['name']:<38} | {r['file']:<24} | {sz:<9} | {lat:<9} | {status_str}")
    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(description="Test QLaya inference directly from Hugging Face Hub")
    parser.add_argument("--model", type=str, default="all", help="Target ONNX filename, or 'all', or 'recommended'")
    args = parser.parse_args()

    if args.model == "all":
        models_to_test = ALL_MODELS
    elif args.model == "recommended":
        models_to_test = [m for m in ALL_MODELS if m.get("recommended")]
    else:
        models_to_test = [m for m in ALL_MODELS if m["file"] == args.model or m["name"] == args.model]
        if not models_to_test:
            print(f"Model '{args.model}' not found in catalog. Available models:")
            for m in ALL_MODELS:
                print(f"  - {m['file']} ({m['name']})")
            sys.exit(1)

    print(f"\n==========================================================================")
    print(f"  QLaya Hugging Face Inference Suite ({len(models_to_test)} models)")
    print(f"  Repository: https://huggingface.co/{REPO_ID}")
    print(f"==========================================================================")

    results = []
    for model_info in models_to_test:
        res = test_single_model(model_info)
        results.append(res)

    print_summary_table(results)

if __name__ == "__main__":
    main()
