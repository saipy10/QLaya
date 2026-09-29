---
license: apache-2.0
tags:
- decision-engine
- agentic-routing
- onnx
- quantization
- distillation
- modernbert
pipeline_tag: text-classification
---

# QLaya: Ultra-Fast On-Device Decision Engine

**QLaya** is an ultra-fast, local, on-device decision engine designed for AI agent routing, policy evaluation, and structured categorical choices.

This repository (`saipy10/qlaya`) provides the full suite of **10 production, quantized, and distilled models** derived from the 28-layer ModernBERT teacher architecture.

---

## Model Benchmark & Family Overview

| # | Model ID / Variant | Technique | File Name | Size | Latency (CPU) | Choice Acc | Recommended For |
|---|---|---|---|---|---|---|---|
| **1** | `QLaya-OriginalBaseline` | FP32 Uncompressed Teacher | `qlaya.fp32.onnx` | 1,688 MB | 382 ms | 100% | Reference Baseline |
| **2** | `QLaya-Balanced` | FP16 Weight Half-Precision | `qlaya.fp16.onnx` | 806.6 MB | 368 ms | 100% | Full Precision GPU/CPU |
| **3** | `QLaya-TopProduction` ⭐ | INT8 Dynamic Quantization | `qlaya.int8.onnx` | **571.9 MB** | **134 ms** | **100%** | **Top Production Choice** |
| **4** | `QLaya-SlowCPU` | INT4 Block-32 Quantization | `qlaya.int4_b32.onnx` | 441.2 MB | 680 ms | 100% | Memory-Constrained Systems |
| **5** | `QLaya-DegradedAccuracy` | INT4 Block-64 Quantization | `qlaya.int4_b64.onnx` | 419.6 MB | 1,047 ms | 75% | Storage Benchmark Only |
| **6** | `QLaya-IntermediateStudent` | 50% Depth Distillation (14L) | `distil_qlaya_14l.fp32.onnx` | 999 MB | 195 ms | 100% | Fast 14-Layer Baseline |
| **7** | `QLaya-HighSpeedProduction` | 14L Student + INT8 Quant | `distil_qlaya_14l.int8.onnx` | **406.2 MB** | **78 ms** | **100%** | **High-Concurrency Services** |
| **8** | `QLaya-CompactStudent` | 6-Layer Compact Student | `distil_qlaya_6l.fp32.onnx` | 607 MB | 94 ms | 92% | Compact 6-Layer Baseline |
| **9** | `QLaya-UltraFastEdge` 🚀 | 6L Student + INT8 Quant | `distil_qlaya_6l.int8.onnx` | **311.4 MB** | **38 ms** | **92%** | **Ultra-Fast Mobile / Edge Gateways** |
| **10** | `QLaya-UltraSmallStorage` 💾 | 6L Student + 4-bit INT4 Quant | `distil_qlaya_6l.int4.onnx` | **278.4 MB** | **112 ms** | **88%** | **Minimal Storage Footprint** |

---

## Quickstart: Python Inference

Install the dependencies:
```bash
pip install qlaya onnxruntime transformers
```

### 1. Run Top Recommended Model (`QLaya-TopProduction` INT8)
```python
from huggingface_hub import hf_hub_download
from qlaya.onnx_agent import ONNXAgent

# Download ONNX checkpoint directly from this repository
onnx_file = hf_hub_download(repo_id="saipy10/qlaya", filename="qlaya.int8.onnx")
agent = ONNXAgent("saipy10/qlaya", onnx_path=onnx_file)

state = "The user is asking for a full refund for duplicate charge on invoice #9481."
questions = {
    "intent": {
        "type": "choice",
        "instructions": "Route this ticket to the appropriate department.",
        "criteria": {
            "billing": "charges, invoices, payment, refunds",
            "technical": "server errors, bug reports, downtime",
            "security": "unauthorized access, account breaches"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this customer issue?",
        "criteria": ["low", "normal", "high", "critical"]
    }
}

result = agent.predict(state, questions)
print("Intent  :", result["answers"]["intent"]["choice"])
print("Urgency :", result["answers"]["urgency"]["score"])
```

### 2. Run Ultra-Fast Edge Model (`QLaya-UltraFastEdge` 38ms)
```python
onnx_file = hf_hub_download(repo_id="saipy10/qlaya", filename="distil_qlaya_6l.int8.onnx")
agent = ONNXAgent("saipy10/qlaya", onnx_path=onnx_file, subfolder="distil_qlaya_6l")

result = agent.predict(state, questions)
print("Edge Prediction:", result["answers"]["intent"]["choice"])
```

---

## PyTorch Checkpoints
For PyTorch training or further distillation, PyTorch weights are provided in:
* Root: 28-layer Teacher (`model.safetensors`, `rl_agent_config.json`)
* Subfolder `distil_qlaya_14l`: 14-layer Student
* Subfolder `distil_qlaya_6l`: 6-layer Student
