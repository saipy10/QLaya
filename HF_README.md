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

# QLaya: On-Device System 1 Decision Engine

**QLaya** is a fast, non-autoregressive decision engine designed for real-time routing, classification, guardrails, and triage. Rather than generating text token-by-token with heavy autoregressive LLMs, QLaya processes typed questions (`choice`, `score`, `noul`) over unstructured text, tickets, or JSON state in a single forward pass (~33 ms on GPU, ~38 ms on edge CPU) with strictly calibrated probability distributions.

[PyPI Package](https://pypi.org/project/qlaya/) | [Hugging Face Models](https://huggingface.co/saipy10/qlaya) | [GitHub Repository](https://github.com/saipy10/QLaya) | [License: Apache 2.0](LICENSE)

---

## The Experiment: Distillation & Quantization

While large encoder models (such as ModernBERT-large at 421M parameters) provide high decision fidelity, their uncompressed memory footprint (~1.7 GB) and CPU latency (~382 ms) pose operational bottlenecks for edge deployment, microservices, and high-concurrency production.

We systematically explored compression across two complementary axes:
1. **Knowledge Distillation**: Training reduced-depth student models (14 layers and 6 layers) initialized from the teacher's weights and trained against the teacher's gold output probability distributions using RLCD (Reinforcement Learning from Calibrated Distributions).
2. **Quantization**: Applying half-precision (FP16/BF16), dynamic per-channel INT8 quantization via ONNX Runtime, and 4-bit weight quantization (block-32 and block-64).
3. **Compound Compression**: Stacking depth distillation with INT8 and INT4 quantization to identify optimal Pareto frontiers for latency, storage, and accuracy.

---

## Empirical Results Matrix

All benchmarks were evaluated under identical conditions on local x86 CPU hardware (Intel Core i5, AVX-512 VNNI, 8 GB RAM) across the 10 evaluated configurations:

| Configuration | QLaya ID | Technique | Params | Disk Size | Size Δ | Latency (p50) | Choice Acc | RAM Working Set | Status / Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **Teacher (FP32)** | `QLaya-OriginalBaseline` | Uncompressed ModernBERT | 421M | 1,685.2 MB | Baseline | 382.4 ms | 100.0% | 1,720 MB | Original Baseline (Heavy on RAM) |
| **Teacher (FP16 / BF16)** | `QLaya-Balanced` | Weight Half-Precision | 421M | 842.6 MB | -50.0% | 368.0 ms | 100.0% | 860 MB | Balanced |
| **ONNX INT8 (Per-Channel)** | `QLaya-TopProduction` | Dynamic Quantization | 421M | **571.9 MB** | **-66.1%** | **134.7 ms** | **100.0%** | **590 MB** | ⭐ **Top Pick (Zero Loss, 2.8× Speedup)** |
| **Distil-QLaya 14L (FP32)** | `QLaya-IntermediateStudent` | 50% Depth Distillation | 244M | 978.0 MB | -42.0% | 195.2 ms | 100.0% | 1,010 MB | Intermediate Student |
| **Distil-QLaya 14L + INT8** | `QLaya-HighSpeedProduction` | Distilled Student + INT8 | 244M | **332.5 MB** | **-80.3%** | **78.4 ms** | **100.0%** | **350 MB** | ⚡ **High Concurrency (4.9× Speedup)** |
| **Distil-QLaya 6L (FP32)** | `QLaya-CompactStudent` | 6-Layer Compact Student | 143M | 573.6 MB | -66.0% | 94.0 ms | 92.0% | 605 MB | Compact Student |
| **Distil-QLaya 6L + INT8** | `QLaya-UltraFastEdge` | Distilled Student + INT8 | 143M | **195.2 MB** | **-88.4%** | **38.6 ms** | **92.0%** | **210 MB** | 🚀 **Ultra-Fast Edge (9.9× Speedup)** |
| **Distil-QLaya 6L + INT4** | `QLaya-UltraSmallStorage` | Distilled Student + INT4 | 143M | **142.1 MB** | **-91.6%** | 112.5 ms | 88.0% | **160 MB** | 💾 **Minimal Storage Footprint** |
| **ONNX INT4 (Block-32)** | `QLaya-SlowCPU` | 4-bit Weight Quantization | 421M | 441.2 MB | -73.8% | 680.9 ms | 100.0% | 460 MB | CPU Software Unpack Penalty |
| **ONNX INT4 (Block-64)** | `QLaya-DegradedAccuracy` | 4-bit Weight Quantization | 421M | 419.6 MB | -75.1% | 1,047.6 ms | 75.0% | 435 MB | Degraded Accuracy |

---

## Key Observations & Hardware Insights

### 1. Hardware Vector Acceleration (AVX-512 VNNI)
Intel 10th-Gen+ and modern server processors feature native hardware vector dot-product instructions (`vpdpbusd`) for 8-bit integers. ONNX Runtime leverages these execution units directly, delivering a **2.8× speedup (134.7 ms vs 382.4 ms)** with **zero loss in categorical accuracy or calibration**. For general server production, `QLaya-TopProduction` represents the optimal configuration.

### 2. The CPU Software Unpack Penalty of INT4
Standard x86 CPUs lack native 4-bit arithmetic units. Consequently, 4-bit packed weights must be expanded to 8-bit or 32-bit registers in software before matrix operations execute. 
- While INT4 block-32 achieves strong storage compression (441.2 MB), its CPU inference latency deteriorates to **680.9 ms** (~1.8× slower than FP32 and ~5× slower than INT8).
- Wider block sizes (block-64) exacerbate unpack overhead to **1,047.6 ms** and drop choice accuracy down to **75.0%**.
- **Conclusion**: On standard CPU architectures, INT8 strictly outperforms INT4 in latency and efficiency. INT4 is advantageous exclusively when disk storage or transmission bandwidth is the primary constraint.

### 3. Synergies of Depth Distillation + Quantization
Combining architectural distillation with quantization bypasses the compression ceiling of quantization alone:
- **`QLaya-HighSpeedProduction` (14L + INT8)**: Cuts storage by **-80.3%** down to 332.5 MB, runs at **78.4 ms p50 latency**, and retains **100.0% accuracy**.
- **`QLaya-UltraFastEdge` (6L + INT8)**: Achieves sub-40ms CPU inference (**38.6 ms**, a 9.9× speedup over the teacher) while requiring only **210 MB of RAM**, making it ideal for edge appliances and mobile devices.
- **`QLaya-UltraSmallStorage` (6L + INT4)**: Shrinks the original 1.7 GB baseline down to **142.1 MB (-91.6% reduction)** with 160 MB working RAM.

### 4. Multilingual Routing Dynamics
Evaluating decision accuracy across 51 languages (MASSIVE benchmark, 20-way intent classification) demonstrated that language script characteristics dictate backbone requirements:
- For English and clean Latin-script text, the English checkpoint achieves peak accuracy with lowest parameter count.
- For non-Latin scripts (Devanagari, Kana, Han, Arabic, Cyrillic, Thai), the multilingual backbone boosts accuracy by **+10% to +40%** (e.g. Thai +40%, Korean +34%, Hindi +33%, Russian +23%).
- QLaya implements zero-latency script and n-gram routing (`qlaya.Router`) to steer each request to the optimal model variant dynamically.

---

## Installation

```bash
pip install qlaya
```

Optional dependencies:
- `pip install qlaya[onnx]` — ONNX Runtime execution for quantized edge models.
- `pip install qlaya[serve]` — FastAPI HTTP server.
- `pip install qlaya[mcp]` — Model Context Protocol stdio server.

---

## Quickstart: Python Inference

### 1. Run Recommended Production ONNX Model

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
            "support": "technical issues, bug reports, how-to",
            "sales": "upgrades, enterprise plans, pricing",
        },
    }
}

result = agent.predict(state, questions)
print("Decision:", result["intent"]["answer"])
print("Confidence:", result["intent"]["confidence"])
```

### 2. Fast Routing (Pure Python, Zero-Weight Overhead)

```python
import qlaya

router = qlaya.Router()
print(router.route("Refund my duplicate order please"))
# -> RouteDecision(model='english', reason='English Latin text')

print(router.route("お客様は二重に請求されたため返金を希望しています。"))
# -> RouteDecision(model='multilingual', reason='non-Latin script (kana, 100% of letters)...')
```

---

## License

Apache 2.0. Developed by saipy10.
