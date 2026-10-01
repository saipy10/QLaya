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
| **Teacher (FP32)** | `QLaya-fp32` | Uncompressed ModernBERT | 421M | 1,685.2 MB | Baseline | 382.4 ms | 100.0% | 1,720 MB | Original Baseline (Heavy on RAM) |
| **Teacher (FP16 / BF16)** | `QLaya-fp16` | Weight Half-Precision | 421M | 842.6 MB | -50.0% | 368.0 ms | 100.0% | 860 MB | Balanced |
| **ONNX INT8 (Per-Channel)** | `QLaya-int8` | Dynamic Quantization | 421M | **571.9 MB** | **-66.1%** | **134.7 ms** | **100.0%** | **590 MB** | ⭐ **Top Pick (Zero Loss, 2.8× Speedup)** |
| **Distil-QLaya 14L (FP32)** | `QLaya-14L-fp32` | 50% Depth Distillation | 244M | 978.0 MB | -42.0% | 195.2 ms | 100.0% | 1,010 MB | Intermediate Student |
| **Distil-QLaya 14L + INT8** | `QLaya-14L-int8` | Distilled Student + INT8 | 244M | **332.5 MB** | **-80.3%** | **78.4 ms** | **100.0%** | **350 MB** | ⚡ **High Concurrency (4.9× Speedup)** |
| **Distil-QLaya 6L (FP32)** | `QLaya-6L-fp32` | 6-Layer Compact Student | 143M | 573.6 MB | -66.0% | 94.0 ms | 92.0% | 605 MB | Compact Student |
| **Distil-QLaya 6L + INT8** | `QLaya-6L-int8` | Distilled Student + INT8 | 143M | **195.2 MB** | **-88.4%** | **38.6 ms** | **92.0%** | **210 MB** | 🚀 **Ultra-Fast Edge (9.9× Speedup)** |
| **Distil-QLaya 6L + INT4** | `QLaya-6L-int4` | Distilled Student + INT4 | 143M | **142.1 MB** | **-91.6%** | 112.5 ms | 88.0% | **160 MB** | 💾 **Minimal Storage Footprint** |
| **ONNX INT4 (Block-32)** | `QLaya-int4-b32` | 4-bit Weight Quantization | 421M | 441.2 MB | -73.8% | 680.9 ms | 100.0% | 460 MB | CPU Software Unpack Penalty |
| **ONNX INT4 (Block-64)** | `QLaya-int4-b64` | 4-bit Weight Quantization | 421M | 419.6 MB | -75.1% | 1,047.6 ms | 75.0% | 435 MB | Degraded Accuracy |

---

## Key Observations & Hardware Insights

### 1. Hardware Vector Acceleration (AVX-512 VNNI)
Intel 10th-Gen+ and modern server processors feature native hardware vector dot-product instructions (`vpdpbusd`) for 8-bit integers. ONNX Runtime leverages these execution units directly, delivering a **2.8× speedup (134.7 ms vs 382.4 ms)** with **zero loss in categorical accuracy or calibration**. For general server production, `QLaya-int8` represents the optimal configuration.

### 2. The CPU Software Unpack Penalty of INT4
Standard x86 CPUs lack native 4-bit arithmetic units. Consequently, 4-bit packed weights must be expanded to 8-bit or 32-bit registers in software before matrix operations execute. 
- While INT4 block-32 achieves strong storage compression (441.2 MB), its CPU inference latency deteriorates to **680.9 ms** (~1.8× slower than FP32 and ~5× slower than INT8).
- Wider block sizes (block-64) exacerbate unpack overhead to **1,047.6 ms** and drop choice accuracy down to **75.0%**.
- **Conclusion**: On standard CPU architectures, INT8 strictly outperforms INT4 in latency and efficiency. INT4 is advantageous exclusively when disk storage or transmission bandwidth is the primary constraint.

### 3. Synergies of Depth Distillation + Quantization
Combining architectural distillation with quantization bypasses the compression ceiling of quantization alone:
- **`QLaya-14L-int8` (14L + INT8)**: Cuts storage by **-80.3%** down to 332.5 MB, runs at **78.4 ms p50 latency**, and retains **100.0% accuracy**.
- **`QLaya-6L-int8` (6L + INT8)**: Achieves sub-40ms CPU inference (**38.6 ms**, a 9.9× speedup over the teacher) while requiring only **210 MB of RAM**, making it ideal for edge appliances and mobile devices.
- **`QLaya-6L-int4` (6L + INT4)**: Shrinks the original 1.7 GB baseline down to **142.1 MB (-91.6% reduction)** with 160 MB working RAM.

### 4. Multilingual Routing Dynamics
Evaluating decision accuracy across 51 languages (MASSIVE benchmark, 20-way intent classification) demonstrated that language script characteristics dictate backbone requirements:
- For English and clean Latin-script text, the English checkpoint achieves peak accuracy with lowest parameter count.
- For non-Latin scripts (Devanagari, Kana, Han, Arabic, Cyrillic, Thai), the multilingual backbone boosts accuracy by **+10% to +40%** (e.g. Thai +40%, Korean +34%, Hindi +33%, Russian +23%).
- QLaya implements zero-latency script and n-gram routing (`qlaya.Router`) to steer each request to the optimal model variant dynamically.

---

## Installation

### Python
```bash
pip install -U qlaya
```

Optional dependencies:
- `pip install qlaya[onnx]` — ONNX Runtime execution for local quantized models.
- `pip install qlaya[serve]` — FastAPI HTTP server.
- `pip install qlaya[mcp]` — Model Context Protocol stdio server.

### TypeScript / Node.js
```bash
npm install qlaya
```

> **Caution:** Do not name your Python test scripts `qlaya.py`. Python's module resolution prioritises files in the current directory over installed packages, causing `import qlaya` to import your own script and producing confusing errors like `AttributeError: module 'qlaya' has no attribute 'Router'`.

---

## Quickstart: Python

### 1. Inspect Available Model Variants

```python
import qlaya

print(qlaya.QLAYA_MODEL_IDS)
# ['QLaya-fp32', 'QLaya-fp16', 'QLaya-int8', 'QLaya-int4-b32', 'QLaya-int4-b64',
#  'QLaya-14L-fp32', 'QLaya-14L-int8', 'QLaya-6L-fp32', 'QLaya-6L-int8', 'QLaya-6L-int4']
```

### 2. Fast Routing (Pure Python, Microsecond Zero-Weight Overhead)

```python
import qlaya

router = qlaya.Router()

# Automatically routes based on language and script analysis:
print(router.route("Refund my duplicate order please"))
# -> RouteDecision(model='english', reason='English Latin text')

print(router.route("お客様は二重に請求されたため返金を希望しています。"))
# -> RouteDecision(model='multilingual', reason='non-Latin script (kana, 100% of letters)...')

print(router.route("ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।"))
# -> RouteDecision(model='multilingual', reason='non-Latin script (devanagari, 100% of letters)...')
```

### 3. Deploying a Quantized Variant Seamlessly

You can select any variant either by name, keyword, or as the default:

```python
import qlaya

# Positional variant selection (top recommended INT8 production model):
router = qlaya.Router("QLaya-int8")

# Or via model keyword argument (e.g. ultra-fast sub-40ms edge model):
edge_router = qlaya.Router(model="QLaya-6L-int8")

# Or set high-concurrency 14L INT8 student as default route:
distil_router = qlaya.Router(default="QLaya-14L-int8")
```

### 4. Running ONNX Quantized Model Inference

```python
from qlaya.onnx_agent import ONNXAgent

# Loads directly via model ID or ONNX filename (auto-downloads from HF if missing):
agent = ONNXAgent("saipy10/qlaya", onnx_path="qlaya.int8.onnx")
# Or load by ID directly: agent = ONNXAgent(model_id_or_path="QLaya-int8")

state = "I was charged twice for my subscription this month. Please refund."
questions = {
    "intent": {
        "type": "choice",
        "instructions": "Route this ticket to the appropriate department.",
        "criteria": {
            "billing": "charges, invoices, payment, refunds",
            "support": "technical issues, bug reports, how-to",
            "sales": "upgrades, enterprise plans, pricing",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this customer issue?",
        "criteria": ["low", "normal", "high", "critical"],
    },
}

# predict() returns {"model": ..., "answers": {<qid>: {...}}, "usage": {...}}
result = agent.predict(state, questions)
answers = result["answers"]

print("Intent:", answers["intent"]["choice"])                 # billing
print("Confidence:", answers["intent"]["confidence"])         # e.g. 0.96
print("Calibrated Conf:", answers["intent"]["answer_confidence"])
print("Urgency Score:", answers["urgency"]["score"])          # continuous score e.g. 2.15
```

### 5. Email Text Cleaning & Spam/Phishing Detection

```python
from qlaya.onnx_agent import ONNXAgent
from qlaya import clean_email_body, email_state, email_questions

agent = ONNXAgent("saipy10/qlaya", onnx_path="qlaya.int8.onnx")

raw_body = """Hi Support,
I need help with my account.
On Mon, Jan 15, 2026 at 10:00 AM, Support <support@example.com> wrote:
> Thank you for contacting us."""

# 1. Clean email body (strips quotation blocks and reply headers)
cleaned = clean_email_body(raw_body)

# 2. Package into calibrated email state
state = email_state(
    subject="Immediate action required: account suspended",
    body=cleaned,
    sender="alerts@security-account-verification.com",
)

# 3. Evaluate specific email security questions
all_q = email_questions()
eval_q = {
    "is_phishing": all_q["is_phishing"],
    "is_spam": all_q["is_spam"],
}

result = agent.predict(state, eval_q)
answers = result["answers"]

# Continuous probabilities (0.0 to 1.0)
phishing_score = answers["is_phishing"]["noul"]
spam_score = answers["is_spam"]["noul"]

print(f"Phishing Score: {phishing_score:.4f} -> Flagged: {phishing_score > 0.5}")
print(f"Spam Score:     {spam_score:.4f} -> Flagged: {spam_score > 0.5}")
```

### 6. Calibrated Confidence Scoring Metrics

```python
import numpy as np
import qlaya

# Calibrated answer confidence (top probability):
probs = np.array([0.88, 0.08, 0.04])
conf = qlaya.answer_confidence(probs, k=len(probs))
print(f"Top answer confidence: {conf:.4f}")  # 0.8800

# Normalized Shannon entropy confidence:
entropy_conf = qlaya.confidence_from_probs(probs, k=len(probs))
print(f"Entropy sharpness: {entropy_conf:.4f}")
```

---

## Quickstart: TypeScript / Node.js

```typescript
import { Router, QLAYA_MODELS, QLAYA_MODEL_IDS, resolveQLModel } from "qlaya";

// Quickstart — select a quantized variant directly:
const router = new Router("QLaya-int8");
const edgeRouter = new Router({ model: "QLaya-6L-int8" });

// Pure zero-latency language/script routing:
const decision = router.route("Refund my duplicate order please");
console.log(decision);
// { model: 'english', repo: 'saipy10/qlaya/qlaya-int8', reason: 'English Latin text' }

// Multilingual detection:
console.log(router.route("お客様は二重に請求されたため返金を希望しています。"));
// { model: 'multilingual', ... }
```

---

## License

Apache 2.0. Developed by saipy10.
