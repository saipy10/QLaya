# qlaya (npm)

**QLaya** (QLaya) — fast, local, on-device decision engine with user-selectable
quantized model variants. TypeScript/JavaScript runtime using ONNX.

## Installation

```sh
npm install qlaya
# ONNX runtime (pick one):
npm install onnxruntime-node   # Node.js
npm install onnxruntime-web    # Browser / Deno
```

## Quick Start

Ensure your `package.json` includes `"type": "module"` (or save your script with a `.mjs` extension).

```ts
import { Router, QLAYA_MODELS, QLAYA_MODEL_IDS, resolveQLModel } from "qlaya";

// List all quantized model variants
console.log(QLAYA_MODEL_IDS);
// ['QLaya-fp32', 'QLaya-fp16', 'QLaya-int8', 'QLaya-int4-b32', 'QLaya-int4-b64',
//  'QLaya-14L-fp32', 'QLaya-14L-int8', 'QLaya-6L-fp32', 'QLaya-6L-int8', 'QLaya-6L-int4']

// Quickstart — select a quantized QLaya variant directly by name:
const router = new Router("QLaya-int8");                // ONNX INT8, recommended
const edge   = new Router("QLaya-6L-int8");             // 6L INT8, ~39 ms edge
const small  = new Router("QLaya-6L-int4");             // 6L INT4, 142 MB

// Optional: preload model explicitly at application startup (otherwise auto-loaded on predict):
await router.load("QLaya-int8");

// Define routing questions
const questions = {
  intent: {
    type: "choice",
    instructions: "What is the customer's primary intent?",
    criteria: {
      refund: "Customer requests refund or dispute charge",
      support: "Technical help or usage question",
      sales: "Pricing, upgrade, or commercial inquiry",
    },
  },
  urgency: {
    type: "score",
    instructions: "Rate urgency from 0 to 3",
    criteria: ["low", "medium", "high", "critical"],
  },
};

// Predict answers and calibrated probabilities
const result = await router.predict(
  { text: "I was charged twice, please refund" },
  questions,
);

console.log("Intent:", result.answers.intent.choice);
console.log("Urgency score:", result.answers.urgency.score);
```

## Available Models

| QLaya ID | Size | Latency (p50) | RAM | Notes |
|---|---|---|---|---|
| `QLaya-fp32` | 1685 MB | 382 ms | 1720 MB | FP32 teacher baseline |
| `QLaya-fp16` | 843 MB | 368 ms | 860 MB | FP16/BF16 balanced |
| `QLaya-int8` ⭐ | 572 MB | 135 ms | 590 MB | ONNX INT8 recommended |
| `QLaya-int4-b32` | 441 MB | 681 ms | 460 MB | ONNX INT4 block-32 |
| `QLaya-int4-b64` | 420 MB | 1048 ms | 435 MB | ONNX INT4 block-64 |
| `QLaya-14L-fp32` | 978 MB | 195 ms | 1010 MB | 14L FP32 distilled student |
| `QLaya-14L-int8` | 333 MB | 78 ms | 350 MB | 14L INT8 distilled student |
| `QLaya-6L-fp32` | 574 MB | 94 ms | 605 MB | 6L FP32 compact student |
| `QLaya-6L-int8` | 195 MB | 39 ms | 210 MB | 6L INT8, ultra-fast edge |
| `QLaya-6L-int4` | 142 MB | 113 ms | 160 MB | 6L INT4, minimal storage |

## License

Apache-2.0
