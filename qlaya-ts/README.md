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

```ts
import { Router, QLAYA_MODELS, QLAYA_MODEL_IDS, resolveQLModel } from "qlaya";

// List all quantized model variants
console.log(QLAYA_MODEL_IDS);

// Quickstart — select a quantized QLaya variant directly by name:
const router = new Router("QLaya-TopProduction");               // ONNX INT8, recommended
const edge   = new Router("QLaya-UltraFastEdge");               // 6L INT8, ~39 ms edge
const small  = new Router("QLaya-UltraSmallStorage");           // 6L INT4, 142 MB

// Resolve a model spec by QLaya ID manually:
const spec     = resolveQLModel("QLaya-TopProduction");  // { repo, subfolder }
const edgeSpec = resolveQLModel("ultra-fast-edge");       // fuzzy slug, 6L INT8

// Default multi-language routing:
const defaultRouter = new Router();
const result = await defaultRouter.predict(
  { text: "I was charged twice, please refund" },
  questions,
);
console.log(result);
```

## Available Models

| QLaya ID | Size | Latency (p50) | RAM | Notes |
|---|---|---|---|---|
| `QLaya-OriginalBaseline` | 1685 MB | 382 ms | 1720 MB | FP32 teacher |
| `QLaya-Balanced` | 843 MB | 368 ms | 860 MB | FP16/BF16 |
| `QLaya-TopProduction` ⭐ | 572 MB | 135 ms | 590 MB | ONNX INT8 recommended |
| `QLaya-SlowCPU` | 441 MB | 681 ms | 460 MB | ONNX INT4 b32 |
| `QLaya-DegradedAccuracy` | 420 MB | 1048 ms | 435 MB | ONNX INT4 b64 |
| `QLaya-IntermediateStudent` | 978 MB | 195 ms | 1010 MB | 14L FP32 distilled |
| `QLaya-HighSpeedProduction` | 333 MB | 78 ms | 350 MB | 14L INT8 distilled |
| `QLaya-CompactStudent` | 574 MB | 94 ms | 605 MB | 6L FP32 distilled |
| `QLaya-UltraFastEdge` | 195 MB | 39 ms | 210 MB | 6L INT8, edge/mobile |
| `QLaya-UltraSmallStorage` | 142 MB | 113 ms | 160 MB | 6L INT4, smallest |

## License

Apache-2.0
