# qlaya (Dart)

**QLaya** (QLaya) — Dart client for the fast, local, on-device decision engine
with user-selectable quantized model variants.

## Installation

Add to your `pubspec.yaml`:

```yaml
dependencies:
  qlaya: ^0.4.0
```

Then run:

```sh
dart pub get
```

## Quick Start

```dart
import 'package:qlaya/qlaya.dart';

void main() async {
  // List all quantized model variants
  print(QLayaModels.allIds);

  // Pick a model
  final model = QLayaModels.resolve('QLaya-TopProduction');  // ONNX INT8, recommended

  // Connect to a local QLaya server (run: qlaya-serve)
  final client = QLayaClient(baseUrl: 'http://localhost:8000');

  final result = await client.predict(
    text: 'I was charged twice, please refund',
    model: model,
  );

  print(result.answers);
  await client.close();
}
```

## Available Models

All model variants come from the project's benchmark experiments:

| QLaya ID | Size | Latency (p50) | RAM | Notes |
|---|---|---|---|---|
| `QLaya-OriginalBaseline` | 1685 MB | 382 ms | 1720 MB | FP32 teacher, max accuracy |
| `QLaya-Balanced` | 843 MB | 368 ms | 860 MB | FP16/BF16 half-precision |
| `QLaya-TopProduction` ⭐ | 572 MB | 135 ms | 590 MB | ONNX INT8, recommended |
| `QLaya-SlowCPU` | 441 MB | 681 ms | 460 MB | ONNX INT4 block-32 |
| `QLaya-DegradedAccuracy` | 420 MB | 1048 ms | 435 MB | ONNX INT4 block-64 |
| `QLaya-IntermediateStudent` | 978 MB | 195 ms | 1010 MB | 14L FP32 distilled |
| `QLaya-HighSpeedProduction` | 333 MB | 78 ms | 350 MB | 14L INT8 distilled |
| `QLaya-CompactStudent` | 574 MB | 94 ms | 605 MB | 6L FP32 distilled |
| `QLaya-UltraFastEdge` | 195 MB | 39 ms | 210 MB | 6L INT8, edge/mobile |
| `QLaya-UltraSmallStorage` | 142 MB | 113 ms | 160 MB | 6L INT4, minimal storage |

## Model Resolution

`QLayaModels.resolve()` supports fuzzy matching:

```dart
QLayaModels.resolve('QLaya-UltraFastEdge')  // exact
QLayaModels.resolve('ultra-fast-edge')       // fuzzy slug
QLayaModels.resolve('ultrafastedge')         // no hyphens
```

## Running the Server

Install and start the QLaya server (Python):

```sh
pip install qlaya[serve]
qlaya-serve --host 0.0.0.0 --port 8000
```

## License

Apache-2.0 — see [LICENSE](../LICENSE).
