import argparse
import os
import sys
import time


def step(msg):
    print("\n" + "=" * 70)
    print("  " + msg)
    print("=" * 70)


def export_fp32(model_path, out_path):
    step("[1/5] Exporting FP32 ONNX from " + model_path)
    import torch
    from qlaya.agent import Agent

    agent = Agent(model_path, compile=False, device="cpu")
    dummy_input_ids = torch.randint(0, 100, (1, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 16), dtype=torch.long)
    dummy_marker_pos = torch.tensor([[1, 5]], dtype=torch.long)
    dummy_marker_mask = torch.tensor([[True, True]], dtype=torch.bool)
    dummy_qtype = torch.tensor([0], dtype=torch.long)
    inputs = (dummy_input_ids, dummy_attention_mask, dummy_marker_pos, dummy_marker_mask, dummy_qtype)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "marker_pos": {0: "batch_size", 1: "num_markers"},
        "marker_mask": {0: "batch_size", 1: "num_markers"},
        "qtype": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "num_markers"},
        "act_logits": {0: "batch_size"},
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    t0 = time.time()
    torch.onnx.export(
        agent.model, inputs, out_path,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes=dynamic_axes,
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print("  OK FP32 exported: %s  (%.1f MB, %.0fs)" % (out_path, size_mb, time.time() - t0))
    return out_path


def export_fp16(fp32_path, out_path):
    step("[2/5] Converting FP32 ONNX to FP16 (Balanced)")
    try:
        from onnxconverter_common import float16
    except ImportError:
        print("  Installing onnxconverter-common...")
        os.system(sys.executable + " -m pip install onnxconverter-common -q")
        from onnxconverter_common import float16
    import onnx
    model = onnx.load(fp32_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print("  OK FP16 saved: %s  (%.1f MB)" % (out_path, size_mb))
    return out_path


def export_int8(fp32_path, out_path):
    step("[3/5] Quantizing to INT8 per-channel (TopProduction)")
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model = onnx.load(fp32_path)
    del model.graph.value_info[:]
    t0 = time.time()
    quantize_dynamic(
        model_input=model, model_output=out_path,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8, per_channel=True,
    )
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print("  OK INT8 saved: %s  (%.1f MB, %.0fs)" % (out_path, size_mb, time.time() - t0))
    return out_path


def export_int4(fp32_path, out_path, block_size, label):
    step("[?/5] Quantizing to INT4 block-%d (%s)" % (block_size, label))
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    model = onnx.load(fp32_path)
    del model.graph.value_info[:]
    t0 = time.time()
    q = MatMulNBitsQuantizer(model=model, bits=4, block_size=block_size, is_symmetric=True)
    q.process()
    q.model.save_model_to_file(out_path, use_external_data_format=False)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print("  OK INT4 b%d saved: %s  (%.1f MB, %.0fs)" % (block_size, out_path, size_mb, time.time() - t0))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Regenerate all QLaya quantized ONNX model variants")
    parser.add_argument("--model", default="F:/Project/laya/models/base",
                        help="Teacher model directory")
    parser.add_argument("--output-dir", default="F:/Project/laya/models/quantized",
                        help="Output directory for ONNX files")
    parser.add_argument("--skip-fp32-export", action="store_true",
                        help="Skip FP32 export if qlaya.fp32.onnx already exists in output-dir")
    args = parser.parse_args()

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    fp32_path = os.path.join(out_dir, "qlaya.fp32.onnx")
    fp16_path = os.path.join(out_dir, "qlaya.fp16.onnx")
    int8_path = os.path.join(out_dir, "qlaya.int8.onnx")
    int4b32   = os.path.join(out_dir, "qlaya.int4_b32.onnx")
    int4b64   = os.path.join(out_dir, "qlaya.int4_b64.onnx")

    print("\nQLaya Quantized Model Regeneration Pipeline")
    print("  Teacher : " + args.model)
    print("  Out dir : " + out_dir)

    if args.skip_fp32_export and os.path.exists(fp32_path):
        print("\n[1/5] Skipping FP32 export — file already exists")
    else:
        export_fp32(args.model, fp32_path)

    export_fp16(fp32_path, fp16_path)
    export_int8(fp32_path, int8_path)

    step("[4/5] Quantizing to INT4 block-32 (SlowCPU)")
    export_int4(fp32_path, int4b32, block_size=32, label="SlowCPU")

    step("[5/5] Quantizing to INT4 block-64 (DegradedAccuracy)")
    export_int4(fp32_path, int4b64, block_size=64, label="DegradedAccuracy")

    print("\n" + "=" * 70)
    print("  ALL QUANTIZED VARIANTS GENERATED")
    print("=" * 70)
    for path in [fp32_path, fp16_path, int8_path, int4b32, int4b64]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print("  %-35s  %8.1f MB" % (os.path.basename(path), size_mb))

    print()
    print("  NOTE: Distilled student models (QLaya-IntermediateStudent,")
    print("        QLaya-HighSpeedProduction, QLaya-CompactStudent,")
    print("        QLaya-UltraFastEdge, QLaya-UltraSmallStorage)")
    print("        require GPU training. Run scripts/distill_student.py")
    print("        on a machine with a CUDA GPU or use Google Colab.")


if __name__ == "__main__":
    main()
