"""Export all distilled student models (14L and 6L) to ONNX and quantized variants."""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
import time
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from safetensors.torch import load_file
from transformers import AutoTokenizer

from qlaya.common import build_model, QTYPES


def step(msg):
    print("\n" + "=" * 70)
    print("  " + msg)
    print("=" * 70)


def export_fp32_onnx(model, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    model.eval()
    model.to("cpu")

    dummy_ids = torch.randint(0, 100, (1, 16), dtype=torch.long)
    dummy_att = torch.ones((1, 16), dtype=torch.long)
    dummy_pos = torch.tensor([[1, 5]], dtype=torch.long)
    dummy_mask = torch.tensor([[True, True]], dtype=torch.bool)
    dummy_qtype = torch.tensor([0], dtype=torch.long)

    inputs = (dummy_ids, dummy_att, dummy_pos, dummy_mask, dummy_qtype)
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "marker_pos": {0: "batch_size", 1: "num_markers"},
        "marker_mask": {0: "batch_size", 1: "num_markers"},
        "qtype": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "num_markers"},
        "act_logits": {0: "batch_size"},
    }

    torch.onnx.export(
        model,
        inputs,
        out_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes=dynamic_axes,
    )
    sz = os.path.getsize(out_path) / (1024 * 1024)
    print("  OK FP32 exported: %s  (%.1f MB, %.0fs)" % (out_path, sz, time.time() - t0))
    return out_path


def export_int8(fp32_path, out_path):
    t0 = time.time()
    m = onnx.load(fp32_path)
    del m.graph.value_info[:]
    quantize_dynamic(
        model_input=m,
        model_output=out_path,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    sz = os.path.getsize(out_path) / (1024 * 1024)
    print("  OK INT8 saved: %s  (%.1f MB, %.0fs)" % (out_path, sz, time.time() - t0))
    return out_path


def export_int4(fp32_path, out_path, block_size=32):
    t0 = time.time()
    m = onnx.load(fp32_path)
    del m.graph.value_info[:]
    q = MatMulNBitsQuantizer(model=m, bits=4, block_size=block_size, is_symmetric=True)
    q.process()
    q.model.save_model_to_file(out_path, use_external_data_format=False)
    sz = os.path.getsize(out_path) / (1024 * 1024)
    print("  OK INT4 b%d saved: %s  (%.1f MB, %.0fs)" % (block_size, out_path, sz, time.time() - t0))
    return out_path


def build_student(ckpt_dir, num_layers, base_encoder_dir):
    import json
    with open(os.path.join(ckpt_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["num_layers"] = num_layers
    
    # build uninitialized architecture
    model = build_model(cfg, encoder_dir=base_encoder_dir, pretrained=False)
    
    # slice encoder layers
    if num_layers == 14:
        model.encoder.layers = torch.nn.ModuleList([model.encoder.layers[i] for i in range(0, 28, 2)])
    elif num_layers == 6:
        indices = [0, 5, 10, 15, 20, 27]
        model.encoder.layers = torch.nn.ModuleList([model.encoder.layers[i] for i in indices])
    model.encoder.config.num_hidden_layers = num_layers
    
    # load trained weights
    weights = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    return model


def main():
    base_encoder_dir = "F:/Project/laya/models/base/encoder"
    distilled_root = "F:/Project/laya/models/distilled"
    onnx_out_dir = os.path.join(distilled_root, "onnx")
    os.makedirs(onnx_out_dir, exist_ok=True)

    # 1. 14L Student
    dir_14l = os.path.join(distilled_root, "distil_qlaya_14l")
    if os.path.exists(os.path.join(dir_14l, "model.safetensors")):
        step("Exporting Distil-QLaya 14L Variants (Models #6 & #7)")
        s14 = build_student(dir_14l, 14, base_encoder_dir)
        fp32_14 = os.path.join(onnx_out_dir, "distil_qlaya_14l.fp32.onnx")
        export_fp32_onnx(s14, fp32_14)
        int8_14 = os.path.join(onnx_out_dir, "distil_qlaya_14l.int8.onnx")
        export_int8(fp32_14, int8_14)

    # 2. 6L Student
    dir_6l = os.path.join(distilled_root, "distil_qlaya_6l")
    if os.path.exists(os.path.join(dir_6l, "model.safetensors")):
        step("Exporting Distil-QLaya 6L Variants (Models #8, #9, #10)")
        s6 = build_student(dir_6l, 6, base_encoder_dir)
        fp32_6 = os.path.join(onnx_out_dir, "distil_qlaya_6l.fp32.onnx")
        export_fp32_onnx(s6, fp32_6)
        int8_6 = os.path.join(onnx_out_dir, "distil_qlaya_6l.int8.onnx")
        export_int8(fp32_6, int8_6)
        int4_6 = os.path.join(onnx_out_dir, "distil_qlaya_6l.int4.onnx")
        export_int4(fp32_6, int4_6, block_size=32)

    step("All Distilled Models Exported and Quantized Successfully!")


if __name__ == "__main__":
    main()
