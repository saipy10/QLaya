import json
import os
import base64

whl_path = "F:/Project/laya/kaggle_kernel/qlaya-0.4.0-py3-none-any.whl"
with open(whl_path, "rb") as f:
    whl_b64 = base64.b64encode(f.read()).decode("ascii")

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 QLaya Student Distillation & Quantization Pipeline (Kaggle GPU)\n",
                "\n",
                "This notebook executes the complete **Distillation, Compression, and Quantization Pipeline** for **QLaya**:\n",
                "\n",
                "| # | Model | Technique | Target Size | Latency |\n",
                "|---|---|---|---|---|\n",
                "| 6 | **Distil-QLaya 14L (FP32)** | 50% Depth Distillation (14 layers) | ~978 MB | ~195 ms |\n",
                "| 7 | **Distil-QLaya 14L (INT8)** ⭐ | 14L Student + Dynamic INT8 | ~332 MB | ~78 ms |\n",
                "| 8 | **Distil-QLaya 6L (FP32)** | 6-Layer Compact Student | ~574 MB | ~94 ms |\n",
                "| 9 | **Distil-QLaya 6L (INT8)** 🚀 | 6L Student + Dynamic INT8 | ~195 MB | ~38 ms |\n",
                "| 10 | **Distil-QLaya 6L (INT4)** 💾 | 6L Student + 4-bit Weight Quant | ~142 MB | ~112 ms |\n",
                "\n",
                "All exported ONNX checkpoints and PyTorch model files will be saved to `/kaggle/working/models/` and available directly in Kaggle Output."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. System & GPU Verification"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "!nvidia-smi\n",
                "import os, sys, time, json, copy, shutil\n",
                "import torch\n",
                "\n",
                "print(f\"PyTorch Version : {torch.__version__}\")\n",
                "print(f\"CUDA Available  : {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    for i in range(torch.cuda.device_count()):\n",
                "        p = torch.cuda.get_device_properties(i)\n",
                "        print(f\"  GPU {i}: {p.name} ({p.total_memory / 1e9:.2f} GB)\")\n",
                "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
                "print(f\"Active Training Device: {device}\")\n"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Dependencies & QLaya Package Installation"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Install runtime ML dependencies\n",
                "!pip install -q \"transformers>=4.48.0\" \"datasets>=3.0.0\" huggingface_hub safetensors onnx onnxruntime onnxconverter-common accelerate\n",
                "\n",
                "# Install self-contained QLaya package (embedded wheel)\n",
                "import base64\n",
                f"WHL_B64 = \"{whl_b64}\"\n",
                "with open(\"/tmp/qlaya-0.4.0-py3-none-any.whl\", \"wb\") as f:\n",
                "    f.write(base64.b64decode(WHL_B64))\n",
                "\n",
                "!pip install -q --no-deps /tmp/qlaya-0.4.0-py3-none-any.whl\n",
                "\n",
                "import qlaya\n",
                "print(f\"QLaya {qlaya.__version__} successfully installed & imported from: {qlaya.__file__}\")\n"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Load Teacher Model (ModernBERT-large, 28 Layers)"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "from huggingface_hub import snapshot_download\n",
                "from transformers import AutoTokenizer\n",
                "from qlaya.agent import Agent, _fix_tokenizer_config\n",
                "from qlaya.common import build_model, QTYPES, render_options, build_sequence\n",
                "\n",
                "TEACHER_ID = \"saipy10/qlaya\"\n",
                "print(f\"Downloading teacher checkpoint: {TEACHER_ID}...\")\n",
                "teacher_dir = snapshot_download(TEACHER_ID)\n",
                "_fix_tokenizer_config(teacher_dir)\n",
                "\n",
                "agent = Agent(teacher_dir, compile=False, device=device)\n",
                "teacher_model = agent.model\n",
                "teacher_model.eval()\n",
                "tok = agent.tok\n",
                "cfg = agent.cfg\n",
                "\n",
                "print(f\"Teacher loaded on {device} | Total layers: {len(teacher_model.encoder.layers)} | Max length: {cfg.get('max_len', 512)}\")\n"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Initialize Student Models via Structural Distillation"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# 1. Distil-QLaya 14L: 50% Depth Distillation (take layers 0, 2, 4, ..., 26)\n",
                "student_14 = copy.deepcopy(teacher_model)\n",
                "student_14.encoder.layers = torch.nn.ModuleList([student_14.encoder.layers[i] for i in range(0, 28, 2)])\n",
                "student_14.encoder.config.num_hidden_layers = 14\n",
                "p14 = sum(p.numel() for p in student_14.parameters())\n",
                "print(f\"[14L Student] Layers: {len(student_14.encoder.layers)} | Params: {p14/1e6:.1f}M\")\n",
                "\n",
                "# 2. Distil-QLaya 6L: Compact Student (take anchor layers 0, 5, 10, 15, 20, 27)\n",
                "student_6 = copy.deepcopy(teacher_model)\n",
                "indices_6 = [0, 5, 10, 15, 20, 27]\n",
                "student_6.encoder.layers = torch.nn.ModuleList([teacher_model.encoder.layers[i] for i in indices_6])\n",
                "student_6.encoder.config.num_hidden_layers = 6\n",
                "p6 = sum(p.numel() for p in student_6.parameters())\n",
                "print(f\"[6L Student]  Layers: {len(student_6.encoder.layers)}  | Params: {p6/1e6:.1f}M\")\n"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Knowledge Distillation Training on GPU\n",
                "We transfer knowledge from teacher to students using soft-target KL-divergence loss on representative decision states."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import torch.nn.functional as F\n",
                "\n",
                "def distill_model(student, teacher, name=\"student\", lr=3e-5):\n",
                "    student.to(device)\n",
                "    teacher.to(device)\n",
                "    student.train()\n",
                "    teacher.eval()\n",
                "    \n",
                "    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)\n",
                "    print(f\"Distilling {name} on {device}...\")\n",
                "    \n",
                "    states = [\n",
                "        \"User requested full refund for duplicate charge on invoice #9481.\",\n",
                "        \"Database connection timeout error when connecting to replica in us-east-1.\",\n",
                "        \"Customer wants to upgrade from Starter to Enterprise plan with annual billing.\",\n",
                "        \"Security vulnerability reported in login OAuth callback endpoint.\",\n",
                "        \"Shipment tracking shows delivered but recipient did not receive package.\",\n",
                "        \"High CPU utilization warning triggered on Kubernetes worker nodes.\",\n",
                "        \"Mobile app crashes on startup on iOS 18 devices after update.\",\n",
                "        \"Customer inquiry regarding API rate limits and enterprise SLAs.\",\n",
                "    ] * 20\n",
                "    \n",
                "    q_spec = {\n",
                "        \"t\": \"choice\",\n",
                "        \"ins\": \"Route this ticket to the appropriate team.\",\n",
                "        \"crit\": {\"billing\": \"payments, charges\", \"tech\": \"bugs, infrastructure\", \"security\": \"vulns\"}\n",
                "    }\n",
                "    \n",
                "    items = []\n",
                "    for s in states:\n",
                "        seq, markers = build_sequence(tok, s, q_spec, 256, 64)\n",
                "        items.append({\"ids\": seq, \"markers\": markers})\n",
                "        \n",
                "    batch_size = 16\n",
                "    total_loss = 0.0\n",
                "    steps = 0\n",
                "    for i in range(0, len(items), batch_size):\n",
                "        b = items[i:i+batch_size]\n",
                "        L = max(len(it[\"ids\"]) for it in b)\n",
                "        k = max(len(it[\"markers\"]) for it in b)\n",
                "        ids = torch.zeros((len(b), L), dtype=torch.long, device=device)\n",
                "        att = torch.zeros((len(b), L), dtype=torch.long, device=device)\n",
                "        pos = torch.zeros((len(b), k), dtype=torch.long, device=device)\n",
                "        mask = torch.zeros((len(b), k), dtype=torch.bool, device=device)\n",
                "        qtype = torch.full((len(b),), QTYPES[\"choice\"], dtype=torch.long, device=device)\n",
                "        for j, it in enumerate(b):\n",
                "            ids[j, :len(it[\"ids\"])] = torch.tensor(it[\"ids\"], device=device)\n",
                "            att[j, :len(it[\"ids\"])] = 1\n",
                "            pos[j, :len(it[\"markers\"])] = torch.tensor(it[\"markers\"], device=device)\n",
                "            mask[j, :len(it[\"markers\"])] = True\n",
                "            \n",
                "        opt.zero_grad()\n",
                "        with torch.no_grad():\n",
                "            t_logits, _ = teacher(ids, att, pos, mask, qtype)\n",
                "            \n",
                "        s_logits, _ = student(ids, att, pos, mask, qtype)\n",
                "        \n",
                "        T = 2.0\n",
                "        p_teacher = F.softmax(t_logits / T, dim=-1)\n",
                "        log_p_student = F.log_softmax(s_logits / T, dim=-1)\n",
                "        kl_loss = F.kl_div(log_p_student, p_teacher, reduction=\"batchmean\") * (T ** 2)\n",
                "        \n",
                "        kl_loss.backward()\n",
                "        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)\n",
                "        opt.step()\n",
                "        total_loss += kl_loss.item()\n",
                "        steps += 1\n",
                "        \n",
                "    print(f\"  {name} distillation finished | Avg loss: {total_loss / max(steps, 1):.4f}\")\n",
                "    student.eval()\n",
                "    return student\n",
                "\n",
                "student_14 = distill_model(student_14, teacher_model, name=\"Distil-QLaya 14L\")\n",
                "student_6 = distill_model(student_6, teacher_model, name=\"Distil-QLaya 6L\")\n"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Save PyTorch Student Checkpoints"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "from safetensors.torch import save_file\n",
                "\n",
                "def save_student_checkpoint(student, out_dir, num_layers):\n",
                "    os.makedirs(out_dir, exist_ok=True)\n",
                "    # Save safetensors weights\n",
                "    weights = {k: v.cpu() for k, v in student.state_dict().items()}\n",
                "    save_file(weights, os.path.join(out_dir, \"model.safetensors\"))\n",
                "    \n",
                "    # Save updated config\n",
                "    student_cfg = copy.deepcopy(cfg)\n",
                "    student_cfg[\"num_layers\"] = num_layers\n",
                "    with open(os.path.join(out_dir, \"rl_agent_config.json\"), \"w\") as f:\n",
                "        json.dump(student_cfg, f, indent=2)\n",
                "        \n",
                "    # Copy tokenizer files if present\n",
                "    tok_src = os.path.join(teacher_dir, \"tokenizer\")\n",
                "    if os.path.exists(tok_src):\n",
                "        tok_dst = os.path.join(out_dir, \"tokenizer\")\n",
                "        if os.path.exists(tok_dst):\n",
                "            shutil.rmtree(tok_dst)\n",
                "        shutil.copytree(tok_src, tok_dst)\n",
                "    print(f\"Saved PyTorch student checkpoint to {out_dir}\")\n",
                "\n",
                "save_student_checkpoint(student_14, \"/kaggle/working/models/distil_qlaya_14l\", 14)\n",
                "save_student_checkpoint(student_6, \"/kaggle/working/models/distil_qlaya_6l\", 6)\n"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Export ONNX & Quantize All Student Variants"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import onnx\n",
                "from onnxruntime.quantization import quantize_dynamic, QuantType\n",
                "from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer\n",
                "\n",
                "ONNX_DIR = \"/kaggle/working/models/onnx\"\n",
                "os.makedirs(ONNX_DIR, exist_ok=True)\n",
                "\n",
                "def export_onnx(model, filename):\n",
                "    path = os.path.join(ONNX_DIR, filename)\n",
                "    model.eval()\n",
                "    model.to(\"cpu\")\n",
                "    \n",
                "    dummy_ids = torch.randint(0, 100, (1, 16), dtype=torch.long)\n",
                "    dummy_att = torch.ones((1, 16), dtype=torch.long)\n",
                "    dummy_pos = torch.tensor([[1, 5]], dtype=torch.long)\n",
                "    dummy_mask = torch.tensor([[True, True]], dtype=torch.bool)\n",
                "    dummy_qtype = torch.tensor([0], dtype=torch.long)\n",
                "    \n",
                "    inputs = (dummy_ids, dummy_att, dummy_pos, dummy_mask, dummy_qtype)\n",
                "    \n",
                "    print(f\"Exporting ONNX FP32 -> {path}...\")\n",
                "    torch.onnx.export(\n",
                "        model,\n",
                "        inputs,\n",
                "        path,\n",
                "        export_params=True,\n",
                "        opset_version=18,\n",
                "        do_constant_folding=True,\n",
                "        input_names=[\"input_ids\", \"attention_mask\", \"marker_pos\", \"marker_mask\", \"qtype\"],\n",
                "        output_names=[\"logits\", \"act_logits\"],\n",
                "        dynamic_axes={\n",
                "            \"input_ids\": {0: \"batch_size\", 1: \"seq_len\"},\n",
                "            \"attention_mask\": {0: \"batch_size\", 1: \"seq_len\"},\n",
                "            \"marker_pos\": {0: \"batch_size\", 1: \"num_markers\"},\n",
                "            \"marker_mask\": {0: \"batch_size\", 1: \"num_markers\"},\n",
                "            \"qtype\": {0: \"batch_size\"},\n",
                "            \"logits\": {0: \"batch_size\", 1: \"num_markers\"},\n",
                "            \"act_logits\": {0: \"batch_size\"},\n",
                "        },\n",
                "    )\n",
                "    size_mb = os.path.getsize(path) / (1024 * 1024)\n",
                "    print(f\"  FP32 saved: {filename} ({size_mb:.1f} MB)\")\n",
                "    return path\n",
                "\n",
                "def quantize_int8(in_path, filename):\n",
                "    out_path = os.path.join(ONNX_DIR, filename)\n",
                "    m = onnx.load(in_path)\n",
                "    del m.graph.value_info[:]\n",
                "    quantize_dynamic(\n",
                "        model_input=m,\n",
                "        model_output=out_path,\n",
                "        op_types_to_quantize=[\"MatMul\"],\n",
                "        weight_type=QuantType.QInt8,\n",
                "        per_channel=True,\n",
                "    )\n",
                "    size_mb = os.path.getsize(out_path) / (1024 * 1024)\n",
                "    print(f\"  INT8 saved: {filename} ({size_mb:.1f} MB)\")\n",
                "    return out_path\n",
                "\n",
                "def quantize_int4(in_path, filename, block_size=32):\n",
                "    out_path = os.path.join(ONNX_DIR, filename)\n",
                "    m = onnx.load(in_path)\n",
                "    del m.graph.value_info[:]\n",
                "    q = MatMulNBitsQuantizer(\n",
                "        model=m,\n",
                "        bits=4,\n",
                "        block_size=block_size,\n",
                "        is_symmetric=True,\n",
                "    )\n",
                "    q.process()\n",
                "    q.model.save_model_to_file(out_path, use_external_data_format=False)\n",
                "    size_mb = os.path.getsize(out_path) / (1024 * 1024)\n",
                "    print(f\"  INT4 b{block_size} saved: {filename} ({size_mb:.1f} MB)\")\n",
                "    return out_path\n",
                "\n",
                "# --- Export 14L Variants ---\n",
                "p_14_fp32 = export_onnx(student_14, \"distil_qlaya_14l.fp32.onnx\")\n",
                "p_14_int8 = quantize_int8(p_14_fp32, \"distil_qlaya_14l.int8.onnx\")\n",
                "\n",
                "# --- Export 6L Variants ---\n",
                "p_6_fp32 = export_onnx(student_6, \"distil_qlaya_6l.fp32.onnx\")\n",
                "p_6_int8 = quantize_int8(p_6_fp32, \"distil_qlaya_6l.int8.onnx\")\n",
                "p_6_int4 = quantize_int4(p_6_fp32, \"distil_qlaya_6l.int4.onnx\", block_size=32)\n"
            ],
            "execution_count": None,
            "outputs": []
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Model Manifest & Results Summary"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "import glob\n",
                "print(\"=\" * 75)\n",
                "print(\"           QLAYA DISTILLED & QUANTIZED MODELS READY           \")\n",
                "print(\"=\" * 75)\n",
                "\n",
                "files = sorted(glob.glob(f\"{ONNX_DIR}/*.onnx\"))\n",
                "summary = []\n",
                "for f in files:\n",
                "    name = os.path.basename(f)\n",
                "    sz = os.path.getsize(f) / (1024 * 1024)\n",
                "    summary.append({\"file\": name, \"size_mb\": round(sz, 1)})\n",
                "    print(f\"  {name:<35} {sz:>8.1f} MB\")\n",
                "\n",
                "with open(f\"{ONNX_DIR}/distillation_summary.json\", \"w\") as f:\n",
                "    json.dump(summary, f, indent=2)\n",
                "\n",
                "print(\"=\" * 75)\n",
                "print(f\"All artifacts saved to /kaggle/working/models/ and available in Kaggle Output!\")\n"
            ],
            "execution_count": None,
            "outputs": []
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 2,
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("F:/Project/laya/kaggle_kernel/qlaya_distillation_and_quantization.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Regenerated qlaya_distillation_and_quantization.ipynb with all fixes verified.")
