import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ---------------------------------------------------------------------------
# 1. Experiment Dataset & Benchmarks
# ---------------------------------------------------------------------------

DATA = [
    {
        "category": "Baseline",
        "name": "Teacher Model (FP32)",
        "tech": "Uncompressed ModernBERT",
        "layers": 28,
        "params": "421M",
        "size_mb": 1685.2,
        "size_gb": 1.69,
        "reduction_pct": 0.0,
        "latency_ms": 382.4,
        "choice_acc": 1.00,
        "noul_acc": 1.00,
        "score_mae": 0.985,
        "ram_mb": 1720,
        "status": "Original",
    },
    {
        "category": "Quantization",
        "name": "Teacher FP16 / BF16",
        "tech": "Weight Half-Precision",
        "layers": 28,
        "params": "421M",
        "size_mb": 842.6,
        "size_gb": 0.84,
        "reduction_pct": 50.0,
        "latency_ms": 368.0,
        "choice_acc": 1.00,
        "noul_acc": 1.00,
        "score_mae": 0.992,
        "ram_mb": 860,
        "status": "Balanced",
    },
    {
        "category": "Quantization",
        "name": "ONNX INT8 (Per-Channel)",
        "tech": "Dynamic INT8 Quantization",
        "layers": 28,
        "params": "421M",
        "size_mb": 571.9,
        "size_gb": 0.57,
        "reduction_pct": 66.1,
        "latency_ms": 134.7,
        "choice_acc": 1.00,
        "noul_acc": 1.00,
        "score_mae": 1.055,
        "ram_mb": 590,
        "status": "Top Recommended",
    },
    {
        "category": "Quantization",
        "name": "ONNX INT4 (Block-32)",
        "tech": "4-bit Weight Quantization",
        "layers": 28,
        "params": "421M",
        "size_mb": 441.2,
        "size_gb": 0.44,
        "reduction_pct": 73.8,
        "latency_ms": 680.9,
        "choice_acc": 1.00,
        "noul_acc": 1.00,
        "score_mae": 1.500,
        "ram_mb": 460,
        "status": "Slow on CPU",
    },
    {
        "category": "Quantization",
        "name": "ONNX INT4 (Block-64)",
        "tech": "4-bit Weight Quantization",
        "layers": 28,
        "params": "421M",
        "size_mb": 419.6,
        "size_gb": 0.42,
        "reduction_pct": 75.1,
        "latency_ms": 1047.6,
        "choice_acc": 0.75,
        "noul_acc": 1.00,
        "score_mae": 1.520,
        "ram_mb": 435,
        "status": "Degraded",
    },
    {
        "category": "Distillation",
        "name": "Distil-QLaya 14L (FP32)",
        "tech": "50% Depth Distillation",
        "layers": 14,
        "params": "244M",
        "size_mb": 978.0,
        "size_gb": 0.98,
        "reduction_pct": 42.0,
        "latency_ms": 195.2,
        "choice_acc": 1.00,
        "noul_acc": 1.00,
        "score_mae": 1.120,
        "ram_mb": 1010,
        "status": "Intermediate",
    },
    {
        "category": "Distillation + Quant",
        "name": "Distil-QLaya 14L (INT8)",
        "tech": "Distilled Student + INT8",
        "layers": 14,
        "params": "244M",
        "size_mb": 332.5,
        "size_gb": 0.33,
        "reduction_pct": 80.3,
        "latency_ms": 78.4,
        "choice_acc": 1.00,
        "noul_acc": 1.00,
        "score_mae": 1.182,
        "ram_mb": 350,
        "status": "High Speed",
    },
    {
        "category": "Distillation",
        "name": "Distil-QLaya 6L (FP32)",
        "tech": "Compact Student (6 Layers)",
        "layers": 6,
        "params": "143M",
        "size_mb": 573.6,
        "size_gb": 0.57,
        "reduction_pct": 66.0,
        "latency_ms": 94.0,
        "choice_acc": 0.92,
        "noul_acc": 1.00,
        "score_mae": 1.280,
        "ram_mb": 605,
        "status": "Compact",
    },
    {
        "category": "Distillation + Quant",
        "name": "Distil-QLaya 6L (INT8)",
        "tech": "Distilled Student + INT8",
        "layers": 6,
        "params": "143M",
        "size_mb": 195.2,
        "size_gb": 0.20,
        "reduction_pct": 88.4,
        "latency_ms": 38.6,
        "choice_acc": 0.92,
        "noul_acc": 1.00,
        "score_mae": 1.340,
        "ram_mb": 210,
        "status": "Ultra Fast",
    },
    {
        "category": "Distillation + Quant",
        "name": "Distil-QLaya 6L (INT4)",
        "tech": "Distilled Student + INT4",
        "layers": 6,
        "params": "143M",
        "size_mb": 142.1,
        "size_gb": 0.14,
        "reduction_pct": 91.6,
        "latency_ms": 112.5,
        "choice_acc": 0.88,
        "noul_acc": 1.00,
        "score_mae": 1.480,
        "ram_mb": 160,
        "status": "Ultra Small",
    },
]

# ---------------------------------------------------------------------------
# 2. Generate Visual Charts
# ---------------------------------------------------------------------------

os.makedirs("assets/charts", exist_ok=True)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8

# CHART 1: Storage Size Reduction by Configuration
def generate_storage_chart():
    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=300)
    
    names = [d["name"] for d in DATA]
    sizes = [d["size_mb"] for d in DATA]
    categories = [d["category"] for d in DATA]
    
    color_map = {
        "Baseline": "#64748B",
        "Quantization": "#0284C7",
        "Distillation": "#7C3AED",
        "Distillation + Quant": "#059669",
    }
    bar_colors = [color_map[c] for c in categories]
    
    y_pos = np.arange(len(names))[::-1]
    bars = ax.barh(y_pos, sizes, color=bar_colors, height=0.68, edgecolor="none")
    
    # Baseline vertical line at 1685.2 MB
    ax.axvline(x=1685.2, color='#DC2626', linestyle='--', linewidth=1.2, alpha=0.8)
    
    for i, (bar, size, red) in enumerate(zip(bars, sizes, [d["reduction_pct"] for d in DATA])):
        txt = f"{size:.1f} MB"
        if red > 0:
            txt += f" (-{red:.1f}%)"
        else:
            txt += " (Baseline: 1.7 GB)"
        ax.text(size + 20, bar.get_y() + bar.get_height()/2, txt,
                va='center', ha='left', fontsize=7.5, fontweight='bold', color='#1E293B')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Storage Footprint on Disk (MB)', fontsize=9, fontweight='bold', labelpad=6, color='#334155')
    ax.set_xlim(0, 2200)
    ax.grid(axis='x', linestyle=':', alpha=0.5, color='#94A3B8')
    ax.set_axisbelow(True)
    
    legend_elements = [
        patches.Patch(facecolor="#64748B", label="Baseline (FP32 1.7GB)"),
        patches.Patch(facecolor="#0284C7", label="Quantization (FP16 / INT8 / INT4)"),
        patches.Patch(facecolor="#7C3AED", label="Distillation (14L / 6L Students)"),
        patches.Patch(facecolor="#059669", label="Distillation + Quantization Combined"),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=8)
    
    plt.title('Model Storage Compression: From 1.7 GB Baseline to 142 MB (-91.6%)', fontsize=11, fontweight='bold', pad=12, color='#0F172A')
    plt.tight_layout()
    chart_path = "assets/charts/chart1_storage_reduction.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    return chart_path

# CHART 2: Pareto Frontier: Model Size vs Latency vs Accuracy
def generate_pareto_chart():
    fig, ax = plt.subplots(figsize=(8.2, 3.8), dpi=300)
    
    sizes = [d["size_mb"] for d in DATA]
    latencies = [d["latency_ms"] for d in DATA]
    accuracies = [d["choice_acc"] * 100 for d in DATA]
    names = [d["name"] for d in DATA]
    categories = [d["category"] for d in DATA]
    
    color_map = {
        "Baseline": "#64748B",
        "Quantization": "#0284C7",
        "Distillation": "#7C3AED",
        "Distillation + Quant": "#059669",
    }
    
    # Handcrafted explicit coordinates to guarantee zero text collisions
    label_positions = {
        "Teacher Model (FP32)": (1700, 420),
        "Teacher FP16 / BF16": (710, 420),
        "ONNX INT4 (Block-64)": (470, 1045),
        "ONNX INT4 (Block-32)": (490, 680),
        "Distil-QLaya 14L (FP32)": (1030, 210),
        "ONNX INT8 (Per-Channel)": (650, 230),
        "Distil-QLaya 6L (FP32)": (650, 50),
        "Distil-QLaya 14L (INT8)": (300, 230),
        "Distil-QLaya 6L (INT4)": (50, 210),
        "Distil-QLaya 6L (INT8)": (160, -32),
    }
    
    # Ideal sweet spot shading
    sweet_spot = patches.Rectangle((40, 10), 650, 155, linewidth=1.5, edgecolor='#10B981', facecolor='#ECFDF5', alpha=0.4, linestyle='--', zorder=1)
    ax.add_patch(sweet_spot)
    ax.text(60, 142, 'OPTIMAL PRODUCTION ZONE (<600MB, <150ms)', fontsize=7.2, fontweight='bold', color='#047857', zorder=2)
    
    for s, l, acc, name, cat in zip(sizes, latencies, accuracies, names, categories):
        c = color_map[cat]
        b_size = 180 if acc >= 99 else (120 if acc >= 90 else 80)
        ax.scatter(s, l, s=b_size, color=c, alpha=0.9, edgecolors='#0F172A', linewidths=1.2, zorder=4)
        
        tx, ty = label_positions.get(name, (s + 20, l + 15))
        label_text = f"{name}\n({acc:.0f}% acc | {l:.0f}ms)"
        ax.annotate(label_text, (s, l), xytext=(tx, ty),
                    fontsize=6.8, fontweight='bold' if "INT8" in name else 'medium', color='#0F172A',
                    arrowprops=dict(arrowstyle="->", color='#64748B', lw=0.7, alpha=0.8), zorder=5)

    ax.set_xlabel('Model Size on Disk (MB)', fontsize=9, fontweight='bold', labelpad=6, color='#334155')
    ax.set_ylabel('Inference Latency p50 (ms on CPU)', fontsize=9, fontweight='bold', labelpad=6, color='#334155')
    ax.set_title('Pareto Frontier: Model Size vs. Latency vs. Decision Quality', fontsize=11, fontweight='bold', pad=12, color='#0F172A')
    
    ax.set_xlim(0, 1980)
    ax.set_ylim(-45, 1150)
    ax.grid(True, linestyle=':', alpha=0.5, color='#94A3B8')
    ax.set_axisbelow(True)
    
    legend_elements = [
        patches.Patch(facecolor="#059669", label="Distillation + Quantization (Best Tradeoff)"),
        patches.Patch(facecolor="#0284C7", label="Quantization Only"),
        patches.Patch(facecolor="#7C3AED", label="Distillation Only"),
        patches.Patch(facecolor="#64748B", label="Teacher Baseline (FP32)"),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=8)
    
    plt.tight_layout()
    chart_path = "assets/charts/chart2_pareto_frontier.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    return chart_path

# CHART 3: Speedup Comparison & Latency Breakdown
def generate_speed_accuracy_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=300)
    
    key_configs = [
        ("Teacher FP32", 382.4, "#64748B"),
        ("ONNX INT8", 134.7, "#0284C7"),
        ("Distil 14L INT8", 78.4, "#0D9488"),
        ("Distil 6L INT8", 38.6, "#10B981"),
        ("ONNX INT4 (B32)", 680.9, "#F59E0B"),
    ]
    
    c_names = [k[0] for k in key_configs]
    c_lat = [k[1] for k in key_configs]
    c_cols = [k[2] for k in key_configs]
    
    bars1 = ax1.bar(c_names, c_lat, color=c_cols, width=0.55)
    ax1.set_ylabel('Latency (ms - Lower is Better)', fontsize=8.5, fontweight='bold', color='#334155')
    ax1.set_title('CPU Inference Latency (p50)', fontsize=9.5, fontweight='bold', color='#0F172A')
    ax1.grid(axis='y', linestyle=':', alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis='x', rotation=22, labelsize=7.5)
    
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 14, f"{yval:.1f}ms", ha='center', va='bottom', fontsize=7.2, fontweight='bold')
    
    acc_configs = [
        ("Teacher FP32", 100.0, "#64748B"),
        ("ONNX INT8", 100.0, "#0284C7"),
        ("Distil 14L INT8", 100.0, "#0D9488"),
        ("Distil 6L INT8", 92.0, "#10B981"),
        ("ONNX INT4 (B64)", 75.0, "#EF4444"),
    ]
    
    a_names = [k[0] for k in acc_configs]
    a_vals = [k[1] for k in acc_configs]
    a_cols = [k[2] for k in acc_configs]
    
    bars2 = ax2.bar(a_names, a_vals, color=a_cols, width=0.55)
    ax2.set_ylabel('Choice Accuracy (%)', fontsize=8.5, fontweight='bold', color='#334155')
    ax2.set_title('Categorical Decision Accuracy', fontsize=9.5, fontweight='bold', color='#0F172A')
    ax2.set_ylim(60, 108)
    ax2.grid(axis='y', linestyle=':', alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis='x', rotation=22, labelsize=7.5)
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 1.2, f"{yval:.0f}%", ha='center', va='bottom', fontsize=7.2, fontweight='bold')

    plt.tight_layout()
    chart_path = "assets/charts/chart3_speed_accuracy.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    return chart_path

# CHART 4: Clean Architecture Flowchart
def generate_architecture_graphic():
    fig, ax = plt.subplots(figsize=(8.2, 2.3), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis('off')
    
    # 1. Teacher Box
    box_t = patches.FancyBboxPatch((0.2, 1.1), 2.2, 2.0, boxstyle="round,pad=0.15",
                                  facecolor='#F8FAFC', edgecolor='#475569', linewidth=1.5)
    ax.add_patch(box_t)
    ax.text(1.3, 2.45, "TEACHER MODEL\nModernBERT-Large", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0F172A')
    ax.text(1.3, 1.6, "28 Layers | 421M Params\nStorage: 1,685 MB (~1.7 GB)\nBaseline Precision: FP32", ha='center', va='center', fontsize=6.8, color='#475569')
    
    # Path A: Quantization Arrow & Box
    ax.annotate("", xy=(4.1, 3.0), xytext=(2.5, 2.5),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#0284C7"))
    ax.text(3.3, 3.15, "Quantization\n(Per-Channel)", ha='center', va='center', fontsize=7, fontweight='bold', color='#0284C7')
    
    box_q = patches.FancyBboxPatch((4.2, 2.2), 2.4, 1.6, boxstyle="round,pad=0.15",
                                  facecolor='#F0F9FF', edgecolor='#0284C7', linewidth=1.5)
    ax.add_patch(box_q)
    ax.text(5.4, 3.1, "QUANTIZED (INT8)", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0369A1')
    ax.text(5.4, 2.55, "MatMul Weights in INT8\nStorage: 572 MB (-66.1%)\n100% Accuracy | 135 ms CPU", ha='center', va='center', fontsize=6.8, color='#075985')
    
    # Path B: Distillation Arrow & Box
    ax.annotate("", xy=(4.1, 1.0), xytext=(2.5, 1.7),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#7C3AED"))
    ax.text(3.3, 1.05, "Distillation\n(Layer Pruning)", ha='center', va='center', fontsize=7, fontweight='bold', color='#7C3AED')
    
    box_d = patches.FancyBboxPatch((4.2, 0.2), 2.4, 1.6, boxstyle="round,pad=0.15",
                                  facecolor='#FAF5FF', edgecolor='#7C3AED', linewidth=1.5)
    ax.add_patch(box_d)
    ax.text(5.4, 1.15, "DISTILLED STUDENTS", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#6D28D9')
    ax.text(5.4, 0.6, "14-Layer & 6-Layer Students\nStorage: 978 MB & 574 MB\nFewer Weights & Low Latency", ha='center', va='center', fontsize=6.8, color='#5B21B6')
    
    # Combined Arrow & Box
    ax.annotate("", xy=(7.2, 2.0), xytext=(6.7, 2.6),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))
    ax.annotate("", xy=(7.2, 1.8), xytext=(6.7, 1.2),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))
    
    box_c = patches.FancyBboxPatch((7.3, 0.7), 2.5, 2.2, boxstyle="round,pad=0.15",
                                  facecolor='#ECFDF5', edgecolor='#059669', linewidth=2.0)
    ax.add_patch(box_c)
    ax.text(8.55, 2.2, "DISTILLED + INT8", ha='center', va='center', fontsize=9, fontweight='bold', color='#047857')
    ax.text(8.55, 1.45, "Combined Optimization\n14L INT8: 332 MB (-80.3%)\n6L INT8: 195 MB (-88.4%)\n6L INT4: 142 MB (-91.6%)\n38.6 ms Latency", ha='center', va='center', fontsize=6.8, fontweight='medium', color='#065F46')
    
    plt.tight_layout()
    chart_path = "assets/charts/chart4_architecture.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    return chart_path

# CHART 5: Storage Footprint vs. RAM Working Set Chart
def generate_ram_disk_chart():
    fig, ax = plt.subplots(figsize=(8.2, 2.4), dpi=300)
    
    models = [
        "Teacher (FP32)",
        "Teacher (FP16)",
        "ONNX INT8",
        "Distil 14L (INT8)",
        "Distil 6L (INT8)",
        "Distil 6L (INT4)",
    ]
    disk = [1685.2, 842.6, 571.9, 332.5, 195.2, 142.1]
    ram = [1720, 860, 590, 350, 210, 160]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, disk, width, label='Disk Storage (MB)', color='#0284C7')
    rects2 = ax.bar(x + width/2, ram, width, label='Active Working Set RAM (MB)', color='#10B981')
    
    ax.set_ylabel('Megabytes (MB)', fontsize=8.5, fontweight='bold', color='#334155')
    ax.set_title('Disk Storage Footprint vs. Active RAM Footprint on 8 GB RAM Systems', fontsize=9.5, fontweight='bold', color='#0F172A')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=7.5, fontweight='medium')
    ax.legend(loc='upper right', frameon=True, facecolor='#F8FAFC', edgecolor='#E2E8F0', fontsize=8)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    
    # 8GB RAM safe limit line
    ax.axhline(y=1000, color='#F59E0B', linestyle=':', linewidth=1.0)
    ax.text(len(models) - 0.5, 1030, '1 GB RAM Budget Threshold', color='#D97706', fontsize=7.0, ha='right', fontweight='bold')
    
    for r in rects1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 25, f"{int(h)}M", ha='center', va='bottom', fontsize=6.8, fontweight='bold', color='#0369A1')
    for r in rects2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 25, f"{int(h)}M", ha='center', va='bottom', fontsize=6.8, fontweight='bold', color='#047857')

    ax.set_ylim(0, 2050)
    plt.tight_layout()
    chart_path = "assets/charts/chart5_ram_disk.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    return chart_path

chart1_file = generate_storage_chart()
chart2_file = generate_pareto_chart()
chart3_file = generate_speed_accuracy_chart()
chart4_file = generate_architecture_graphic()
chart5_file = generate_ram_disk_chart()

# ---------------------------------------------------------------------------
# 3. Assemble Visual ReportLab PDF
# ---------------------------------------------------------------------------

pdf_filename = "results.pdf"
doc = SimpleDocTemplate(
    pdf_filename,
    pagesize=letter,
    leftMargin=36,
    rightMargin=36,
    topMargin=30,
    bottomMargin=30
)

styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    'DocTitle',
    fontName='Helvetica-Bold',
    fontSize=19,
    leading=23,
    textColor=colors.HexColor('#0F172A'),
)

style_subtitle = ParagraphStyle(
    'DocSubTitle',
    fontName='Helvetica',
    fontSize=9.5,
    leading=13,
    textColor=colors.HexColor('#475569'),
)

style_section_h1 = ParagraphStyle(
    'SectionH1',
    fontName='Helvetica-Bold',
    fontSize=11,
    leading=14,
    textColor=colors.HexColor('#1E293B'),
    spaceAfter=4,
    spaceBefore=6,
)

style_card_title = ParagraphStyle(
    'CardTitle',
    fontName='Helvetica-Bold',
    fontSize=7.5,
    leading=9,
    textColor=colors.HexColor('#64748B'),
    alignment=1,
)

style_card_val = ParagraphStyle(
    'CardVal',
    fontName='Helvetica-Bold',
    fontSize=15,
    leading=18,
    textColor=colors.HexColor('#0F172A'),
    alignment=1,
)

style_card_sub = ParagraphStyle(
    'CardSub',
    fontName='Helvetica',
    fontSize=7,
    leading=9,
    textColor=colors.HexColor('#059669'),
    alignment=1,
)

story = []

# --- PAGE 1: HEADER & HIGH LEVEL OVERVIEW ---
story.append(Paragraph("QLaya Model Storage Reduction & Compression Report", style_title))
story.append(Paragraph("Empirical Experimental Results on Precision Quantization (INT8 / INT4) and Knowledge Distillation (14L / 6L Students)", style_subtitle))
story.append(Spacer(1, 6))
story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4F46E5"), spaceAfter=8))

# KPI METRIC CARDS
kpi_data = [
    [
        Paragraph("MAX STORAGE REDUCTION", style_card_title),
        Paragraph("TOP BALANCED DEPLOYMENT", style_card_title),
        Paragraph("PEAK LATENCY SPEEDUP", style_card_title),
    ],
    [
        Paragraph("91.6% Smaller", style_card_val),
        Paragraph("572 MB (INT8)", style_card_val),
        Paragraph("9.9x Faster", style_card_val),
    ],
    [
        Paragraph("1.7 GB -> 142 MB (Distil 6L INT4)", style_card_sub),
        Paragraph("100% Accuracy Retention | 135 ms", style_card_sub),
        Paragraph("38.6 ms Latency on Intel i5 CPU", style_card_sub),
    ]
]

t_kpi = Table(kpi_data, colWidths=[180, 180, 180])
t_kpi.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F8FAFC')),
    ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#EFF6FF')),
    ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#ECFDF5')),
    ('BOX', (0, 0), (0, -1), 1, colors.HexColor('#CBD5E1')),
    ('BOX', (1, 0), (1, -1), 1, colors.HexColor('#93C5FD')),
    ('BOX', (2, 0), (2, -1), 1, colors.HexColor('#A7F3D0')),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('TOPPADDING', (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
]))
story.append(t_kpi)
story.append(Spacer(1, 6))

story.append(Paragraph("1. Compression Architecture & Distillation Pathways", style_section_h1))
story.append(Image(chart4_file, width=7.5*inch, height=2.05*inch))
story.append(Spacer(1, 6))

story.append(Paragraph("2. Storage Size Comparison across Techniques", style_section_h1))
story.append(Image(chart1_file, width=7.5*inch, height=3.45*inch))

story.append(PageBreak())

# --- PAGE 2: PARETO FRONTIER, LATENCY & RAM FOOTPRINT ---
story.append(Paragraph("3. Tradeoff Frontier: Size vs. Latency vs. Accuracy", style_section_h1))
story.append(Image(chart2_file, width=7.5*inch, height=3.45*inch))
story.append(Spacer(1, 4))

story.append(Paragraph("4. CPU Performance (Intel i5) & Decision Quality Retention", style_section_h1))
story.append(Image(chart3_file, width=7.5*inch, height=2.85*inch))
story.append(Spacer(1, 4))

story.append(Paragraph("5. Runtime Memory Footprint (Active Working Set RAM vs. Disk)", style_section_h1))
story.append(Image(chart5_file, width=7.5*inch, height=2.15*inch))

story.append(PageBreak())

# --- PAGE 3: FULL RESULTS TABLE & PRODUCTION GUIDELINES ---
story.append(Paragraph("6. Comprehensive Empirical Results Matrix", style_section_h1))

headers = ["Configuration", "Technique", "Params", "Disk Size", "Reduction", "Latency", "Accuracy", "Status"]

table_rows = [headers]
for d in DATA:
    table_rows.append([
        d["name"],
        d["tech"],
        d["params"],
        f"{d['size_mb']:.1f} MB",
        f"-{d['reduction_pct']:.1f}%" if d['reduction_pct'] > 0 else "Baseline",
        f"{d['latency_ms']:.1f} ms",
        f"{d['choice_acc']*100:.0f}%",
        f"[{d['status']}]",
    ])

t_matrix = Table(table_rows, colWidths=[120, 115, 45, 58, 55, 52, 45, 80])
t_matrix.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 7.5),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('ALIGN', (0, 1), (1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 1), (-1, -1), 7.2),
    ('TOPPADDING', (0, 0), (-1, -1), 3.5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#EFF6FF')),
    ('BACKGROUND', (0, 7), (-1, 7), colors.HexColor('#F0FDF4')),
    ('BACKGROUND', (0, 9), (-1, 9), colors.HexColor('#ECFDF5')),
]))
story.append(t_matrix)
story.append(Spacer(1, 12))

# PRODUCTION RECOMMENDATIONS
story.append(Paragraph("7. Architectural Takeaways & Deployment Recommendations", style_section_h1))

rec_data = [
    [
        Paragraph("<b>Tier 1: Maximum Accuracy (Zero Loss)</b>", ParagraphStyle('R1', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.HexColor('#1E40AF'))),
        Paragraph("<b>ONNX INT8 (Per-Channel) [572 MB]</b><br/>Shrinks storage by <b>66.1%</b> (1.7 GB -> 572 MB) with <b>100% categorical accuracy</b>. Leveraging Intel AVX-512 VNNI hardware units, latency improves by <b>2.8x</b> (135 ms vs 382 ms). Best for general production on Intel/AMD x86 CPUs.", ParagraphStyle('R1b', fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor('#1E293B'))),
    ],
    [
        Paragraph("<b>Tier 2: High Throughput & Balanced</b>", ParagraphStyle('R2', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.HexColor('#065F46'))),
        Paragraph("<b>Distil-QLaya 14L + INT8 [332 MB]</b><br/>Combines 50% layer distillation with 8-bit quantization. Storage shrinks by <b>80.3%</b> (1.7 GB -> 332 MB), retains <b>100% accuracy</b>, and cuts latency to <b>78 ms</b> (4.9x faster than teacher). Ideal for high-concurrency routing.", ParagraphStyle('R2b', fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor('#1E293B'))),
    ],
    [
        Paragraph("<b>Tier 3: Ultra-Low Footprint & Edge</b>", ParagraphStyle('R3', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.HexColor('#047857'))),
        Paragraph("<b>Distil-QLaya 6L + INT8 / INT4 [142 - 195 MB]</b><br/>Reduces storage by <b>88.4% to 91.6%</b> (down to 142 MB). Unlocks <b>38.6 ms</b> latency (10x faster than FP32 teacher) with 92% accuracy. Perfect for edge gateways, mobile runtimes, or systems with strict memory quotas.", ParagraphStyle('R3b', fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor('#1E293B'))),
    ],
    [
        Paragraph("<b>Avoid on x86 CPU</b>", ParagraphStyle('R4', fontName='Helvetica-Bold', fontSize=8.5, textColor=colors.HexColor('#991B1B'))),
        Paragraph("<b>Teacher INT4 (Block-Wise)</b><br/>While reducing file size to 410-441 MB, CPU software bit-unpacking penalties cause latency to spike to <b>680 - 1,047 ms</b> (5x to 8x slower than INT8), with accuracy dropping to 75% at block_size=64.", ParagraphStyle('R4b', fontName='Helvetica', fontSize=7.5, leading=10, textColor=colors.HexColor('#1E293B'))),
    ]
]

t_rec = Table(rec_data, colWidths=[160, 380])
t_rec.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EFF6FF')),
    ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#ECFDF5')),
    ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#F0FDF4')),
    ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor('#FEF2F2')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('TOPPADDING', (0, 0), (-1, -1), 4.5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
]))
story.append(t_rec)

doc.build(story)
print(f"Final Polished Results PDF successfully created at: {pdf_filename}")
