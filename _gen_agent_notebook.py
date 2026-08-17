"""
Generates experiment7_agentic_freqscout.ipynb
"""
import json
from pathlib import Path

NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [],
}


def code(src, **meta):
    return {"cell_type": "code", "source": src, "metadata": meta,
            "outputs": [], "execution_count": None}


def md(src):
    return {"cell_type": "markdown", "source": src, "metadata": {}}


cells = [

md("""# Experiment 7 – Agentic Reasoning for bSSFP CINE Frequency-Offset Selection at 3.0T

**FYP: Agentic Reasoning for Enhanced Automatic Frequency Offset Selection in bSSFP CINE MRI at 3.0T**

This notebook implements the full agentic pipeline:
1. **T4 Neural Ranker** – DINOv2-L feature scoring
2. **T1 ROI Segmenter** – LV/myo mask from cleanest slice
3. **T2 Banding Detector** – CV banding detection per offset
4. **T3 Quality Metrics** – CoV, CNR, SNR per slice
5. **T5 Physics Descriptor** – offset Hz, band physics
6. **T6 VLM Descriptor** – MedGemma/Qwen2.5-VL banding description
7. **Retrieval** – BM25+FAISS knowledge retrieval
8. **Agentic Reasoning Loop** – gap analysis + consistency check (ADRE Algorithm 1)
9. **Stage-A Generation** – structured findings JSON
10. **Stage-B Generation** – clinical narrative in frequency-scouter language
11. **Repair** – faithfulness/hallucination check
12. **Evaluation** – Top-1/3 accuracy, confidence calibration, faithfulness

---
**Compatible with:** Local GPU (12GB) or Kaggle P100/T4.
**VLM:** Runs in mock mode if no GPU/VLM is available; swap `provider="hf"` for real inference.
"""),

code("""# ── Cell 1: Install dependencies ──────────────────────────────────────────────
import subprocess, sys

pkgs = [
    "pydicom", "openpyxl", "pandas", "pillow", "numpy", "scipy",
    "pyyaml", "tqdm", "seaborn", "matplotlib",
    "rank-bm25",          # BM25 retrieval
    "sentence-transformers",  # MiniLM-L6-v2 dense retrieval
    # Optional but recommended:
    # "faiss-cpu",          # FAISS indexing (install faiss-gpu on Kaggle)
    # "transformers",       # DINOv2 + VLM
    # "bitsandbytes",       # 4-bit quantisation
    # "accelerate",         # device_map="auto"
    # "pdfplumber",         # PDF parsing
]

for pkg in pkgs:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
print("Dependencies installed.")
"""),

code("""# ── Cell 2: Environment & paths ───────────────────────────────────────────────
import os, sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# Detect environment
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if IS_KAGGLE:
    DATA_ROOT  = Path("/kaggle/input/cine-test1/Dr-Dataset")
    EXCEL_PATH = Path("/kaggle/input/cine-test1/1stcohortSAX (1).xlsx")
    REPO_ROOT  = Path("/kaggle/input/cine-test1")
    OUT_DIR    = Path("/kaggle/working/agent_outputs")
else:
    DATA_ROOT  = Path("Dr-Dataset")
    EXCEL_PATH = Path("1stcohortSAX (1).xlsx")
    REPO_ROOT  = Path(".")
    OUT_DIR    = Path("agent_outputs")

OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO_ROOT))

print(f"DATA_ROOT  : {DATA_ROOT}  (exists={DATA_ROOT.exists()})")
print(f"EXCEL_PATH : {EXCEL_PATH}  (exists={EXCEL_PATH.exists()})")
print(f"OUT_DIR    : {OUT_DIR}")
"""),

code("""# ── Cell 3: Build Knowledge Base (run once) ────────────────────────────────────
from freqscout_agent.knowledge.parse_pdfs import parse_all
from freqscout_agent.knowledge.index import KnowledgeBaseBuilder

SPEC_PATH  = REPO_ROOT / "freqscout_agent/knowledge/clinical_spec.md"
PDF_DIR    = REPO_ROOT / "freqscout_agent/knowledge/pdfs"
INDEX_DIR  = REPO_ROOT / "freqscout_agent/knowledge/index_store"

chunks = parse_all(str(PDF_DIR), str(SPEC_PATH))
print(f"Total knowledge chunks: {len(chunks)}")

builder = KnowledgeBaseBuilder(index_dir=str(INDEX_DIR))
builder.build(chunks, force=False)   # set force=True to rebuild
print("Knowledge base ready.")
"""),

code("""# ── Cell 4: Load configuration & initialise pipeline ────────────────────────────
import yaml

CONFIG_PATH = REPO_ROOT / "freqscout_agent/configs/models.yaml"
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

# Override paths for this environment
cfg["paths"]["data_root"]   = str(DATA_ROOT)
cfg["paths"]["excel_path"]  = str(EXCEL_PATH)
cfg["paths"]["agent_outputs"] = str(OUT_DIR)

# For quick testing without GPU/VLM, the VLM falls back to mock responses.
# To use a real VLM:
#   cfg["vlm"]["provider"]  = "hf"
#   cfg["vlm"]["model_id"]  = "google/medgemma-4b-it"
# To use Ollama (local):
#   cfg["vlm"]["provider"]  = "ollama"
#   cfg["vlm"]["model_id"]  = "medgemma:4b"

from freqscout_agent.pipeline import FreqScoutPipeline
pipeline = FreqScoutPipeline(cfg, output_dir=str(OUT_DIR))
print("Pipeline initialised.")
"""),

code("""# ── Cell 5: Discover patients ─────────────────────────────────────────────────
patient_ids = pipeline._discover_patients(str(DATA_ROOT))
print(f"Found {len(patient_ids)} patients:")
for i, pid in enumerate(patient_ids):
    dcm_files = list((DATA_ROOT / pid).rglob("*.dcm"))
    print(f"  {i+1:2d}. {pid}  ({len(dcm_files)} DICOMs)")
"""),

code("""# ── Cell 6: Run pipeline on a single patient (demo) ──────────────────────────
import warnings
warnings.filterwarnings("ignore")

if patient_ids:
    DEMO_PID = patient_ids[0]
    print(f"\\nRunning agentic pipeline for: {DEMO_PID}")
    print("=" * 60)

    report = pipeline.run_patient(DEMO_PID, dicom_folder=str(DATA_ROOT / DEMO_PID))

    print("\\n--- RECOMMENDED OFFSET ---")
    rec = report.stage_a["recommended_offset"]
    print(f"  Slice {rec['slice_idx']} | {rec['offset_hz']:+.0f} Hz | "
          f"Confidence: {rec['confidence']:.3f} ({rec['confidence_label']})")

    print("\\n--- ALTERNATIVE OFFSETS ---")
    for a in report.stage_a["alternative_offsets"]:
        print(f"  Slice {a['slice_idx']} | {a['offset_hz']:+.0f} Hz | {a['verdict']}")

    print("\\n--- EVIDENCE AGREEMENT ---")
    print(f"  {report.stage_a['evidence_agreement']}")
    if report.consistency.disagreements:
        for d in report.consistency.disagreements:
            print(f"  ⚠ {d}")
else:
    print("No patients found in DATA_ROOT.")
"""),

code("""# ── Cell 7: Display Stage-A structured findings ───────────────────────────────
import json

if patient_ids:
    print("\\n=== STAGE-A: STRUCTURED FINDINGS (JSON) ===\\n")
    print(json.dumps(report.stage_a, indent=2)[:4000])   # cap for display
"""),

code("""# ── Cell 8: Display Stage-B clinical narrative ────────────────────────────────
if patient_ids:
    print("\\n=== STAGE-B: CLINICAL NARRATIVE ===\\n")
    print(report.narrative)

    print("\\n--- Repair report ---")
    r = report.repair_report
    print(f"  Was repaired: {r.was_repaired}")
    print(f"  Issues found: {len(r.issues_found)}")
    for issue in r.issues_found:
        print(f"    - {issue}")
"""),

code("""# ── Cell 9: Run pipeline on ALL patients ─────────────────────────────────────
from tqdm import tqdm

all_reports = []
all_proxy   = []

for pid in tqdm(patient_ids, desc="Patients"):
    try:
        rpt = pipeline.run_patient(pid, dicom_folder=str(DATA_ROOT / pid))
        all_reports.append(rpt)
        if rpt.proxy_result:
            all_proxy.append(rpt.proxy_result)
    except Exception as e:
        print(f"  ERROR for {pid}: {e}")

print(f"\\nCompleted {len(all_reports)}/{len(patient_ids)} patients.")
"""),

code("""# ── Cell 10: Aggregate & print proxy metrics ─────────────────────────────────
from freqscout_agent.eval.proxy_metrics import ProxyMetrics

if all_proxy:
    pm = ProxyMetrics()
    result = pm.compute(all_proxy)
    pm.print_summary(result)
    pm.save_adre_cases(result, str(OUT_DIR / "adre_cases.json"))
else:
    print("No GT available yet — proxy metrics not computed.")
    print("Tip: Ensure Excel has a 'best slice' column to enable proxy evaluation.")
"""),

code("""# ── Cell 11: Reasoning quality metrics ───────────────────────────────────────
from freqscout_agent.eval.reasoning_metrics import ReasoningMetrics

rm = ReasoningMetrics()
reasoning_results = []
for rpt in all_reports:
    res = rm.evaluate(
        narrative=rpt.narrative,
        stage_a=rpt.stage_a,
        doctor_gt=None,           # None until doctor GT arrives
        patient_id=rpt.patient_id,
    )
    reasoning_results.append(res)

rm.print_summary(reasoning_results)
"""),

code("""# ── Cell 12: VLM Ablation – Zero-shot vs Few-shot vs RAG-only ────────────────
# Run only if you want the ablation study. Requires VLM to be properly loaded.

from freqscout_agent.eval.vlm_ablation import VLMAblation

ablation = VLMAblation(
    vlm_tool=pipeline._tools.get("vlm"),
    retriever=pipeline._retriever,
)

if patient_ids and len(patient_ids) >= 3:
    ablation_pid = patient_ids[0]
    imgs_abl, off_abl = pipeline._load_dicom(ablation_pid, str(DATA_ROOT / ablation_pid))

    modes = {
        "zero_shot": ablation.run_zero_shot(imgs_abl, off_abl, ablation_pid),
        "few_shot":  ablation.run_few_shot(imgs_abl, off_abl, ablation_pid),
        "rag_only":  ablation.run_rag_only(imgs_abl, off_abl, ablation_pid),
    }
    print(f"\\n=== VLM ABLATION for {ablation_pid} ===")
    for mode, res in modes.items():
        print(f"  {mode:12s}: pred_slice={res.pred_slice} ({res.pred_hz:+.0f} Hz) conf={res.confidence:.2f}")
else:
    print("Ablation skipped (need ≥3 patients or no patient IDs found).")
"""),

code("""# ── Cell 13: Prepare LoRA dataset (for future fine-tuning) ──────────────────
# Builds training data from pipeline outputs.
# When doctor GT reasoning notes arrive, replace `report.narrative` with the doctor's text.

pipeline_outputs = [
    {
        "patient_id": rpt.patient_id,
        "stage_a": rpt.stage_a,
        "narrative": rpt.narrative,
        "image_paths": [],   # optionally fill with DICOM paths
    }
    for rpt in all_reports
]

from freqscout_agent.eval.vlm_ablation import VLMAblation
abl2 = VLMAblation()
abl2.prepare_lora_dataset(pipeline_outputs, output_dir=str(OUT_DIR / "lora_data"))
print(f"LoRA dataset prepared: {len(pipeline_outputs)} samples.")
print("Edit lora_data/lora_finetune.py to wire the Trainer and run on Kaggle GPU.")
"""),

code("""# ── Cell 14: Visualise per-patient results ────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if all_reports:
    pids   = [r.patient_id for r in all_reports]
    confs  = [r.stage_a["recommended_offset"]["confidence"] for r in all_reports]
    agree  = [{"full": 1.0, "partial": 0.5, "conflicted": 0.0}[r.stage_a["evidence_agreement"]]
              for r in all_reports]
    n_disc = [len(r.consistency.disagreements) for r in all_reports]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].bar(range(len(pids)), confs, color="steelblue")
    axes[0].axhline(0.75, color="green",  ls="--", label="High threshold")
    axes[0].axhline(0.45, color="orange", ls="--", label="Medium threshold")
    axes[0].set_title("Agent Confidence per Patient")
    axes[0].set_xlabel("Patient")
    axes[0].set_ylabel("Confidence")
    axes[0].set_xticks(range(len(pids)))
    axes[0].set_xticklabels([p[:6] for p in pids], rotation=45, ha="right")
    axes[0].legend(fontsize=7)

    colors = ["green" if a == 1.0 else ("orange" if a == 0.5 else "red") for a in agree]
    axes[1].bar(range(len(pids)), [1]*len(pids), color=colors)
    axes[1].set_title("Evidence Agreement\\n(green=full, orange=partial, red=conflicted)")
    axes[1].set_xticks(range(len(pids)))
    axes[1].set_xticklabels([p[:6] for p in pids], rotation=45, ha="right")
    axes[1].set_yticks([])

    axes[2].bar(range(len(pids)), n_disc, color="salmon")
    axes[2].set_title("# Disagreements per Patient")
    axes[2].set_xlabel("Patient")
    axes[2].set_ylabel("Count")
    axes[2].set_xticks(range(len(pids)))
    axes[2].set_xticklabels([p[:6] for p in pids], rotation=45, ha="right")

    plt.tight_layout()
    fig_path = OUT_DIR / "agent_summary.png"
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    print(f"Summary chart saved to {fig_path}")

    try:
        from IPython.display import Image, display
        display(Image(filename=str(fig_path)))
    except Exception:
        pass
else:
    print("No reports to visualise.")
"""),

code("""# ── Cell 15: Print loop trace for one patient ─────────────────────────────────
if all_reports:
    trace = all_reports[0].loop_trace
    print(f"\\n=== LOOP TRACE for {all_reports[0].patient_id} ===")
    print(f"  Terminated at iteration {trace.terminated_at} ({trace.termination_reason})")
    for it in trace.iterations:
        status = "✓" if it["next_tool"] is None else f"→ {it['next_tool']}"
        print(f"  Iter {it['iteration']:2d}: unsatisfied={it['unsatisfied']}  {status}")
    print(f"  Consistency: neural={trace.consistency_report.neural_top1} "
          f"cv={trace.consistency_report.cv_top1} "
          f"vlm={trace.consistency_report.vlm_top1} "
          f"conf={trace.consistency_report.confidence:.3f}")
"""),

]

NB["cells"] = cells

out = Path("experiment7_agentic_freqscout.ipynb")
out.write_text(json.dumps(NB, indent=1))
print(f"Generated {out}")
