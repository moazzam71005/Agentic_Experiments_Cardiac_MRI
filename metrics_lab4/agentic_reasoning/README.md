# Experiment 15: Multimodal VLM Grounded Scouting & Visual Inspection

Self-contained deployment package for **Experiment 15** of the FYP:
*Agentic Reasoning for Enhanced Automatic Frequency Selection of bSSFP CINE MRI at 3.0T*.

---

## 🌟 Key Innovations in Exp 15

1. **Visual Grounding (Overcoming the "Blind LLM" Bottleneck)**:
   - In previous experiments (Exp 13 & 14), the LLMs were text-only, reasoning purely over tables of summary statistics.
   - Exp 15 directly attaches **pre-rendered multi-offset visual scout strips and zoomed myocardial ROI images (with green anatomical contour overlays)** to a Vision-Language Model.
   - The VLM visually inspects the myocardial wall (anterior, lateral, inferior, and septal sectors) to confirm whether dark banding artifact lines intersect cardiac tissue.

2. **Multimodal VLM Inspection (`core/vlm_inspector.py`)**:
   - Gated triggering on ambiguous cases (`wide_plateau`, `borderline_veto`, `divergent`).
   - The VLM examines the visual strip and selects the genuinely cleanest, artifact-free frame.

3. **Strict Physical Guardrails & Graceful Fallback**:
   - Any frame with Banded Fraction Metric $\text{BFM} > 0.08$ is automatically vetoed by the physics safety harness.
   - Deterministic composite score fallback guarantees 0% catastrophic errors.

---

## 🚀 Execution Guide

### Option 1: Run Locally on Linux GPU (RTX 3080 10GB via Qwen2-VL)
```bash
cd metrics_lab4/agentic_reasoning

# Run with local 4-bit quantized Qwen2-VL
HF_HOME=/data1tb/hf_cache python run_exp15_pipeline.py \
  --backend hf \
  --model "Qwen/Qwen2-VL-7B-Instruct"
```
*(Or use `"Qwen/Qwen2-VL-2B-Instruct"` for ultra-fast inference).*

### Option 2: Run via Cloud Vision API (OpenAI / OpenRouter / Groq Vision)
```bash
# Using OpenAI GPT-4o-mini Vision
export OPENAI_API_KEY="your_key_here"

python run_exp15_pipeline.py \
  --backend openai \
  --api-base "https://api.openai.com/v1" \
  --api-key "$OPENAI_API_KEY" \
  --model "gpt-4o-mini"
```

### Option 3: Deterministic Template Fallback Mode
```bash
python run_exp15_pipeline.py --backend template
```

---

## 📁 Directory Structure

```text
metrics_lab4/agentic_reasoning/
├── data/                       # CSV inputs and pre-rendered patient visual strips
│   └── patient_strips/         # 13-image scout strips & myocardial zoom crops
├── core/
│   ├── evidence_store.py       # Append-only evidence repository with image linking
│   ├── gap_analyzer.py         # ADRE diagnostic checklist analyzer
│   ├── consistency_checker.py  # Wide-plateau and conflict detector
│   ├── vlm_inspector.py        # Visual strip packaging and VLM invocation
│   └── stage_a_generator.py    # Structured JSON report generator
├── llm/
│   ├── vlm_client.py           # Multimodal client (HF Qwen2-VL & OpenAI Vision)
│   ├── prompts_vlm.py          # Radiologist visual inspection prompts
│   └── narrative_generator.py  # 7-block clinical report generator
├── eval/
│   ├── evaluator.py            # Cohort accuracy & VLM statistics
│   └── audit.py                # Clinical safety & hallucination verifier
├── run_exp15_pipeline.py       # Main CLI executable
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── outputs_exp15/              # Output reports, traces, and benchmark summaries
```
