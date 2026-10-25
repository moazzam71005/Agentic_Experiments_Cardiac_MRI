# Metrics Lab 5 — Gated LLM Arbitration with ViT + BiLSTM Neural Scorer

This folder implements the **Gated LLM Arbitration Pipeline** (from Experiment 13) where the single-slice CNN has been completely swapped with the **Pretrained DINOv2-Large + SliceSelector BiLSTM sequence ranker** (from Experiment 17).

---

## 🏗️ Architecture

```
                      Multi-Offset Physical & Sequence Tools
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  [ViT + BiLSTM]              [Banded Fraction (BFM)]          [Myocardial Homogeneity]
 (Exp 17 Sequence Score)     (UNet Banding Detector)          (UNet Signal Uniformity)
        │                                │                                │
        └────────────────────────────────┼────────────────────────────────┘
                                         ▼
                          [Stage-A Multi-Metric Fusion]
                                         │
                          [Consistency Gating Router]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            [Full Consensus]                         [Divergent / Veto]
          (Fast Deterministic)                     (Gated LLM Arbitrator)
```

---

## 🚀 How to Run

### 1. Run with Groq API (Qwen-2.5-32B or Llama-3.3-70B)
Set your `GROQ_API_KEY` and run:

```bash
# In PowerShell / Command Prompt:
set GROQ_API_KEY=gsk_your_actual_key_here
python run_exp13_pipeline.py --backend groq --model qwen-2.5-32b

# Or pass the key explicitly:
python run_exp13_pipeline.py --backend groq --api-key gsk_your_actual_key_here --model qwen-2.5-32b
```

*(Available Groq models: `qwen-2.5-32b`, `llama-3.3-70b-versatile`)*

---

### 2. Run Deterministic Baseline (Template Mode — No API Key Required)
```bash
python run_exp13_pipeline.py --backend template
```

---

### 3. Run on Local GPU Machine (RTX 3080 via HuggingFace 4-bit)
```bash
HF_HOME=/data1tb/hf_cache python run_exp13_pipeline.py --backend hf --model Qwen/Qwen2.5-7B-Instruct
```

---

## 📁 Outputs & Traces
Results, per-patient clinical markdown reports, and arbitration logs will be saved to:
`metrics_lab5/agentic_reasoning/outputs_exp18_vit_bilstm_gated/`
