# Experiment 14: ADRE Multi-Agent Debate & Wide-Plateau Refinement

Self-contained deployment directory for **Experiment 14** of the FYP:
*Agentic Reasoning for Enhanced Automatic Frequency Selection of bSSFP CINE MRI at 3.0T*.

---

## 🌟 Key Advancements in Exp 14

Exp 14 introduces the full **Autonomous Diagnostic Reasoning Engine (ADRE) Tri-Agent Debate Architecture** with **Wide-Passband Plateau Gating**:

1. **Wide-Plateau Gating (64% Invocation Coverage)**:
   - In Exp 13, only 16% of patients triggered arbitration (`divergent` / `borderline_veto`), leaving 84% to a static formula that suffered from edge-offset errors on wide plateaus.
   - Exp 14 identifies ambiguous plateaus ($\ge 50\text{ Hz}$ width) and initiates multi-agent deliberation to pinpoint the deep center.

2. **Tri-Agent Role Specialization**:
   - **Physics Specialist Agent**: Analyzes off-resonance margin $\Delta f$, BFM gradient across frequency, and proximity to $1/\text{TR}$ null lines.
   - **Contrast & Texture Specialist Agent**: Analyzes Myocardial SNR, CNR (myo-to-blood ratio), 6-sector AHA homogeneity, and boundary sharpness.
   - **Clinical Meta-Arbitrator & Critic**: Synthesizes arguments, performs counterfactual critique, and outputs verified JSON decisions.

3. **Multi-Tier Safety Guardrails**:
   - Hard physics veto pre-filters candidates with Banded Fraction Metric $> 0.08$.
   - Automatic fallback to deterministic composite scoring if safety audit fails.

---

## 🚀 Execution Guide

### Option 1: Run Locally on Linux GPU (RTX 3080 10GB)
```bash
cd metrics_lab3/agentic_reasoning

# Using 4-bit quantized local model
python run_exp14_pipeline.py \
  --backend hf \
  --model "Qwen/Qwen2.5-7B-Instruct"
```

### Option 2: Run via Cloud API (Groq / OpenAI)
```bash
# Set your API key
export GROQ_API_KEY="your_key_here"

# Run with Groq
python run_exp14_pipeline.py \
  --backend openai \
  --api-base "https://api.groq.com/openai/v1" \
  --api-key "$GROQ_API_KEY" \
  --model "qwen/qwen3.8-27b"
```

### Option 3: Deterministic Template Fallback Mode
```bash
python run_exp14_pipeline.py --backend template
```

---

## 📁 Directory Structure

```text
metrics_lab3/agentic_reasoning/
├── data/                       # CSV inputs (CNN scores, BFM, homogeneity, ground truth)
├── core/
│   ├── evidence_store.py       # Append-only evidence repository
│   ├── gap_analyzer.py         # ADRE diagnostic checklist analyzer
│   ├── consistency_checker.py  # Wide-plateau and conflict detector
│   ├── adre_debate.py          # Tri-agent debate orchestration engine
│   └── stage_a_generator.py    # Structured JSON report generator
├── llm/
│   ├── client.py               # HuggingFace 4-bit / OpenAI / Groq client with backoff
│   ├── prompts_debate.py       # Specialist agent system & user prompts
│   └── narrative_generator.py  # 7-block clinical report generator
├── eval/
│   ├── evaluator.py            # Cohort accuracy & debate statistics
│   └── audit.py                # Clinical safety & hallucination verifier
├── run_exp14_pipeline.py       # Main CLI executable
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── outputs_exp14/              # Output reports, traces, and benchmark summaries
```
