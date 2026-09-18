# FreqScout Agentic Reasoning (Exp 12)
### Automatic Frequency Selection for 3.0T Cardiac bSSFP CINE MRI with Multi-Metric Evidence Grounding & Remote LLM Reasoning

This directory (`metrics_lab/agentic_reasoning/`) is a **self-contained deployment bundle** containing all validated multi-metric perceptual measurements, evidence stores, consistency checking, deterministic Stage-A findings generators, and LLM clinical reasoning modules.

---

## Benchmark Results (Cohort $n=25$, excluding Patient-KP)

| Method | Top-1 Accuracy | ±1 Step (±25 Hz) | Mean Absolute Error (MAE) |
|---|---|---|---|
| Center Slice (Naive 0 Hz) | 26.9% | 57.7% | 32.8 Hz |
| Exp5 DINOv2-Large LOOCV | 23.1% | 57.7% | 40.0 Hz |
| Standalone Banded Frac (`argmin bfm`) | 40.0% | 76.0% | 25.0 Hz |
| Standalone Homogeneity ($1 - \text{CoV}$) | 44.0% | 80.0% | 27.0 Hz |
| EfficientNet-B0 CNN (LOOCV from NB08) | 44.0% | 72.0% | 24.0 Hz |
| Exp10 Fused Agent | 42.3% | 76.9% | 22.1 Hz |
| **Agentic Multi-Metric Reasoning (Exp 12)** | **48.0% (12/25)** | **76.0% (19/25)** | **24.0 Hz** |

- **Audit Pass Rate**: 100.0%
- **Frequency Faithfulness**: 100.0%
- **Hallucination Rate**: 0.0%
- **Hard-case success**: Successfully resolved **Patient-CEJP** (#4 at -75 Hz), which failed across every prior method, via multi-source physics veto.

---

## Directory Structure

```
metrics_lab/agentic_reasoning/
├── data/                         # Self-contained quantitative metrics & ground truth
│   ├── cnn_scores.csv            # EfficientNet-B0 LOOCV scores (44% Top-1)
│   ├── banding_metrics.csv       # Banded fraction, trough depth, AHA sectors
│   ├── signal_metrics.csv        # Homogeneity (1-CoV), mean signal
│   ├── contrast_metrics.csv      # Blood-myocardium CNR, contrast ratio
│   ├── temporal_metrics.csv      # Temporal sweep stability
│   ├── edge_metrics.csv          # Boundary sharpness
│   └── ground_truth.csv          # 1-indexed ground truth selections
├── core/
│   ├── evidence_store.py         # ADRE-style append-only evidence store
│   ├── gap_analyzer.py           # Diagnostic checklist analyzer
│   ├── consistency_checker.py    # Cross-source consensus & physics veto engine
│   └── stage_a_generator.py      # Deterministic findings generator & passband model
├── llm/
│   ├── prompts.py                # 7-block clinical report system prompt & template
│   ├── client.py                 # Multi-backend LLM client (OpenAI / HF / Template)
│   ├── narrative_generator.py    # Generates Stage-B/C radiologist reports
│   └── server.py                 # Lightweight FastAPI server for 10GB RTX 3080
├── eval/
│   ├── audit.py                  # Hallucination & faithfulness verifier
│   └── evaluator.py              # Cohort benchmark evaluator
├── outputs/                      # Generated JSONs, Markdown reports & CSV tables
├── requirements.txt              # Remote GPU requirements
├── run_agentic_reasoning.py      # Main pipeline entrypoint
└── README.md                     # This documentation
```

---

## Transfer to Remote Machine via SCP

To copy this entire self-contained folder to your remote PC with the 10GB RTX 3080:

```bash
# Run from your local machine (e:\CINE-Test1):
scp -r metrics_lab/agentic_reasoning <username>@<remote_ip>:~/agentic_reasoning
```

Then SSH into the remote machine:
```bash
ssh <username>@<remote_ip>
cd ~/agentic_reasoning
```

---

## Remote GPU Setup (10GB RTX 3080)

### 1. Install Environment Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execution Options

#### Option A: Direct GPU Execution on the Remote Machine
Run the complete pipeline directly on the remote machine with `Qwen/Qwen2.5-7B-Instruct` loaded in 4-bit (takes ~5.2 GB VRAM):
```bash
python run_agentic_reasoning.py --backend hf --model Qwen/Qwen2.5-7B-Instruct
```

#### Option B: Run FastAPI Server on Remote GPU + Client on Local PC
1. On the remote RTX 3080 PC, start the server:
   ```bash
   python llm/server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000 --load-in-4bit
   ```
2. On your local machine, open an SSH tunnel for port 8000:
   ```bash
   ssh -L 8000:localhost:8000 <username>@<remote_ip>
   ```
3. Run the pipeline locally (queries the remote GPU seamlessly):
   ```bash
   python run_agentic_reasoning.py --backend openai --api-base http://localhost:8000/v1
   ```

#### Option C: Instant Local Deterministic Dry-Run (No GPU Required)
```bash
python run_agentic_reasoning.py --backend template
```

---

## Clinical Report Output Schema (7 Blocks)

Every patient report in `outputs/<patient_id>/clinical_report.md` follows the standardized radiologist schema:
1. **CLINICAL INDICATION & OBJECTIVE**: Explains 3.0T bSSFP shimming goals.
2. **PRIMARY FREQUENCY RECOMMENDATION**: Recommended Hz, image number, and composite rating.
3. **DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)**: Usable ±25 Hz to ±50 Hz tolerance range.
4. **REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION**: Specific offsets vetoed due to dark banding, citing exact AHA myocardial sectors (anterior, inferior, lateral).
5. **MULTI-SOURCE EVIDENCE RECONCILIATION**: Deep learning vs physics vs homogeneity alignment or veto justification.
6. **TECHNICAL LIMITATIONS & PITFALLS**: Patient shimming notes & susceptibility warnings.
7. **SUMMARY & OPERATOR DIRECTIVE**: Clear scanner offset instruction.
