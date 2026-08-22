# FreqScout Agent — Experiment Log & Knowledge Base

## Project Overview

**Goal:** Develop an agentic system for automatic optimal frequency offset selection in
bSSFP CINE cardiac MRI. The system must be clinically grounded, physics-coherent, and
provide interpretable confidence alongside its recommendation.

**Clinical context:** Balanced Steady-State Free Precession (bSSFP) MRI produces banding
artifacts (dark stripes) at off-resonance frequencies. A frequency scout acquires 13 images
at evenly spaced RF offsets (−150 Hz to +150 Hz, step 25 Hz). The radiographer manually
selects the image where banding is outside the heart. This project automates that selection.

**Dataset:** 26 patients, 1st cohort SAX (short-axis) bSSFP frequency scout.
- Each patient: 13 DICOM images at offsets −150, −125, …, +150 Hz
- Ground truth: Excel file `1stcohortSAX (1).xlsx` → `image no` column (1-indexed)
- Exception: Patient-KP has only 7 images and looks anatomically atypical (apical slice)
- Data root: `Dr-Dataset/1st_cohort_SAX/Patient-*/`

---

## System Architecture

```
DICOM stack (13 slices)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                  Agentic Reasoning Loop                   │
│                                                           │
│  T1: NeuralRanker    — Exp5 LOOCV DINOv2-Large scores    │
│  T2: SegmentationTool— LVSA U-Net consensus mask          │
│  T3: BandingDetector — intra-myocardial banding metrics   │
│  T4: QualityMetrics  — homogeneity, CNR, banded fraction  │
│  T5: PhysicsDescriptor—offset sweep, band period          │
│  T6: VLMDescriptor   — Qwen2-VL image quality verdicts    │
│  T7: KnowledgeRetriever— bSSFP physics knowledge          │
│                                                           │
│  GapAnalyser → decides which tool to call next            │
│  ConsistencyChecker → cross-source agreement + confidence │
└─────────────────────────────────────────────────────────┘
       │
       ▼
  StageAGenerator  →  composite score per slice
       │                 W_NEURAL=0.50 · neural_score
       │               + W_HOMOG=0.20  · homogeneity
       │               + W_NO_BURDEN=0.20 · (1−banding_burden)
       │               + W_CENTER=0.10  · Gaussian(σ=100Hz)
       │               + W_VLM=0.00    · vlm_score  (disabled)
       ▼
  Recommended offset + confidence label
```

### Key Models

| Component | Model | Notes |
|---|---|---|
| Neural ranker | DINOv2-Large | Exp5 LOOCV per-patient scores (CSV) |
| Segmentation | LVSA U-Net (UNet_64, advchain) | `Unet_LVSA_best.pkl`, Singapore data |
| VLM | Qwen2-VL-2B (local) / Qwen2.5-VL-7B (lab) | 4-bit quantised |
| Text reasoner | Qwen2.5-7B-Instruct | Stage-B narrative only |

### Segmentation Labels
- 0 = Background
- 1 = RV (yellow in overlays)
- 2 = Myocardium (green)
- 3 = LV blood pool (blue)

### Key Files
```
freqscout_agent/
  pipeline.py              — end-to-end entry point
  configs/models.yaml      — all model paths and weights
  tools/
    neural_scorer.py       — DINOv2 Exp5 ranker
    seg_advchain/          — vendored U-Net (from E:/cardiotune)
      adapter.py           — LVSASegAdapter: preprocess, infer, unpad
      unet.py / unet_parts.py / init_weight.py
    segment.py             — SegmentationTool wrapper
    banding.py             — BandingDetector + BandingResult
    quality.py             — QualityMetrics + QualityResult
    vlm_describe.py        — VLMDescriptor (Qwen2-VL)
    physics.py             — PhysicsDescriptor (Excel GT parsing)
  generation/
    stage_a_findings.py    — StageAGenerator: composite score
  agent/
    consistency.py         — ConsistencyChecker: confidence
    reasoning_loop.py      — agentic tool-call loop
    gap_analysis.py        — gap detection checklist
run_exp8_seg_qc.py         — segmentation visual QA script
run_exp9_pipeline.py       — full pipeline run script
experiment8_seg_grounded.ipynb — ablation notebook
```

---

## Experiment History

### Exp1–4: Feature Extraction (DINOv2)
- Extracted DINOv2-Large features for all 26 × 13 = 338 images
- Stored in `exp4_outputs/dinov2_features.pkl` (later deleted from git — too large)
- Established DINOv2-Large as the best visual feature extractor for this task

### Exp5: LOOCV Neural Ranker
- Leave-one-out cross-validation ranker using DINOv2-Large features
- Per-patient scores stored in `exp5_outputs/exp5_loocv_per_patient.csv`
- This CSV is the primary neural signal for all downstream experiments

### Exp7: Neural-Only Baseline (reference point)
| Metric | Value |
|---|---|
| Top-1 | 6/26 = **23.1%** |
| Top-3 | 16/26 = 61.5% |
| Within ±1 step | 10/26 = 38.5% |
| MAE | **64.4 Hz** |

### Exp8: Myocardium-Grounded Composite (notebook ablation)
New components added:
- LVSA U-Net consensus segmentation for real myocardium masks
- Intra-myocardial banding metrics: `banding_burden`, `banded_fraction_myo`, `band_trough_depth`
- Fixed confidence: removed inflated +0.15 "all-agree" boost

Ablation results:

| Selector | Top-1 | Top-3 | ±1 step | MAE |
|---|---|---|---|---|
| Neural-only (Exp7) | 23.1% | 61.5% | 38.5% | 64.4 Hz |
| Burden-only | **3.8%** | 11.5% | 38.5% | 69.2 Hz |
| Fused (neural+CV) | **30.8%** | **73.1%** | 42.3% | 63.5 Hz |
| Fused+VLM placeholder | 30.8% | 73.1% | 42.3% | 63.5 Hz |

**Key finding — burden_only fails:** At extreme offsets (±150 Hz), banding has not yet
entered the myocardium, so burden ≈ 0 there too. argmin(burden) always picks −150 Hz.
Burden must be used as a soft penalty, not a standalone selector.

### Exp9: Full Pipeline + Physics Center Prior (main results)

**New additions:**
- Center-frequency Gaussian prior: `exp(−0.5·(Hz/100)²)` normalised to [0,1]
  - Physics rationale: optimal offset is rarely at sweep extremes; B0 inhomogeneity
    shifts it by ≤100 Hz from centre for most patients
  - Weight W_CENTER = 0.10 in composite
- Weight rebalance: W_NEURAL 0.40→0.50, W_HOMOG 0.25→0.20, W_NO_BURDEN 0.30→0.20
- Composite-margin confidence: uses gap between top-1 and top-2 composite scores
- Fixed VLM prompt: per-image absolute criteria (removed "at most one optimal" comparison)

#### Results across hardware / VLM sizes:

| Run | Hardware | VLM | Top-1 | ±1 step | MAE | Conf(C) | Conf(W) | ECE |
|---|---|---|---|---|---|---|---|---|
| Exp9-v1 (old prompt) | RTX 3050 | 2B | 34.6% | 65.4% | 26.9 Hz | 0.228 | 0.230 | 0.174 |
| Exp9-v2 (new prompt) | RTX 3050 | 2B | 34.6% | 65.4% | 26.9 Hz | 0.350 | 0.359 | 0.257 |
| Exp9-v3 | RTX 3080 | 7B | 34.6% | 65.4% | 26.9 Hz | 0.323 | 0.336 | 0.243 |

**Top-1 accuracy (per patient — all runs identical):**
Correct (9/26 = 34.6%): BGT, BVE, DEM, FAJ, IAM, INA, MJ, SWJ, WBC

**Within ±1 step (18/26 = 69.2%):**
BEV, BGT, BVE, DEM, FAJ, HMJ, IAM, INA, KP, MJ, MS, MSR, OAF, PJ, RAK, SWJ, WBC, WMR

**Fails (±2+ steps):** CEA, CEJP, FEC, GJ, HRJ, RK, SK, TL

#### MAE improvement: 64.4 Hz → 26.9 Hz (−58%)
The center prior is the primary driver: it stops the composite from over-weighting
extreme offsets where burden is accidentally near-zero.

---

## Key Findings & Lessons

### 1. VLM is NOT useful for this task (critical negative result)

Both Qwen2-VL-2B and Qwen2.5-VL-7B fail as discriminators:

| VLM | Verdict distribution | VLM top-1 accuracy |
|---|---|---|
| 2B (old prompt) | all "suboptimal" | 0% (no discrimination) |
| 2B (new prompt) | "acceptable"=138, "optimal"=20 | ~0% |
| 7B (new prompt) | "acceptable"=138, "optimal"=20 | **0/20 = 0%** |

**Root cause — center-frequency bias:** The 7B VLM called 14/20 "optimal" verdicts
at exactly 0 Hz, regardless of patient. It is applying a learned prior
("center frequency is optimal") rather than reading the actual image content. Enabling
W_VLM > 0 would hurt: 0 patients helped, 8 patients hurt.

**Scientific implication:** General-purpose VLMs cannot reliably distinguish bSSFP
banding artifact severity at the sub-myocardium level. A medically fine-tuned VLM
or a contrastive pair-wise prompt (show two images, ask which is better) may be needed.

### 2. Segmentation quality
- 25/26 patients: U-Net produces valid myocardium mask (green ring visible)
- Patient-KP: all-zero mask (7 images, atypical slice anatomy — likely apical view)
- Bug fixed (Exp9): `_unpad` sign error caused 24px rightward shift of masks
  - Fix: `max(-pad_w, 0)` → `max(pad_w, 0)` in `adapter.py`
- The consensus mask (majority vote across 13 offsets) is more robust than single-frame

### 3. Banding burden metric behaviour
- Near-zero at extreme frequencies (±150 Hz): banding null hasn't entered myocardium yet
- Causes argmin(burden) to always pick −150 Hz → useless as standalone selector
- As a soft penalty within composite: contributes correctly to penalise offsets
  where banding overlaps the myocardium (×0.25 hard penalty)

### 4. Confidence calibration is poor
- Correct vs wrong confidence: 0.323 vs 0.336 (nearly identical)
- ECE ≈ 0.25 (significant overconfidence at medium/high labels)
- Root cause: composite margin doesn't correlate well with correctness at n=26
- Structural limitation: with 26 patients and LOOCV, the neural scores are flat
  (13-class softmax → typical margin ~3%)

### 5. Physics center prior works well
- The single biggest accuracy gain came from the Gaussian center prior
- MAE halved from ~64 Hz to 26.9 Hz
- Top-1 improved from 30.8% (Exp8) to 34.6% (Exp9)
- ±1 step improved from 42.3% (Exp8) to 65.4% (Exp9)
- Physically grounded: bSSFP null shifts by ≤100 Hz from 0 for typical B0 at 3T

---

## Known Bugs Fixed

| Bug | File | Fix |
|---|---|---|
| Mask 24px right shift | `seg_advchain/adapter.py` `_unpad` | `max(-pad_w,0)` → `max(pad_w,0)` |
| GT loading `StopIteration` | `experiment8_seg_grounded.ipynb` | Replaced openpyxl with pandas auto-header |
| `from_config()` signature | `run_exp9_pipeline.py` | Pass `output_dir` to `__init__`, not `from_config` |
| VLM all-suboptimal | `vlm_describe.py` `_SYSTEM_PROMPT` | Removed "at most one optimal" comparison rule |
| Flat confidence (all low) | `consistency.py` + `stage_a_findings.py` | Composite-margin formula, scale ×4 |
| VLM penalty when disabled | `consistency.py` | Skip penalty when `vlm_pkt` is empty |

---

## Current Best Results (Exp9)

```
Top-1 accuracy   : 9/26  = 34.6%
Top-3 accuracy   : 16/26 = 61.5%
Within ±1 step   : 18/26 = 69.2%   (proxy eval reports 17/26 due to GT Hz convention)
Mean MAE         : 26.9 Hz
Hallucination    : 0%
```

**Composite weights (models.yaml / stage_a_findings.py):**
```
W_NEURAL    = 0.50   # Exp5 LOOCV neural score
W_HOMOG     = 0.20   # 1 - intra-myocardial CoV
W_NO_BURDEN = 0.20   # 1 - banding_burden
W_CENTER    = 0.10   # Gaussian center prior σ=100 Hz
W_VLM       = 0.00   # VLM disabled (biased, unreliable)
```

---

## Infrastructure

### Local PC (development)
- GPU: NVIDIA RTX 3050 8 GB
- OS: Windows 11
- VLM: Qwen2-VL-2B-Instruct (4-bit, fits in 8 GB)
- Segmentation: CPU (fast enough for U-Net consensus)
- Path: `E:\CINE-Test1\`

### Lab PC (experiment runs)
- GPU: NVIDIA RTX 3080 (TUKL)
- OS: Linux (Ubuntu)
- VLM: Qwen2.5-VL-7B-Instruct (4-bit)
- Repo: `~/Agentic_Experiments_Cardiac_MRI/`
- SSH: `ssh <user>@<lab-ip>`
- **Note:** CUDA driver mismatch encountered (driver version 12080 too old for latest PyTorch)
  - Fix: `pip install torch --index-url https://download.pytorch.org/whl/cu<version>`

### GitHub
- Remote: `https://github.com/moazzam71005/Agentic_Experiments_Cardiac_MRI.git`
- Branch: `main`
- Large files NOT in git: DICOMs, `.pkl` checkpoints, `exp4_outputs/`, `agent_outputs_*/`

### Checkpoint location
- Local: `E:/cardiotune/backend/ml/Cardiac_Multi_view_segmentation/checkpoints/Unet_LVSA_best.pkl`
- Lab: manually SCP'd to `~/Agentic_Experiments_Cardiac_MRI/`

---

## Next Steps (Exp10 and beyond)

### High priority
1. **Retrain or fine-tune the neural ranker** — DINOv2 LOOCV on 26 patients is the
   accuracy ceiling. Options:
   - Collect more patients
   - Use stronger augmentation in LOOCV
   - Try a contrastive ranking loss instead of softmax
2. **Investigate the 8 failing patients** (CEA, CEJP, FEC, GJ, HRJ, RK, SK, TL):
   - All have ≥50 Hz error
   - Check if their DICOMs have unusual anatomy or image artefacts
   - Check if their GT in the Excel is clinically well-defined

### Medium priority
3. **VLM redesign** — if VLM is to be used:
   - Use **pairwise comparison prompt**: show 2 images, ask "which has less myocardial banding?"
   - Or use a medically fine-tuned model (MedGemma, BioViL-T)
   - Avoid absolute quality verdicts with a 4-point scale — small models can't calibrate this
4. **Confidence calibration** — post-hoc Platt scaling or isotonic regression on the
   composite margin to reduce ECE below 0.15
5. **Patient-KP** — investigate whether this patient should be excluded from evaluation
   or treated specially (different frequency sweep range)

### Low priority
6. **Stage-B narrative** — improve the clinical narrative generation
7. **Run on second cohort** — if additional patient data becomes available
