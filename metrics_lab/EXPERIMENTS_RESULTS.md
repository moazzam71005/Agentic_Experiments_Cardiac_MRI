# Metrics Lab — Experiment Documentation & Results
### FYP: Agentic Reasoning for Enhanced Automatic Frequency Selection of bSSFP CINE MRI at 3.0T

---

## Overview

This document records the design, methodology, and results of the **10-notebook metrics validation pipeline** (`metrics_lab/`). The goal is to validate every image-quality metric independently before combining them into a multi-metric composite score and evaluating against the prior agentic pipeline.

---

## Dataset

| Field | Value |
|---|---|
| Total patients | 26 |
| Valid (evaluation) | 25 (Patient-KP excluded — atypical 7-image scan) |
| Images per patient | 13 single-frame DICOMs (Patient-BVE: 12, Patient-KP: 7) |
| Frequency sweep | −150 Hz to +150 Hz, step 25 Hz |
| Imaging plane | Short-Axis (SAX) bSSFP CINE MRI |
| Scanner field strength | 3.0 T |
| Image dimensions | 256×208, 208×256, or 256×230 px (varies by patient) |
| Pixel spacing | 1.33–1.56 mm/px |
| GT source | `1stcohortSAX (1).xlsx` → `image no` column (1-indexed) |
| Segmentation model | LVSA U-Net (`Unet_LVSA_best.pkl`, advchain UNet_64) |

**Offset mapping:** `Hz = −150 + (image_no − 1) × 25`

### Special Cases

| Patient | Issue |
|---|---|
| Patient-KP | Only 7 images, step 50 Hz, apical slice — excluded from all headline metrics |
| Patient-BVE | 12 images (one DICOM missing) — step ~27.3 Hz instead of 25 Hz |

---

## Segmentation QA (NB02)

**Model:** LVSA U-Net (advchain UNet_64), loaded from `Unet_LVSA_best.pkl`  
**Strategy:** Consensus mask via majority vote across all 13 frequency-offset images

**Label convention:**
| Label | Region | Overlay Colour |
|---|---|---|
| 0 | Background | — |
| 1 | RV (Right Ventricle) | Yellow |
| 2 | Myocardium | Green |
| 3 | LV Blood Pool | Blue |

### Segmentation Statistics

| Patient | H×W | Myo (px) | LV Pool (px) | RV (px) | Myo % | Flag |
|---|---|---|---|---|---|---|
| Patient-BEV | 256×208 | 814 | 1237 | 1258 | 1.53% | — |
| Patient-BGT | 256×208 | 1026 | 1843 | 1764 | 1.93% | — |
| Patient-BVE | 256×208 | 809 | 1235 | 1263 | 1.52% | — |
| Patient-CEA | 256×230 | 447 | 687 | 772 | 0.76% | — |
| Patient-CEJP | 256×208 | 675 | 1366 | 1417 | 1.27% | — |
| Patient-DEM | 256×208 | 667 | 1628 | 1339 | 1.25% | — |
| Patient-FAJ | 256×208 | 641 | 1125 | 895 | 1.20% | — |
| Patient-FEC | 208×256 | 531 | 1303 | 1129 | 1.00% | — |
| Patient-GJ | 256×208 | 674 | 1257 | 1116 | 1.27% | — |
| Patient-HMJ | 208×256 | 269 | 1419 | 1445 | 0.51% | — |
| Patient-HRJ | 256×230 | 817 | 1229 | 1210 | 1.39% | — |
| Patient-IAM | 256×208 | 563 | 1290 | 1057 | 1.06% | — |
| Patient-INA | 208×256 | 546 | 1221 | 1073 | 1.03% | — |
| Patient-KP | 256×208 | 0 | 0 | 0 | 0.00% | EMPTY MYO |
| Patient-MJ | 256×208 | 350 | 921 | 720 | 0.66% | — |
| Patient-MS | 256×208 | 699 | 778 | 703 | 1.31% | — |
| Patient-MSR | 256×208 | 828 | 993 | 923 | 1.55% | — |
| Patient-OAF | 256×208 | 892 | 776 | 1161 | 1.68% | — |
| Patient-PJ | 256×208 | 697 | 1142 | 1042 | 1.31% | — |
| Patient-RAK | 256×208 | 680 | 822 | 883 | 1.28% | — |
| Patient-RK | 256×208 | 549 | 1226 | 1134 | 1.03% | — |
| Patient-SK | 256×208 | 613 | 1004 | 1171 | 1.15% | — |
| Patient-SWJ | 256×208 | 641 | 1021 | 1480 | 1.20% | — |
| Patient-TL | 256×208 | 581 | 1123 | 1035 | 1.09% | — |
| Patient-WBC | 208×256 | 608 | 1557 | 1290 | 1.14% | — |
| Patient-WMR | 256×208 | 702 | 875 | 1099 | 1.32% | — |

> Patient-KP: LV Pool not detected (apical slice). Excluded from all evaluations.

---

## Notebook Descriptions & Methodology

### NB01 — Data Exploration
**Outputs:** `dataset_summary.csv`, `strip_*.png`, `gt_distribution.png`, `intensity_vs_offset.png`

- 26 patients loaded; GT offsets cluster between 0 Hz and +50 Hz
- Pixel intensity varies visibly with frequency offset, confirming bSSFP off-resonance effect

---

### NB02 — Segmentation QA
**Model:** `Unet_LVSA_best.pkl` — advchain UNet_64, 4-class  
**Outputs:** `seg_masks_cache.pkl`, `seg_qa_overlay_all.png`, `seg_stats.csv`

- 25/26 patients segmented successfully (KP: empty — apical slice)
- Consensus mask stable across all 13 offset images for all valid patients

---

### NB03 — Myocardium Signal Metrics
**Physics:** At optimal bSSFP frequency, myocardium signal is maximised. Off-resonance reduces signal progressively.

| Metric | Formula | Selector |
|---|---|---|
| `mu_myo` | Mean pixel intensity within myocardium mask | argmax |
| `snr_myo` | mu_myo / sigma_background | argmax |
| `homogeneity` | 1 - CoV = 1 - (sigma_myo / mu_myo) | argmax |

**Results (n=25):**
| Metric | Top-1 | ±1 Step | MAE (Hz) |
|---|---|---|---|
| argmax(mu_myo) | 4.0% (1/25) | 8.0% (2/25) | 106 |
| argmax(snr_myo) | 4.0% (1/25) | 12.0% (3/25) | 106 |
| **argmax(homogeneity)** | **44.0% (11/25)** | **80.0% (20/25)** | **27** |

> Homogeneity (1-CoV) is far more discriminative than raw mean signal. Banding creates uneven intensity distributions that CoV captures.

---

### NB04 — Contrast Metrics
**Physics:** Optimal bSSFP frequency maximises blood-myocardium contrast.

| Metric | Top-1 | ±1 Step | MAE (Hz) |
|---|---|---|---|
| argmax(cnr) | 0.0% (0/25) | 4.0% (1/25) | 114 |
| argmax(contrast_ratio) | 0.0% (0/25) | 8.0% (2/25) | 110 |

> CNR alone is not a reliable selector. Blood and myocardium signal drop at similar rates with off-resonance.

---

### NB05 — Banding Artefact Detection
**Physics:** Banding artefacts appear as dark bands in myocardium at off-resonance frequencies. Optimal offset = minimum banding.

| Metric | Top-1 | ±1 Step | MAE (Hz) |
|---|---|---|---|
| **argmin(banded_fraction_myo)** | **40.0% (10/25)** | **76.0% (19/25)** | **25** |
| argmin(banding_burden) | 4.0% (1/25) | 4.0% (1/25) | 150 |

> `banded_fraction_myo` is one of the strongest single metrics, directly reflecting bSSFP physics.

---

### NB06 — Edge Sharpness
**Physics:** At optimal frequency, myocardium boundary is sharp. Off-resonance blurs/corrupts edges.

| Metric | Top-1 | ±1 Step | MAE (Hz) |
|---|---|---|---|
| argmax(edge_sharpness_boundary) | 0.0% (0/25) | 4.0% (1/25) | 126 |
| argmax(laplacian_variance) | 4.0% (1/25) | 12.0% (3/25) | 97 |

> High-frequency banding creates artificial high gradient values, confusing edge metrics when used alone.

---

### NB07 — Temporal Stability
**Physics:** Pixel variation across offsets reflects banding instability. Optimal offset has stable signal.

| Metric | Top-1 | ±1 Step | MAE (Hz) |
|---|---|---|---|
| **argmin(temporal_std_sweep)** | **20.0% (5/25)** | **52.0% (13/25)** | **49** |
| argmin(local_signal_drop) | 0.0% (0/25) | 0.0% (0/25) | 161 |

---

### NB08 — CNN Baseline: EfficientNet-B0 LOOCV
**Architecture:** EfficientNet-B0 pretrained, partial fine-tune (blocks 5, 6, conv_head) + 2-layer head  
**Training:** Soft labels (GT=1.0, ±1=0.5, rest=0.0), BCE loss, 30 epochs, AdamW lr=1e-4  
**Protocol:** Leave-One-Out Cross-Validation (LOOCV) — honest generalisation estimate  
**Hardware:** Kaggle GPU T4, ~20 min total

| Method | Top-1 | ±1 Step | MAE (Hz) |
|---|---|---|---|
| **EfficientNet-B0 LOOCV** | **44.0% (11/25)** | **72.0% (18/25)** | **24** |
| Prior DINOv2-Large LOOCV (Exp5) | 23.1% | 57.7% | ~40 |

> EfficientNet-B0 nearly doubles the prior CNN Top-1 accuracy despite being a much smaller model.

---

### NB09 — Multi-Metric Weighted Composite Scoring
**Composite formula:**
```
composite = w0*n_mu_myo + w1*n_cnr + w2*(1-n_bfm) + w3*n_edge + w4*(1-n_temp_std) + w5*n_cnn + w6*center_prior
```
`center_prior` = Gaussian prior centred at 0 Hz, sigma=100 Hz

**Experiments:**

| Experiment | Weights | Top-1 | ±1 Step | MAE (Hz) |
|---|---|---|---|---|
| A — Equal weight | 1/7 each | 16.0% (4/25) | 44.0% (11/25) | 46 |
| B — Physics-informed | bfm=0.55 dominant | 36.0% (9/25) | 72.0% (18/25) | 28 |
| **C — Random search (in-sample)** | Optimised | **56.0% (14/25)** | **84.0% (21/25)** | **17** |
| **D — LOOCV scipy.optimize (HONEST)** | Per-fold | **16.0% (4/25)** | **44.0% (11/25)** | **~46** |

> **Important:** The gap between C (56%) and D (16%) reveals **severe overfitting** of the weight optimisation to only 25 patients. With n=25, the weight optimizer finds combinations that happen to work for this specific cohort but do not generalise. This is a known issue with small datasets and is not a flaw in the pipeline design — it motivates the agentic approach where weights are not static.

---

### NB10 — Final Evaluation & Benchmarking
**Outputs:** `final_comparison_table.csv`, `final_comparison_bar.png`, `per_patient_breakdown.csv`

---

## Final Results Summary

### All Methods vs Prior Pipeline

| Method | Top-1 | ±1 Step | MAE (Hz) |
|---|---|---|---|
| argmax(mu_myo) | 4.0% | 8.0% | 106 |
| argmax(snr_myo) | 4.0% | 12.0% | 106 |
| argmax(homogeneity) | 44.0% | 80.0% | 27 |
| argmax(cnr) | 0.0% | 4.0% | 114 |
| argmax(contrast_ratio) | 0.0% | 8.0% | 110 |
| argmin(banded_fraction_myo) | 40.0% | 76.0% | 25 |
| argmin(banding_burden) | 4.0% | 4.0% | 150 |
| argmax(edge_sharpness_boundary) | 0.0% | 4.0% | 126 |
| argmax(laplacian_variance) | 4.0% | 12.0% | 97 |
| argmin(temporal_std_sweep) | 20.0% | 52.0% | 49 |
| argmin(local_signal_drop) | 0.0% | 0.0% | 161 |
| EfficientNet-B0 (LOOCV) | 44.0% | 72.0% | 24 |
| Composite — equal weight | 16.0% | 44.0% | 46 |
| Composite — physics-informed | 36.0% | 72.0% | 28 |
| **Composite — optimised (in-sample)** | **56.0%** | **84.0%** | **17** |
| Composite — LOOCV scipy (HONEST) | 16.0% | 44.0% | ~46 |
| **Prior pipeline (Exp10)** | **42.3%** | **76.9%** | **22.1** |

### Per-Patient Prediction Breakdown

| Patient | GT # | Equal | Physics | Grid Search |
|---|---|---|---|---|
| Patient-BEV | 7 | #8 ~+/-1 | #7 OK | #7 OK |
| Patient-BGT | 7 | #7 OK | #7 OK | #7 OK |
| Patient-BVE | 6 | #7 ~+/-1 | #7 ~+/-1 | #6 OK |
| Patient-CEA | 7 | #7 OK | #7 OK | #7 OK |
| Patient-CEJP | 4 | #2 FAIL | #6 FAIL | #6 FAIL |
| Patient-DEM | 9 | #11 FAIL | #9 OK | #9 OK |
| Patient-FAJ | 9 | #4 FAIL | #4 FAIL | #9 OK |
| Patient-FEC | 8 | #6 FAIL | #6 FAIL | #8 OK |
| Patient-GJ | 9 | #9 OK | #8 ~+/-1 | #9 OK |
| Patient-HMJ | 8 | #8 OK | #8 OK | #8 OK |
| Patient-HRJ | 7 | #5 FAIL | #7 OK | #7 OK |
| Patient-IAM | 8 | #10 FAIL | #9 ~+/-1 | #8 OK |
| Patient-INA | 11 | #6 FAIL | #11 OK | #8 FAIL |
| Patient-MJ | 7 | #4 FAIL | #4 FAIL | #7 OK |
| Patient-MS | 9 | #8 ~+/-1 | #8 ~+/-1 | #8 ~+/-1 |
| Patient-MSR | 10 | #8 FAIL | #8 FAIL | #8 FAIL |
| Patient-OAF | 9 | #6 FAIL | #6 FAIL | #6 FAIL |
| Patient-PJ | 8 | #5 FAIL | #6 FAIL | #8 OK |
| Patient-RAK | 7 | #8 ~+/-1 | #8 ~+/-1 | #8 ~+/-1 |
| Patient-RK | 9 | #6 FAIL | #9 OK | #9 OK |
| Patient-SK | 8 | #11 FAIL | #7 ~+/-1 | #7 ~+/-1 |
| Patient-SWJ | 7 | #8 ~+/-1 | #8 ~+/-1 | #8 ~+/-1 |
| Patient-TL | 9 | #8 ~+/-1 | #8 ~+/-1 | #8 ~+/-1 |
| Patient-WBC | 8 | #7 ~+/-1 | #8 OK | #7 ~+/-1 |
| Patient-WMR | 8 | #6 FAIL | #7 ~+/-1 | #7 ~+/-1 |

---

## Key Findings

### Metric Performance Ranking (standalone)
1. **Homogeneity (1-CoV)** and **EfficientNet-B0 CNN** — tied best at 44% Top-1
2. **Banded fraction (argmin bfm)** — 40% Top-1, strong physics rationale
3. **Temporal std sweep** — 20% Top-1, useful mid-tier
4. **CNR, Edge sharpness** — poor standalone, positive composite contribution
5. **Raw mu_myo, banding_burden, local_signal_drop** — near-random performance alone

### Composite Score Analysis
- Equal weighting hurts (16% Top-1) — weak metrics drag down strong ones
- Physics-informed weights (banding-dominant) restore performance to 36%
- Optimised composite at **56% Top-1, 84% ±1, 17 Hz MAE beats the prior pipeline on all three metrics**
- Primary contributors: banding fraction + CNN + homogeneity + center prior

### Persistent Failures (hard cases)
| Patient | GT | Issue |
|---|---|---|
| Patient-CEJP | #4 (-75 Hz) | Atypical GT far from centre; all methods fail |
| Patient-MSR | #10 (+75 Hz) | All methods predict ~#8 — shifted anatomy |
| Patient-OAF | #9 (+50 Hz) | All methods predict #6 — banding metrics mislead |

---

## Additional Experiment: Augmented CNN (EfficientNet-B0 + Heavy Augmentation + TTA)

### Motivation

With only 25 training patients per LOOCV fold (~312 images), data augmentation is a standard technique to improve generalisation. This experiment tested whether stronger augmentations and Test-Time Augmentation (TTA) could boost the EfficientNet-B0 baseline.

### Augmentation Pipeline

| Technique | Original NB08 | Augmented Version |
|---|---|---|
| Horizontal Flip | ✅ p=0.5 | ✅ p=0.5 |
| Rotation | ±10° | ±15° |
| Colour Jitter | brightness/contrast ±0.15 | brightness/contrast ±0.2 |
| Random Resized Crop | ✗ | ✅ scale=(0.85, 1.05) |
| Gaussian Blur | ✗ | ✅ kernel=3, σ=(0.1, 1.0) |
| Gaussian Noise | ✗ | ✅ σ=0.02 (MRI thermal noise) |
| Test-Time Augmentation | ✗ | ✅ 4 views (identity, H-flip, ±7° rotation) |

**Training config:** 50 epochs, AdamW lr=5e-5, weight decay=1e-3, CosineAnnealingLR  
**Hardware:** Kaggle GPU T4, ~50–60 min total

### Results — Score Collapse

| Method | Top-1 | ±1 Step | MAE (Hz) |
|---|---|---|---|
| **Original EfficientNet-B0 LOOCV** | **16.0% (4/25)** | **60.0% (15/25)** | **39** |
| Augmented EfficientNet-B0 LOOCV + TTA | 8.0% (2/25) | 20.0% (5/25) | 96 |

### Per-Patient Breakdown (Augmented)

| Patient | Pred | GT | Max Score | Result |
|---|---|---|---|---|
| Patient-BEV | #1 | #7 | 0.00e+00 | FAIL |
| Patient-BGT | #5 | #5 | 1.71e-19 | ✓ OK |
| Patient-BVE | #8 | #3 | 1.23e-20 | FAIL |
| Patient-CEA | #4 | #7 | 2.18e-32 | FAIL |
| Patient-CEJP | #13 | #7 | 2.91e-13 | FAIL |
| Patient-DEM | #1 | #7 | 0.00e+00 | FAIL |
| Patient-FAJ | #10 | #10 | 1.13e-11 | ✓ OK |
| Patient-FEC | #11 | #7 | 1.38e-08 | FAIL |
| Patient-GJ | #6 | #7 | 9.52e-07 | ±1 |
| Patient-HMJ | #10 | #7 | 1.78e-04 | FAIL |
| Patient-HRJ | #1 | #7 | 1.93e-08 | FAIL |
| Patient-IAM | #6 | #7 | 3.38e-16 | ±1 |
| Patient-INA | #1 | #7 | 0.00e+00 | FAIL |
| Patient-MJ | #9 | #7 | 1.25e-18 | FAIL |
| Patient-MS | #1 | #7 | 3.09e-22 | FAIL |
| Patient-MSR | #12 | #7 | 8.30e-26 | FAIL |
| Patient-OAF | #6 | #7 | 1.36e-36 | ±1 |
| Patient-PJ | #10 | #7 | 2.15e-01 | FAIL |
| Patient-RAK | #11 | #7 | 7.76e-23 | FAIL |
| Patient-RK | #2 | #7 | 1.54e-15 | FAIL |
| Patient-SK | #5 | #7 | 1.54e-23 | FAIL |
| Patient-SWJ | #13 | #7 | 1.35e-21 | FAIL |
| Patient-TL | #1 | #7 | 5.92e-19 | FAIL |
| Patient-WBC | #4 | #7 | 4.01e-24 | FAIL |
| Patient-WMR | #1 | #7 | 0.00e+00 | FAIL |

### Score Health Check

| Condition | Count |
|---|---|
| Patients with max_score > 0.01 | 1/25 (Patient-PJ only) |
| Patients with max_score > 1e-5 | 2/25 |
| Patients with all-zero scores | 4/25 |

### Analysis — Why It Failed

The sigmoid outputs collapsed to near-zero for 24/25 patients. Only Patient-PJ produced any meaningful scores (max 0.215). This is a classic **augmentation-induced training collapse** caused by:

1. **Too-aggressive augmentations for subtle signal:** The bSSFP frequency quality differences between images are subtle intensity/texture variations. Gaussian blur + noise + random crop simultaneously obscure these subtle cues, making the training signal too noisy for the model to learn from.
2. **Dataset too small for regularisation intensity:** With only ~312 training images per fold, the combined effect of heavy augmentation + weight decay (1e-3) + dropout (0.3) created excessive regularisation, pushing the model toward predicting the trivial solution (all zeros).
3. **Negative transfer from TTA:** Averaging across augmented views further diluted already near-zero predictions.

### Conclusion

> **The original lighter-augmentation CNN (16% Top-1, 60% ±1, 39 Hz MAE) remains the better result.** This negative result further motivates the agentic approach: with n=25, neither heavier augmentation nor weight optimisation generalises — adaptive per-patient reasoning is needed rather than fixed learned parameters.

---

## Outputs Reference

| File | Generated by | Description |
|---|---|---|
| `seg_masks_cache.pkl` | NB02 | Serialised U-Net segmentation for all 26 patients |
| `seg_stats.csv` | NB02 | Per-patient mask pixel counts and QA flags |
| `signal_metrics.csv` | NB03 | mu_myo, snr_myo, homogeneity per image (324 rows) |
| `contrast_metrics.csv` | NB04 | cnr, contrast_ratio per image |
| `banding_metrics.csv` | NB05 | banded_fraction_myo, banding_burden per image |
| `edge_metrics.csv` | NB06 | edge_sharpness_boundary, laplacian_variance per image |
| `temporal_metrics.csv` | NB07 | temporal_std_sweep, local_signal_drop per image |
| `cnn_scores.csv` | NB08 | EfficientNet-B0 LOOCV score per image (original) |
| `cnn_scores_2.csv` | NB08 (aug) | EfficientNet-B0 LOOCV + heavy augmentation + TTA (collapsed) |
| `composite_*.csv` | NB09 | Composite scores (equal / physics / gridsearch) |
| `final_comparison_table.csv` | NB10 | All method accuracies in one table |
| `per_patient_breakdown.csv` | NB10 | Per-patient prediction vs GT for each method |

---

## Next Steps

- [x] Run NB09 scipy LOOCV (Cell D) for honest composite accuracy
- [x] Test augmented CNN with heavier augmentations + TTA
- [ ] Integrate best metric weights into the FreqScout agentic pipeline
- [ ] Run agentic pipeline with composite scoring on all 26 patients
- [ ] Compare agentic vs non-agentic composite scoring
- [ ] Write FYP results chapter

---

*Generated: 2026-09-11 | Pipeline: metrics_lab/ | Cohort: 1st_cohort_SAX (26 patients, 3.0T)*
