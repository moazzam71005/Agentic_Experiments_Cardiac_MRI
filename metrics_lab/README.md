# FreqScout — Metrics Lab

Independent validation of every image-quality metric for automatic bSSFP CINE frequency-offset selection.

## Goal

Before plugging metrics into any agentic pipeline, validate each one individually:
- Does it correlate with the GT selection?
- Does argmax/argmin predict correctly?
- Which offset does it prefer, and why?

Once confident, combine them into a transparent **multi-metric composite score** (nb09),
then evaluate against the prior pipeline (Exp10: 42.3% Top-1, 76.9% ±1).

## Dataset

| Field | Value |
|---|---|
| Patients | 26 (Patient-BEV … Patient-WMR) |
| Images per patient | 13 single-frame DICOMs |
| Frequency sweep | −150 Hz to +150 Hz, step 25 Hz |
| Plane | Short-Axis (SAX) bSSFP CINE |
| GT file | `../1stcohortSAX (1).xlsx` → `image no` column (1-indexed) |
| Data root | `../Dr-Dataset/1st_cohort_SAX/` |
| Segmentation checkpoint | `../Unet_LVSA_best.pkl` |

**Offset mapping:** image_no (1-indexed) → Hz = −150 + (image_no − 1) × 25

**Special case:** Patient-KP has only 7 images (step 50 Hz); exclude from headline metrics.

## Notebook Execution Order

```
nb01 → nb02 (saves seg cache) → nb03, nb04, nb05, nb06, nb07 (any order)
     → nb08 (Kaggle GPU, produces cnn_scores.csv)
     → nb09 (needs all CSVs) → nb10 (final eval)
```

## Segmentation Labels (LVSA U-Net)

| Label | Region | Colour in overlays |
|---|---|---|
| 0 | Background | — |
| 1 | RV | Yellow |
| 2 | Myocardium | Green |
| 3 | LV Blood Pool | Red/Blue |

## Results Summary (fill in as notebooks run)

| Method | Top-1 | ±1 step | MAE (Hz) |
|---|---|---|---|
| argmax(mu_myo) — nb03 | TBD | TBD | TBD |
| argmax(cnr) — nb04 | TBD | TBD | TBD |
| argmin(banded_fraction_myo) — nb05 | TBD | TBD | TBD |
| argmax(edge_sharpness_myo) — nb06 | TBD | TBD | TBD |
| argmin(temporal_std) — nb07 | TBD | TBD | TBD |
| CNN EfficientNet-B0 LOOCV — nb08 | TBD | TBD | TBD |
| Equal-weight composite — nb09 | TBD | TBD | TBD |
| LOOCV-optimised composite — nb09 | TBD | TBD | TBD |
| **Prior pipeline (Exp10)** | **42.3%** | **76.9%** | **22.1 Hz** |
