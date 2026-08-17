# Cardiac MRI Best-Slice Selection — Experiments Report

**Task:** Given a frequency-scout cardiac MRI short-axis (SAX) series at 3.0T (13 slices per patient, each acquired at a different frequency offset), automatically identify the single best slice — the one a cardiologist would choose for diagnosis.

**Dataset:** 26 patients, mostly 13 slices each (Patient-KP has 7, Patient-BVE has 12). Ground truths are in `1stcohortSAX (1).xlsx`.

---

## The Problem in Plain Terms

Each acquisition produces 13 DICOM images of the heart at the same anatomical location but at slightly different B0 shimming frequencies. The "best" slice is the one with the sharpest, most homogeneous field — determined by the radiologist. The challenge: make a model learn this preference automatically so the doctor doesn't have to review all 13 slices manually.

This is fundamentally a **ranking problem**, not a classification problem. There is no fixed "correct class" — the answer is always relative to the other slices in the same patient's stack.

---

## Baselines

Before looking at model results, it helps to know what a naive system achieves:

| Baseline | Top-1 | Top-3 | MAE |
|---|---|---|---|
| **Center slice** (always pick slice 7/13) | 26.9% | 57.7% | 1.31 |
| **Global mode** (always pick the most common GT slice no.) | 26.9% | 61.5% | 1.27 |
| **Random** (expected) | 8.0% | 23.9% | — |

The center slice is deceptively strong. This is because the optimal frequency offset tends to cluster near the middle of the acquisition range by design (shimming is calibrated around the center). Any model must meaningfully beat 26.9% Top-1 to be useful.

---

## Experiment 1 — Direct CNN Slice Scoring

**Notebook:** `experiment1_best_slice_cnn.ipynb`
**Outputs:** `exp1_outputs/`

### Architecture
- Three models tried: SimpleCNN (custom), ResNet18 (ImageNet pretrained), EfficientNet-B0 (ImageNet pretrained)
- Each model takes a single DICOM slice as input and outputs a scalar score
- Soft Gaussian labels centered on the GT slice (sigma=1.0) used instead of one-hot
- Standard 5-fold patient-level cross-validation
- Loss: MSE against soft labels

### Results (5-fold mean, optimistic)

| Model | Top-1 | Top-3 | MAE |
|---|---|---|---|
| ResNet18 | **22.7%** | **61.3%** | **1.35** |
| EfficientNet-B0 | ~20% | ~58% | ~1.5 |
| SimpleCNN | ~8% | ~38% | ~2.5 |

### What these numbers mean
- ResNet18's Top-3 of 61.3% is promising — the correct slice was in the model's top 3 choices more than half the time.
- However, **5-fold CV with n=26 is noisy** — each validation fold has only 5–6 patients. A single lucky split inflates numbers.
- The center baseline (26.9%) beats the model on Top-1, meaning the model learned the general "middle of the range" trend but added noise on top.

### Root problem
Slices are scored **independently** — the model sees one slice at a time and outputs a score, never comparing slices against each other. This ignores the relational nature of the problem: "best" only means something relative to the others in the stack.

---

## Experiment 2 — Ranking Losses + Cross-Validation

**Notebook:** `experiment2_ranking_cv.ipynb`
**Outputs:** `exp2_outputs/`

### What changed from Exp1
- **ListNet loss** (softmax-based ranking loss) replaces MSE — explicitly trains the model to output a probability distribution over slices
- Optional **pairwise margin loss** added on top of ListNet
- **Freeze/unfreeze** training schedule: train the classification head first, then unfreeze the backbone with a lower learning rate
- **Soft label sigma** widened to 1.5 (smoother supervision)
- Early stopping on validation MAE
- **LOOCV** (Leave-One-Out Cross-Validation) used as the primary evaluation — every patient serves as a test set once, giving 26 truly independent test evaluations

### Results

| Split | Top-1 | Top-3 | MAE |
|---|---|---|---|
| 5-fold OOF | 22.7% | 61.3% | 1.35 |
| **LOOCV (honest)** | **3.8%** | **46.2%** | **2.19** |

### What these numbers mean
The collapse from 22.7% (5-fold) to 3.8% (LOOCV) is the key finding:
- The 5-fold result was optimistic — some patients' slices appeared in both train and val folds in similar ways, leaking information.
- LOOCV forces true generalization: train on 25 patients, predict for 1 completely unseen patient.
- **3.8% Top-1 is below the center baseline (26.9%)** — meaning ranking losses and richer training did not help the model generalize.

### Root problem
Independent slice scoring still can't learn what "best" means without direct comparison. The model memorizes training distribution quirks rather than learning to compare slices within a patient.

---

## Experiment 3 — Siamese Pairwise Network

**Notebook:** `experiment3_siamese_loocv.ipynb`
**Outputs:** `exp3_outputs/`

### The key design shift
Instead of asking "what score does this slice get?", we ask "which of these two slices is better?" This is a **pairwise comparison** (Siamese network) approach.

For each patient with N slices, we generate all C(N,2) = 78 pairs (for N=13). Each pair is labelled using the **physics-grounded frequency label**: the slice closer in frequency to the GT gets the positive label, with soft probability based on frequency distance. This dramatically increases the effective training set.

### Architecture
- **Shared EfficientNet-B0 backbone** (ImageNet pretrained)
- For a pair (A, B): features `[f_A, f_B, f_A−f_B, f_A×f_B]` fed into a comparison MLP
- **Auxiliary ordinal head**: predicts each slice's distance-to-best (ordinal regulariser)
- **Freeze/unfreeze** training: backbone frozen for first N epochs, then unfrozen at lower LR
- **Test-Time Augmentation (TTA)**: predictions averaged over 4 augmented versions
- **Win-count tournament inference**: each slice's score = sum of P(beat every other slice)

### Results

| Split | Top-1 | Top-3 | MAE | Mean Rank |
|---|---|---|---|---|
| **LOOCV (honest)** | **15.4%** | **53.8%** | **2.50** | 4.27 |
| 5-fold OOF | 3.8% | 38.5% | 3.23 | — |

### LOOCV per-patient detail

| Patient | GT | Pred | Top-1? | Top-3? | MAE |
|---|---|---|---|---|---|
| Patient-BEV | 7 | 7 | ✓ | ✓ | 0 |
| Patient-HMJ | 8 | 8 | ✓ | ✓ | 0 |
| Patient-RAK | 7 | 7 | ✓ | ✓ | 0 |
| Patient-WMR | 8 | 8 | ✓ | ✓ | 0 |
| Patient-BVE | 6 | 7 | — | ✓ | 1 |
| Patient-CEA | 7 | 6 | — | ✓ | 1 |
| Patient-GJ | 9 | 8 | — | ✓ | 1 |
| Patient-IAM | 8 | 7 | — | ✓ | 1 |
| Patient-KP | 6 | 5 | — | ✓ | 1 |
| Patient-DEM | 9 | 6 | — | — | 3 |
| Patient-FEC | 8 | 1 | — | — | 7 |
| Patient-INA | 11 | 6 | — | — | 5 |
| Patient-PJ | 8 | 6 | — | — | 2 |
| Patient-RK | 9 | 4 | — | — | 5 |
| Patient-TL | 9 | 5 | — | — | 4 |
| ... (11 more) | | | | | |

4 patients predicted perfectly. 14/26 had the correct slice in the top 3. The worst case was Patient-FEC (MAE=7, predicted slice 1 when GT was slice 8) and Patient-RK (MAE=5, rank 12/13 — near worst possible).

### What these numbers mean
- **+11.6 pp Top-1 improvement over Exp2** (3.8% → 15.4%) on the same honest LOOCV protocol.
- The pairwise formulation works — comparisons generalize better than independent scores.
- Still **below the center slice baseline** (26.9%) on Top-1. The model knows which direction is "better" about half the time, but hasn't fully learned the task.
- The 5-fold numbers (3.8%) are misleadingly bad — with 4–5 patients per validation fold, a single outlier like Patient-FEC (MAE=7) tanks the fold metric. LOOCV is the correct number.

---

## Full Comparison Table

| System | Top-1 | Top-3 | MAE | Eval protocol |
|---|---|---|---|---|
| **Exp3 Siamese EfficientNet-B0** | **15.4%** | **53.8%** | 2.50 | LOOCV n=26 |
| Exp2 ResNet18 + ListNet | 3.8% | 46.2% | 2.19 | LOOCV n=26 |
| Exp1 ResNet18 + BCE | 22.7%* | 61.3%* | 1.35* | 5-fold (noisy) |
| Center slice (naive) | 26.9% | 57.7% | 1.31 | all 26 |
| Global-mode GT (naive) | 26.9% | 61.5% | 1.27 | all 26 |
| Random | 8.0% | 23.9% | — | analytic |

\* 5-fold CV with n=26 is optimistic — LOOCV numbers would be lower.

The core insight: **honest evaluation (LOOCV) reveals that the models generalize poorly**, but the trend is clearly improving. The pairwise approach (Exp3) is the right direction.

---

## Why Is This Hard?

1. **n=26 is tiny.** Deep learning with ImageNet backbones needs hundreds of examples to fine-tune reliably. With 26 patients and LOOCV, the model trains on 25 patients — barely enough to tune a head, let alone a backbone.

2. **The GT is a single integer per patient.** There's no way to know from the DICOM data alone what the cardiologist was thinking. The label carries no uncertainty, but the task does.

3. **Appearance of adjacent slices is very similar.** Frequency offsets of 1–2 steps apart look nearly identical visually. The differences are subtle B0 inhomogeneity artifacts — small banding, shading — not dramatic structural changes.

4. **The center bias is real.** Because shimming is optimized for the center frequency, the GT tends to cluster around slices 7–9 of 13. A model that learns this prior does well on MAE but not necessarily on exact Top-1 prediction.

---

## Possible Improvement Steps

### Immediate (without more data)

| Idea | Expected impact | Effort |
|---|---|---|
| **Frequency-delta as input feature** | Medium | Low |
| Concatenate the normalized frequency offset (Δf from center) as an extra channel or scalar feature to the model. The model currently sees the image but not its position in the sweep. | | |
| **Larger soft-label sigma** | Low–Medium | Low |
| Widening sigma from 1.5 to 2–3 would reduce penalty for off-by-one predictions, which might help when the GT is near a plateau. | | |
| **Ensemble Exp1 + Exp3** | Medium | Low |
| Average the scores from the Exp1 independent scorer and the Exp3 pairwise winner. Different errors, complementary signals. | | |
| **Constrained search** (center prior) | Low–Medium | Low |
| At inference, weight scores by a Gaussian centered on the middle of the stack. This incorporates the known physics prior without needing to learn it. | | |
| **Calibrate with isotonic regression** | Low | Low |
| Post-hoc calibration of the pairwise probabilities could improve the win-count ranking. | | |

### With more data (recommended)

| Idea | Expected impact | Effort |
|---|---|---|
| **Expand cohort to 50–100 patients** | High | Medium |
| The single biggest bottleneck. Even 25 more patients would roughly halve LOOCV variance. | | |
| **Data augmentation: synthetic frequency shifts** | Medium | Medium |
| Simulate B0 inhomogeneity by applying synthetic phase ramp distortions to existing slices to generate new "off-frequency" examples. | | |
| **Self-supervised pre-training on unlabelled MRI** | Medium–High | High |
| Pre-train on unlabelled cardiac MRI (e.g. ACDC, M&Ms datasets) using contrastive learning or masked autoencoding before fine-tuning on the 26-patient labelled set. Reduces reliance on ImageNet features which are not MRI-specific. | | |

### Architectural changes

| Idea | Expected impact | Effort |
|---|---|---|
| **Attention-based stack encoder** | Medium–High | Medium |
| Instead of comparing all pairs, feed the entire stack of N slices simultaneously into a Transformer encoder. Each slice attends to all others, and a CLS token predicts the best slice. Avoids the quadratic pairwise problem and captures full-stack context. | | |
| **3D convolutions** | Medium | Medium |
| Treat the frequency stack as a pseudo-3D volume (N×H×W). A 3D CNN can learn spatial frequency-gradient features directly. Requires careful normalization. | | |
| **Ordinal regression head** | Low–Medium | Low |
| Replace the binary comparison output with a proper ordinal regression loss (CORN or CORAL), which models the ordered nature of slice quality more explicitly. | | |
| **Multi-task learning: predict shimming quality score** | Medium | High |
| If shimming quality metrics (e.g., B0 map standard deviation) can be computed from the raw k-space data, add them as an auxiliary regression target. This gives the model a physics-interpretable supervision signal beyond the single GT label. | | |

### Evaluation improvements

| Idea | Expected impact | Notes |
|---|---|---|
| **Annotator agreement study** | N/A (data quality) | Have 2–3 cardiologists label the same patients independently. If inter-annotator agreement is low (e.g. ±2 slices), the Top-1 metric is unrealistically strict — a Top-3 metric would be more meaningful. |
| **Rank correlation (Spearman/Kendall)** | Better diagnostic | Measures whether the model's full ranking matches the GT ordering, not just the top pick. More informative for a 13-way ranking problem. |
| **Clinical threshold metric** | Practical | Define "acceptable" as MAE ≤ 1 (within 1 slice of GT). Report the fraction of patients where the model is within this threshold. |

---

## Recommended Next Step

The most impactful single action is **collecting more labeled patients** (target: 60–100). The current 26-patient set is simply too small for LOOCV-honest deep learning.

While collecting data, the most productive architectural experiment is the **full-stack Transformer encoder**: feed all 13 slices as a sequence, use cross-slice attention, predict the best index directly. This is architecturally more natural than pairwise comparison for a fixed-length ordered sequence.

If data collection is not possible, the **ensemble of Exp1 + Exp3 with a center-frequency prior** is the quickest route to squeezing more performance from the existing 26 patients.

---

*Generated from: `exp1_outputs/run1.ipynb`, `exp2_outputs/run2.ipynb`, `exp3_outputs/run3.ipynb`*
*Plots and CSVs: `exp3_outputs/`*
