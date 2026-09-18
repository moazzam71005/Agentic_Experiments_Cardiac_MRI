# Multi-Tool Feature Extraction & Clinical Relevance Guide
### Automated bSSFP Frequency Offset Selection at 3.0T Cardiac MRI

---

## 1. Executive Summary & Clinical Context

In 3.0T balanced Steady-State Free Precession (bSSFP) cardiac CINE MRI, radiofrequency (RF) frequency offset selection is a critical diagnostic step. $B_0$ main magnetic field inhomogeneities—caused by susceptibility differences between cardiac tissue, lungs, and stomach air—induce off-resonance phase accumulation $\theta = 2\pi \Delta f \text{TR}$. When phase accumulation reaches $\pm \pi$, destructive interference creates **dark banding artifacts** (signal nulls).

If a dark band cuts through the left ventricular (LV) myocardium or endocardial border, it causes localized signal voids that mimic myocardial thinning, pseudo-infarction, or wall motion abnormalities. 

To automate frequency selection, our framework extracts quantitative perceptual features across **7 specialized tool modules** (validated across Notebooks NB01–NB10 in `metrics_lab/`). This document provides a comprehensive breakdown of all extracted features, their mathematical formulas, physical units, and their direct **clinical relevance** to cardiologists and radiographers.

---

## 2. Comprehensive Tool-by-Tool Feature Extraction Matrix

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               MULTI-TOOL FEATURE EXTRACTION PIPELINE                             │
│                                                                                                  │
│  [1. LVSA U-Net]        --> Consensus Myocardium & Blood Pool Masks                              │
│  [2. Banding Detector]  --> Banded Fraction (BFM), Band Depth, AHA Wall Sectors                    │
│  [3. Signal Uniformity] --> Myocardial Homogeneity (1 - CoV), Mean Intensity                     │
│  [4. Contrast Metrics]  --> Blood-to-Myocardium CNR, Contrast Ratio                                │
│  [5. Temporal Sweep]    --> Sweep Standard Deviation, Local Signal Drop                              │
│  [6. Deep Scorer]       --> EfficientNet-B0 Out-of-Fold LOOCV Probabilities                       │
│  [7. Passband Model]    --> Gaussian Center Prior, Passband Plateau, Hard Vetoes                   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Tool 1: LVSA U-Net Consensus Segmentation Mask Generator
*Source: `metrics_lab/utils/seg_wrapper.py` & `Unet_LVSA_best.pkl` (advchain UNet_64)*

#### Extracted Features & Metrics
- `M_myo`: Binary 2D mask of the left ventricular myocardium.
- `M_pool`: Binary 2D mask of the LV blood pool.
- `M_rv`: Binary 2D mask of the right ventricle.
- `myo_pixel_count`: Total pixel area of the myocardium across the frame.
- `consensus_mask`: Majority-vote mask generated across all 13 frequency offset images.

#### Mathematical Formulation
$$\mathcal{M}_{\text{myo}}(x,y) = \text{mode}_{i=1..13} \left( \mathcal{M}_i(x,y) \right)$$

#### Clinical Relevance & Grounding
- **Spatial Target Scoping:** Raw whole-image metrics are misleading because off-resonance banding in non-cardiac structures (e.g. chest wall, subcutaneous fat) does not impact cardiac diagnosis. By calculating all subsequent metrics strictly within $\mathcal{M}_{\text{myo}}$, the system isolates intra-cardiac artifacts.
- **QA & Atypical Scan Detection:** If `myo_pixel_count` $= 0$ (as in Patient-KP, an apical slice), the system immediately flags the scan as atypical, preventing faulty processing on invalid anatomy.

---

### Tool 2: Intra-Myocardial Banding Detector & AHA Sector Localizer
*Source: `metrics_lab/nb05_banding_detection.ipynb` & `freqscout_agent/tools/banding.py`*

#### Extracted Features & Metrics
- `banded_fraction_myo` ($\text{BFM}$): Fraction of myocardial pixels intersected by signal nulls ($0.0$ to $1.0$).
- `band_trough_depth`: Severity of signal drop within the band null ($5^{\text{th}}$ percentile intensity).
- `band_present`: Binary flag indicating presence of intra-myocardial banding ($1$ or $0$).
- `band_sectors`: List of affected American Heart Association (AHA) 16-segment wall sectors.

#### Mathematical Formulation
$$\text{BFM}(i) = \frac{|\mathcal{M}_{\text{myo}} \cap \mathcal{A}_{\text{band}, i}|}{|\mathcal{M}_{\text{myo}}|}, \quad \mathcal{A}_{\text{band}, i} = \{ (x,y) \mid \mathcal{I}_i(x,y) < \tau_{\text{local}}(x,y) \}$$

#### Clinical Relevance & Grounding
- **Direct Physical Distortion Measure:** $\text{BFM}$ is the single strongest physical metric (40.0% Top-1 standalone). It directly measures how much diagnostic myocardial tissue is destroyed by off-resonance nulls.
- **Anatomical Localization:** Citing generic "banding" is insufficient for a radiologist. Mapping $\mathcal{A}_{\text{band}}$ to AHA sectors (e.g. *anterior*, *inferolateral*, *inferoseptal*) pinpoints exactly which wall segment is obscured.
- **Physics Veto Safety Trigger:** If $\text{BFM} > 0.08$ (more than 8% of myocardium corrupted), the system triggers an automatic **Physics Veto**, preventing the selection of dangerous frames even if deep learning scores them high.

---

### Tool 3: Myocardial Signal Uniformity & Homogeneity Engine
*Source: `metrics_lab/nb03_signal_metrics.ipynb` & `freqscout_agent/tools/quality.py`*

#### Extracted Features & Metrics
- `mu_myo`: Mean pixel intensity within the myocardium mask.
- `sigma_myo`: Standard deviation of pixel intensity within the myocardium mask.
- `cov_myo`: Coefficient of Variation ($\text{CoV} = \sigma / \mu$).
- `homogeneity`: Myocardial signal uniformity ($H = 1 - \text{CoV}_{\text{myo}}$).

#### Mathematical Formulation
$$\text{CoV}_{\text{myo}}(i) = \frac{\sigma_{\text{myo}}(i)}{\mu_{\text{myo}}(i)}, \quad H(i) = 1 - \text{CoV}_{\text{myo}}(i)$$

#### Clinical Relevance & Grounding
- **Early Off-Resonance Shading Detection:** At optimal frequency, healthy myocardium exhibits uniform brightness. Off-resonance gradients cause regional signal shading before full dark bands appear.
- **Superiority over Raw Intensity:** Raw mean signal ($\mu_{\text{myo}}$) performs poorly (4% Top-1) because high-intensity artifacts inflate the mean. Homogeneity ($H = 1 - \text{CoV}$) captures intensity distribution variance, achieving **44.0% Top-1 standalone**.

---

### Tool 4: Blood-to-Myocardium Contrast Metrics
*Source: `metrics_lab/nb04_contrast_metrics.ipynb` & `freqscout_agent/tools/quality.py`*

#### Extracted Features & Metrics
- `mu_blood`: Mean pixel intensity within the LV blood pool mask.
- `cnr`: Contrast-to-Noise Ratio between LV blood pool and myocardium.
- `contrast_ratio` ($\text{CR}$): Ratio of blood signal to myocardium signal ($\mu_{\text{blood}} / \mu_{\text{myo}}$).

#### Mathematical Formulation
$$\text{CNR}(i) = \frac{|\mu_{\text{blood}}(i) - \mu_{\text{myo}}(i)|}{\sigma_{\text{bg}}(i)}, \quad \text{CR}(i) = \frac{\mu_{\text{blood}}(i)}{\mu_{\text{myo}}(i)}$$

#### Clinical Relevance & Grounding
- **Endocardial Border Delineation:** Automated trabeculation and ejection fraction (EF) software requires crisp contrast between blood and myocardium. 
- **Boundary Verification:** Ensures that candidate offsets maintain adequate blood-pool brightness relative to the myocardium, avoiding frames where blood signal drops due to flow dephasing.

---

### Tool 5: Temporal Sweep Stability Analyzer
*Source: `metrics_lab/nb07_temporal_stability.ipynb`*

#### Extracted Features & Metrics
- `temporal_std_sweep`: Pixelwise standard deviation of intensity across all 13 sweep frames.
- `local_signal_drop`: Rate of localized intensity attenuation relative to adjacent frequency frames.

#### Mathematical Formulation
$$\text{Std}_{\text{sweep}}(x,y) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( \mathcal{I}_i(x,y) - \bar{\mathcal{I}}(x,y) \right)^2}$$

#### Clinical Relevance & Grounding
- **Passband Boundary Pinpointing:** Pixels in the flat passband remain stable across adjacent frequency offsets, whereas pixels in the transition zone fluctuate wildly as frequency changes. High temporal variation flags unstable offsets near band edges.

---

### Tool 6: Deep Learning Perceptual Scorer (EfficientNet-B0 LOOCV)
*Source: `metrics_lab/nb08_cnn_baseline.ipynb` & `freqscout_agent/tools/neural_scorer.py`*

#### Extracted Features & Metrics
- `cnn_score`: Softmax-normalized probability assigned to frame $i$ under Leave-One-Out Cross-Validation.
- `top1_idx`, `top3_idx`: Deep learning visual ranking of candidate frames.

#### Mathematical Formulation
$$S_{\text{CNN}}(i) = \frac{\exp(z_i)}{\sum_{j=1}^{13} \exp(z_j)}$$
*Trained with soft Gaussian labels ($\sigma=1.0$) centered on ground truth, AdamW ($lr=1e-4$), Cosine Annealing.*

#### Clinical Relevance & Grounding
- **Global Visual Aesthetics:** While hand-crafted physical metrics measure specific artifacts, the CNN evaluates overall image quality, structural sharpness, and global stack aesthetics. It achieves **44.0% Top-1 standalone** (nearly double DINOv2's 23.1%).

---

### Tool 7: bSSFP Passband Estimator & Center Frequency Prior
*Source: `freqscout_agent/tools/passband.py` & `metrics_lab/nb09_multimetric_scoring.ipynb`*

#### Extracted Features & Metrics
- `center_prior` ($P_c$): Gaussian prior probability favoring offsets near $0\text{ Hz}$.
- `passband_plateau`: Contiguous window of adjacent frames within $\pm 25\text{ Hz}$ of optimum.
- `acceptable_set`: Set of clinically diagnostic alternative offsets.
- `rejected_set`: Set of offsets vetoed due to intra-cardiac banding.

#### Mathematical Formulation
$$P_c(f_i) = \exp\left( -\frac{1}{2} \left( \frac{f_i}{100} \right)^2 \right)$$

#### Clinical Relevance & Grounding
- **Incorporating Scanner Physics:** Scanner shimming routines attempt to place resonance near $0\text{ Hz}$. The Gaussian center prior ($\sigma=100\text{ Hz}$) prevents the model from choosing extreme sweep edges ($-150\text{ Hz}$ or $+150\text{ Hz}$) where banding has temporarily shifted out of the slice view.

---

## 3. Clinical Relevance of the 7-Block Radiologist Report

The LLM engine synthesizes the Stage-A structured findings into a **Stage-C Radiologist Report**. Every block directly addresses an operational requirement of the scanning session:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        7-BLOCK RADIOLOGIST REPORT STRUCTURE                            │
│                                                                                        │
│  Block 1: CLINICAL INDICATION & OBJECTIVE      --> Protocol & diagnostic purpose       │
│  Block 2: PRIMARY FREQUENCY RECOMMENDATION     --> Recommended Hz & image #            │
│  Block 3: DIAGNOSTIC OPERATING WINDOW          --> Passband plateau (±25 Hz tolerance) │
│  Block 4: REJECTED OFFSETS & AHA LOCALIZATION  --> Vetoed frames & affected wall sites │
│  Block 5: MULTI-SOURCE EVIDENCE RECONCILIATION --> CNN vs Physics agreement / veto     │
│  Block 6: TECHNICAL LIMITATIONS & PITFALLS     --> Air-tissue susceptibility warnings  │
│  Block 7: SUMMARY & OPERATOR DIRECTIVE         --> Actionable scanner frequency dial   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Section Breakdown & Clinical Grounding

#### 1. CLINICAL INDICATION & OBJECTIVE
- **Clinical Content:** States the patient ID, magnetic field strength (3.0T), imaging plane (SAX CINE), and diagnostic objective (mitigating off-resonance dark banding while preserving wall motion definition).
- **Why It Matters:** Establishes standard medical record context for audit compliance.

#### 2. PRIMARY FREQUENCY RECOMMENDATION
- **Clinical Content:** Specifies the primary recommended offset in Hz (e.g. `+0.0 Hz`, Image `#7`), composite score, and confidence level (`HIGH`, `MEDIUM`, `LOW`).
- **Why It Matters:** Gives the radiographer the exact frequency adjustment to input into the scanner console.

#### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
- **Clinical Content:** Lists all acceptable alternative offsets (e.g. `#6 (-25 Hz)` and `#8 (+25 Hz)`).
- **Why It Matters:** **Scanner Operating Tolerance.** In clinical practice, frequency offsets within $\pm 25\text{ Hz}$ lie on the flat bSSFP passband plateau and are diagnostically equivalent. If minor patient motion occurs, the radiographer knows the safe operating margin without re-scouting.

#### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
- **Clinical Content:** Lists vetoed offsets and explicitly names the affected AHA myocardial sectors (e.g. `Image #1 (-150 Hz): 12.3% banding in anterior wall`).
- **Why It Matters:** **Diagnostic Safety.** Warns the radiologist which offsets cause false-positive wall thinning or pseudo-infarction, preventing misdiagnosis.

#### 5. MULTI-SOURCE EVIDENCE RECONCILIATION
- **Clinical Content:** Explains agreement/disagreements between deep learning (CNN), physics artifact metrics ($\text{BFM}$), and homogeneity. Details why a **Physics Veto** was applied if the CNN picked a banded frame.
- **Why It Matters:** **AI Transparency & Safety.** Demonstrates to the physician that the AI system has built-in physical safeguards and will not blindly trust a deep learning model if intra-cardiac banding is physically present.

#### 6. TECHNICAL LIMITATIONS & PITFALLS
- **Clinical Content:** Flags localized susceptibility risks near lungs or stomach air pockets.
- **Why It Matters:** Prompts the operator to inspect high-risk anatomical boundaries manually if severe susceptibility gradients exist.

#### 7. SUMMARY & OPERATOR DIRECTIVE
- **Clinical Content:** Actionable summary line: `OPERATOR ACTION: Set scanner center frequency offset to +0.0 Hz (Image #7)`.
- **Why It Matters:** Immediate, unambiguous execution instruction for the radiographer at the console.

---

## 4. Automated Faithfulness & Zero-Hallucination Audit

To guarantee patient safety, every generated report is evaluated by an automated `ClinicalAudit` module (`metrics_lab/agentic_reasoning/eval/audit.py`):

1. **Frequency Faithfulness Check:** Scans the text for all numerical $\text{Hz}$ values and verifies that every single cited frequency exists in the patient's actual scout stack ($100\%$ score).
2. **Anatomical Sector Grounding Check:** Cross-references mentioned wall names (*anterior*, *inferior*, *lateral*, *septal*) against the physical U-Net segmentation intersection map ($100\%$ score).
3. **Section Coverage Check:** Confirms that all 7 required clinical blocks are fully present ($100\%$ coverage).
4. **Hallucination Rate:** Measured at **0.0%** across the entire 25-patient evaluation cohort.

---

*Document Summary: Prepared for FYP Thesis & Journal Submission | Repository: `metrics_lab/` & `metrics_lab/agentic_reasoning/`*
