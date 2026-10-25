# Clinical Frequency Selection Report
**Patient ID:** Patient-BEV
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13-slice stack, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this automatic frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging. At 3.0T, the increased magnetic field strength amplifies susceptibility-induced off-resonance effects, leading to characteristic dark banding artifacts that can obscure myocardial tissue and compromise the assessment of wall motion and function. The primary goal is to mitigate these dark banding stripes while preserving high signal-to-noise ratio (SNR) and robust blood-myocardium contrast, ensuring diagnostic image quality for cardiac functional evaluation.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** **+0.0 Hz**
**Corresponding Image:** **#7**

**Justification:**
Image #7 (+0.0 Hz) is designated as the optimal frequency center based on a composite score of **0.9972**. This offset represents the global peak of the multi-metric evaluation:
*   **Minimal Banding:** It exhibits the lowest intra-myocardial banded fraction (**0.0037** or 0.37%), effectively eliminating off-resonance dark stripes within the myocardium.
*   **High Homogeneity:** The signal homogeneity (1-CoV) is **0.8300**, indicating uniform signal distribution across the myocardial wall.
*   **Neural Confidence:** The CNN classifier (EfficientNet-B0) assigns a high confidence score of **0.7952**, confirming the absence of significant artifacts.
*   **Consensus:** Both the deep learning model and the physics-based artifact metrics independently identify Image #7 as the optimal choice, resulting in a high confidence rating (**0.82**).

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The frequency response for Patient-BEV demonstrates a clear **passband plateau** centered at 0.0 Hz. This plateau indicates a range of frequencies where image quality remains diagnostically viable, providing flexibility for minor frequency drift or shimming adjustments during the scan.

**Acceptable Alternative Offsets:**
The following offsets fall within the clinical tolerance window (±25 Hz) and are considered acceptable alternatives:
*   **Image #6 (-25.0 Hz):**
    *   Composite Score: 0.9821
    *   Banded Fraction: 0.0061 (0.61%)
    *   Homogeneity: 0.8322
    *   *Note:* This slice actually exhibits the highest homogeneity in the stack, though with slightly higher banding than #7.
*   **Image #8 (+25.0 Hz):**
    *   Composite Score: 0.9717
    *   Banded Fraction: 0.0098 (0.98%)
    *   Homogeneity: 0.8212

**Viability:**
Both alternatives maintain negligible intra-myocardial banding (<1.0%) and high signal homogeneity (>0.82). They remain fully viable for cine function evaluation. However, Image #7 is preferred due to its superior balance of minimal banding and strong CNN consensus.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
Six offsets were strictly rejected due to severe off-resonance dark banding that compromises diagnostic utility.

**Rejected Offsets & Artifact Severity:**
1.  **Image #1 (-150.0 Hz):**
    *   Banded Fraction: **0.0897 (8.97%)**
    *   Affected Sectors: None explicitly localized (global/diffuse banding).
    *   Severity: Severe.
2.  **Image #2 (-125.0 Hz):**
    *   Banded Fraction: **0.0934 (9.34%)**
    *   Affected Sectors: None explicitly localized.
    *   Severity: Severe.
3.  **Image #10 (+75.0 Hz):**
    *   Banded Fraction: **0.1806 (18.06%)**
    *   Affected Sectors: **Anterior** wall.
    *   Severity: Severe. Significant signal loss in the anterior myocardium.
4.  **Image #11 (+100.0 Hz):**