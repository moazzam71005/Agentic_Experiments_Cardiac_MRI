# Clinical Frequency Selection Report
**Patient:** Patient-TL
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13 slices, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging to mitigate off-resonance artifacts. At 3.0T, the increased susceptibility to magnetic field inhomogeneities results in characteristic "dark banding" stripes that can obscure myocardial tissue and compromise the assessment of wall motion and ventricular function. The primary goal is to select a frequency offset that minimizes intra-myocardial signal voids (dark bands) while preserving high signal homogeneity and robust blood-myocardium contrast, thereby ensuring diagnostic image quality for cardiac functional evaluation.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** +25.0 Hz
**Corresponding Image:** #8

**Justification:**
Image #8 is designated as the optimal frequency setting based on a composite score of 0.9618 and high confidence (0.85). This selection is driven by the following evidence:
*   **Minimal Artifact Burden:** Image #8 exhibits the lowest banded fraction in the myocardium (BFM) among all viable candidates at **0.0155** (1.55%). This indicates that less than 2% of the myocardial volume is affected by off-resonance dark banding, effectively preserving tissue visibility.
*   **Signal Homogeneity:** The signal homogeneity (1-CoV) is **0.6797**, which is high and comparable to the peak homogeneity observed in the stack (0.6926 at 0 Hz). This ensures uniform signal intensity across the myocardium, critical for accurate tissue characterization.
*   **Neural Confidence:** The CNN score is **0.0881**, reflecting strong classifier confidence in the image quality.
*   **Arbitration Outcome:** While Image #7 (0 Hz) offers slightly higher homogeneity, Image #8 provides superior artifact reduction (lower BFM). Image #9 (+50 Hz), while having a higher CNN score (0.0902), was rejected as primary due to borderline banding (BFM 0.0757) and lower homogeneity (0.6084). Image #8 represents the optimal balance of artifact avoidance and signal integrity.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The diagnostic passband plateau for Patient-TL is centered around the 0 Hz to +25 Hz range, where both artifact burden and signal homogeneity remain within clinically acceptable limits.

**Acceptable Alternative:**
*   **Image #7 (+0.0 Hz):**
    *   **BFM:** 0.0172 (1.72%)
    *   **Homogeneity:** 0.6926
    *   **Composite Score:** 0.9246
    *   **Viability:** Image #7 is a viable alternative. It demonstrates the highest signal homogeneity in the stack (0.6926) and negligible banding (1.72%). It remains diagnostically viable for cine function evaluation as the artifact burden is minimal. However, it is ranked secondary to Image #8 due to the slightly higher BFM, which may present a marginally higher risk of subtle signal dropout in specific myocardial segments compared to the +25 Hz offset.

**Passband Characteristics:**
The passband is relatively narrow. Moving away from the 0 Hz to +25 Hz window in either direction results in a rapid degradation of image quality:
*   **Negative Direction (-25 Hz):** Image #6 shows a significant increase in BFM to 0.105 (10.5%) with septal involvement, dropping out of the acceptable range.
*   **Positive Direction (+50 Hz):** Image #9 shows a rise in BFM to 0.0757 (7.57%) and a drop in homogeneity to 0.6084, classifying it as suboptimal.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
The following offsets are strictly rejected due to severe off-resonance dark banding that compromises diagnostic utility.

**Rejected Offsets with Anatomical Localization:**
*   **Image #3 (-100.0 Hz):**
    *   **BFM:** 0.2392 (23.92%)
    *   **Affected