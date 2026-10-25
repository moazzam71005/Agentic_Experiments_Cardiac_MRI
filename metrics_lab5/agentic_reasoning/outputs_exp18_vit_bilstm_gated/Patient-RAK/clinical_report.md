# Clinical Frequency Selection Report
**Patient ID:** Patient-RAK
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13-slice stack, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this automatic frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging. At 3.0T, the increased chemical shift and susceptibility effects necessitate precise frequency tuning to mitigate off-resonance dark banding artifacts. The primary goal is to select a frequency offset that maximizes signal homogeneity within the myocardium while preserving adequate blood-myocardium contrast, thereby ensuring diagnostic image quality for cardiac functional assessment.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** +25.0 Hz
**Corresponding Image:** #8

**Justification:**
Image #8 (+25.0 Hz) is designated as the **Optimal** selection based on a composite score of 0.9955. This offset represents the global multi-metric peak, characterized by:
*   **Minimal Intra-myocardial Banding:** The banded fraction of the myocardium is negligible at 0.15% (0.0015), indicating virtually no dark banding artifacts intersecting the cardiac muscle.
*   **High Signal Homogeneity:** The homogeneity index (1-CoV) is 0.7791, reflecting uniform signal intensity across the myocardial wall.
*   **Strong Neural Confidence:** The CNN score of 0.0916 is the highest in the stack, indicating high confidence in the absence of significant artifacts.
*   **Low Temporal Instability:** The temporal standard deviation sweep is 0.0182, suggesting stable signal acquisition over the cardiac cycle.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The analysis identifies a **Passband Plateau** status, indicating a robust region of acceptable image quality surrounding the primary recommendation. The following offsets are classified as **Acceptable** alternatives within the clinical tolerance window:

*   **Image #7 (+0.0 Hz):**
    *   Banded Fraction: 0.59%
    *   Homogeneity: 0.7808
    *   *Note:* This offset offers the highest homogeneity in the stack but slightly higher banding than #8.
*   **Image #6 (-25.0 Hz):**
    *   Banded Fraction: 2.35%
    *   Homogeneity: 0.7309
    *   *Note:* This offset remains diagnostically viable with negligible banding and acceptable homogeneity.

These offsets form a contiguous passband from -25.0 Hz to +25.0 Hz. All three slices (#6, #7, #8) exhibit banded fractions below 2.5% and homogeneity indices above 0.73, ensuring reliable cine function evaluation. The primary recommendation of +25.0 Hz is preferred due to its superior composite score and minimal banding fraction.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
Nine offsets were strictly **Rejected** due to severe off-resonance dark banding or suboptimal signal characteristics.

**Critical Rejection with Anatomical Localization:**
*   **Image #10 (+75.0 Hz):**
    *   **Severity:** Severe off-resonance dark banding.
    *   **Banded Fraction:** 32.06% (highest in stack).
    *   **Affected Sectors:** **Anterior, Septal**.
    *   **Impact:** Significant signal loss in the anterior and septal walls, compromising assessment of these regions. Homogeneity is critically low at 0.3040.

**Other Rejected Offsets (Generalized Banding/Signal Degradation):**
*   **Image #1 (-150.0 Hz):** Banded fraction 18.38%; Homogeneity 0.4875.
*   **Image #2 (-125.0 Hz):** Banded fraction 30.00%; Homogeneity 0.3525.
*   **Image #3 (-100.0 Hz):** Banded fraction 20.74%; Homogeneity 0.4864.
*   **Image #4 (-75.0 Hz):** Banded fraction 12.50%; Homogeneity 0.6180.
*   **Image #5