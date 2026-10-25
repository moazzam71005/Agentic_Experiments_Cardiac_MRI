# CLINICAL FREQUENCY SELECTION REPORT
**Patient ID:** Patient-CEJP
**Modality:** 3.0T Cardiac MRI – Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13-slice stack, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this automatic frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging. At 3.0T, the increased Larmor frequency exacerbates off-resonance effects, leading to characteristic dark banding artifacts (stripes) that can obscure myocardial tissue and compromise the assessment of wall motion and ejection fraction. The primary goal is to mitigate these dark banding stripes while simultaneously preserving high signal-to-noise ratio (SNR) and robust blood-myocardium contrast, ensuring diagnostic image quality for functional evaluation.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** **-25.0 Hz**
**Corresponding Image:** **#6**
**Composite Score:** 0.9623
**Confidence:** High (0.82)

**Justification:**
Image #6 (-25.0 Hz) represents the global multi-metric peak for this patient. It demonstrates the lowest intra-myocardial banded fraction (2.37%) across the entire scout stack, indicating minimal signal loss due to off-resonance effects. Concurrently, it maintains a high signal homogeneity index (1-CoV) of 0.7038, ensuring uniform signal distribution within the myocardium. The neural confidence score (CNN) is also at its maximum (0.0872), reflecting strong agreement between the deep learning classifier and the physics-based artifact metrics. This offset provides the optimal balance between artifact suppression and signal integrity required for diagnostic CINE imaging.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The analysis identifies a "passband plateau" status, indicating a region of acceptable image quality surrounding the primary recommendation.

**Acceptable Alternative:**
*   **Image #4 (-75.0 Hz):**
    *   **Banded Fraction (Myo):** 3.85%
    *   **Homogeneity:** 0.7456
    *   **Delta from Primary:** -50.0 Hz

**Viability Explanation:**
Image #4 (-75.0 Hz) remains diagnostically viable for cine function evaluation. Although it is 50 Hz away from the primary recommendation, its banded fraction (3.85%) is still negligible, and it exhibits the highest homogeneity score in the stack (0.7456). This suggests that while -25.0 Hz is the global optimum, the frequency range from -75.0 Hz to -25.0 Hz constitutes a stable operating window where off-resonance artifacts are sufficiently suppressed to allow for accurate assessment of myocardial motion and wall thickness.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
Eleven of the thirteen scout images were rejected due to severe off-resonance dark banding intersecting the myocardium. The specific rejections and anatomical localizations are as follows:

*   **Negative Offsets (Left of Passband):**
    *   **#1 (-150.0 Hz):** Rejected. Banded fraction 12.3%. **Affected Sector: Anterior.**
    *   **#2 (-125.0 Hz):** Rejected. Banded fraction 9.04%. No specific sector identified, but global banding is significant.
    *   **#3 (-100.0 Hz):** Rejected. Banded fraction 11.56%. **Affected Sector: Anterior.**
    *   **#5 (-50.0 Hz):** Rejected. Banded fraction 9.93%. No specific sector identified, but homogeneity is degraded (0.5765).

*   **Positive Offsets (Right of Passband):**
    *   **#7 (0.0 Hz):** Rejected. Banded fraction 17.93%. **Affected Sector: Anterior.**
    *   **#8 (+25.0 Hz):** Rejected. Banded fraction 23.26%. No specific sector identified, but homogeneity is poor (0.4131).
    *   **#9 (+50.0 Hz):** Rejected. Banded fraction 27.56%. No specific sector identified.
    *   **#10 (+75.0 Hz):** Rejected. Banded fraction 29.04%