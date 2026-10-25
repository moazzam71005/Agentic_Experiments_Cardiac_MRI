# Clinical Frequency Selection Report
**Patient ID:** Patient-GJ
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-Metric Frequency Scout (13-slice stack, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this automatic frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging. At 3.0T, the increased susceptibility to magnetic field inhomogeneities significantly elevates the risk of off-resonance artifacts, specifically dark banding stripes that intersect the myocardium. These artifacts degrade the signal-to-noise ratio (SNR) and obscure the blood-myocardium contrast essential for accurate ventricular function assessment. This report aims to mitigate these artifacts by selecting a frequency offset that maximizes intra-myocardial signal homogeneity while preserving diagnostic contrast, ensuring reliable evaluation of cardiac wall motion and ejection fraction.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** +25.0 Hz
**Corresponding Image:** #8
**Composite Score:** 0.9654
**Confidence:** High (0.82)

The primary recommendation is a center frequency offset of **+25.0 Hz** (Image #8). This offset represents the global multi-metric peak within the scout stack. It achieves a composite score of 0.9654, driven by a complete absence of intra-myocardial banding (banded fraction = 0.0%) and a high neural confidence score (CNN score = 0.0855). While the signal homogeneity (1-CoV) at this offset is 0.7655, which is slightly lower than adjacent slices, the total elimination of dark banding artifacts ensures that the myocardial wall boundaries remain clearly defined. The high CNN consensus indicates that the image quality is diagnostically superior to all other candidates, providing the most robust balance between artifact avoidance and signal integrity for clinical interpretation.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The diagnostic operating window for Patient-GJ is characterized by a stable **passband plateau** spanning from **-25.0 Hz to +75.0 Hz** (Images #6 through #10). Within this range, the myocardium is free from severe off-resonance banding, and signal homogeneity remains above 0.80 for most slices, ensuring consistent contrast.

The following offsets are identified as acceptable alternatives within the clinical tolerance window (±25 Hz of the primary recommendation or within the plateau):

*   **Image #6 (-25.0 Hz):** Banded fraction 0.0089; Homogeneity 0.8007.
*   **Image #7 (0.0 Hz):** Banded fraction 0.0163; Homogeneity 0.8126.
*   **Image #9 (+50.0 Hz):** Banded fraction 0.0000; Homogeneity 0.8614.
*   **Image #10 (+75.0 Hz):** Banded fraction 0.0134; Homogeneity 0.8596.

These alternatives remain diagnostically viable for cine function evaluation because they exhibit negligible intra-myocardial banding (≤1.6%) and maintain high signal homogeneity (≥0.80). Specifically, Image #9 (+50.0 Hz) offers the highest homogeneity (0.8614) and zero banding, making it a strong secondary candidate if the operator prefers slightly higher signal uniformity over the peak CNN confidence of Image #8. However, Image #8 remains the primary choice due to its superior composite scoring.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
Seven offsets are strictly rejected due to severe off-resonance dark banding that compromises diagnostic utility. The anatomical localization of these artifacts is as follows:

*   **Image #1 (-150.0 Hz):** Severe banding (17.1% of myocardium) affecting the **septal** wall.
*   **Image #2 (-125.0 Hz):** Severe banding (27.0% of myocardium); no specific sector identified, but global signal degradation is significant.
*   **Image #3 (-100.0 Hz):** Severe banding (33.8% of myocardium) affecting the **anterior** wall. This is the most severe artifact in the stack.
*   **Image #4 (-75.0 Hz):** Severe banding (22.1