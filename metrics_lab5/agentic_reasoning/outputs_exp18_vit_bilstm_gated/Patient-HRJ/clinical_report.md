# Clinical Frequency Selection Report
**Patient ID:** Patient-HRJ
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13 slices, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this automatic frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging. At 3.0T, the increased magnetic field strength amplifies susceptibility-induced off-resonance effects, leading to characteristic dark banding artifacts that can obscure myocardial tissue and compromise the assessment of wall motion and ejection fraction. The primary goal is to mitigate these dark banding stripes while preserving high blood-myocardium contrast and signal homogeneity, ensuring diagnostic image quality for cardiac functional evaluation.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** 0.0 Hz
**Corresponding Image:** #7
**Composite Score:** 0.9064
**Confidence:** High (0.85)

**Justification:**
Image #7 (0.0 Hz) is designated as the optimal frequency offset based on LLM arbitration. This slice demonstrates the only instance of **zero myocardial banding** (Banded Fraction Myo = 0.000) across the entire scout stack. Furthermore, it exhibits the highest signal homogeneity (1-CoV = 0.8157), which is critical for accurate tissue characterization in bSSFP sequences. Although Image #9 (+50 Hz) possesses a slightly higher CNN score (0.0904 vs. 0.0812), it exhibits borderline septal banding (BFM = 0.0734). The physics-grounded constraint of eliminating intra-myocardial artifacts takes precedence over marginal gains in neural confidence, making 0.0 Hz the superior choice for diagnostic integrity.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The diagnostic operating window is centered around the primary recommendation at 0.0 Hz. The passband plateau is defined by offsets where myocardial banding is negligible and signal homogeneity remains high.

*   **Primary Optimal:** #7 (0.0 Hz) – BFM 0.000, Homogeneity 0.8157.
*   **Acceptable Alternative:** #6 (-25.0 Hz) – BFM 0.0379, Homogeneity 0.7758.

**Viability Analysis:**
Image #6 (-25.0 Hz) is classified as an acceptable alternative. It maintains a very low banded fraction (3.79%) and high homogeneity (0.7758), well within the diagnostic threshold for cine function evaluation. The transition from #6 to #7 shows a significant improvement in banding elimination (from 3.79% to 0.00%) with a modest increase in homogeneity. Offsets beyond this window (e.g., #8 at +25.0 Hz) show a decline in homogeneity (0.7485) and the onset of minor banding (4.53%), marking the edge of the optimal passband.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
The following offsets are strictly rejected due to significant off-resonance dark banding intersecting the myocardium, which compromises diagnostic utility.

*   **Negative Offsets (Left Side of Spectrum):**
    *   **#3 (-100.0 Hz):** Severe banding (BFM 35.01%). No specific sector identified, but global signal degradation is severe.
    *   **#2 (-125.0 Hz):** Severe banding (BFM 30.48%). **Affected Sector: Anterior.**
    *   **#4 (-75.0 Hz):** Significant banding (BFM 15.79%). No specific sector identified.
    *   **#5 (-50.0 Hz):** Significant banding (BFM 14.20%). No specific sector identified.
    *   **#1 (-150.0 Hz):** Moderate banding (BFM 9.79%). No specific sector identified.

*   **Positive Offsets (Right Side of Spectrum):**
    *   **#13 (+150.0 Hz):** Severe banding (BFM 23.62%). **Affected Sectors: Anterior, Septal.**
    *   **#11 (+100.0 Hz):** Significant banding (BFM 11.51%). **Affected S