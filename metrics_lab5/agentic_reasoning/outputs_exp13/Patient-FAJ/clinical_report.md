# Clinical Frequency Selection Report
**Patient ID:** Patient-FAJ
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13 slices, -150 Hz to +150 Hz)

## 1. CLINICAL INDICATION & OBJECTIVE
The objective of this automatic frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging. At 3.0T, the increased susceptibility to off-resonance effects necessitates precise frequency tuning to mitigate dark banding artifacts (stripes) that arise from B0 field inhomogeneities. The primary goal is to select a frequency offset that minimizes intra-myocardial signal loss while preserving high blood-myocardium contrast and signal homogeneity, thereby ensuring diagnostic quality for cardiac function evaluation.

## 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** -75.0 Hz
**Corresponding Image:** #4

**Justification:**
Image #4 (-75.0 Hz) is designated as the optimal frequency offset based on a composite score of 0.7561, representing the global peak across all evaluated metrics.
*   **Banding Avoidance:** This offset yields the lowest intra-myocardial banded fraction (0.0686 or 6.86%) among all tested slices, effectively avoiding the severe dark banding observed at other offsets.
*   **Signal Homogeneity:** It achieves the highest signal homogeneity (1-CoV) score of 0.6993, indicating uniform signal distribution within the myocardium.
*   **Temporal Stability:** The temporal standard deviation sweep is minimal (0.0054), suggesting stable signal acquisition over the cardiac cycle.
*   **Contrast:** The Contrast-to-Noise Ratio (CNR) is robust at 6.03, ensuring adequate differentiation between blood pool and myocardial tissue.

## 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
**Status:** No Acceptable Alternatives Identified.

The analysis reveals a narrow passband plateau centered exclusively at -75.0 Hz. No other offsets within the ±25 Hz tolerance (i.e., -100.0 Hz, -50.0 Hz) meet the criteria for diagnostic viability due to significant artifact presence or suboptimal metrics:
*   **-100.0 Hz (Image #3):** Rejected due to a banded fraction of 0.1248 (12.48%), which exceeds the threshold for acceptable artifact load.
*   **-50.0 Hz (Image #5):** Rejected due to septal banding (banded fraction 0.0842) and significantly higher temporal instability (temporal std 0.0485) compared to the primary recommendation.

Consequently, the operator must utilize the -75.0 Hz offset to ensure diagnostic quality. There is no viable alternative window for cine function evaluation in this scout stack.

## 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
All offsets other than -75.0 Hz are strictly rejected due to off-resonance dark banding. The anatomical localization of artifacts is detailed below:

*   **Severe Global/Unlocalized Banding (No specific sector identified):**
    *   **#1 (-150.0 Hz):** Banded fraction 26.36%.
    *   **#2 (-125.0 Hz):** Banded fraction 24.34%.
    *   **#3 (-100.0 Hz):** Banded fraction 12.48%.
    *   **#12 (+125.0 Hz):** Banded fraction 32.45%.
    *   **#13 (+150.0 Hz):** Banded fraction 33.70%.
    *   *Note:* These slices exhibit severe signal loss without specific sectoral localization in the evidence matrix, indicating widespread B0 inhomogeneity effects.

*   **Septal Wall Artifacts:**
    *   **#5 (-50.0 Hz):** Banded fraction 8.42%.
    *   **#9 (+50.0 Hz):** Banded fraction 10.30%.
    *   **#10 (+75.0 Hz):** Banded fraction 12.64%.
    *   **#11 (+100.0 Hz):** Banded fraction 19.81%.
    *   *Clinical Impact:* Banding in the septal wall may obscure assessment of septal motion and wall thickness.

*   **Anterior and Sept