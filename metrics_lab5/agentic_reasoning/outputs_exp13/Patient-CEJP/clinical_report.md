# Clinical Frequency Selection Report: Patient-CEJP

## 1. CLINICAL INDICATION & OBJECTIVE
The objective of this automatic frequency scouting procedure is to identify the optimal center frequency offset for 3.0T balanced Steady-State Free Precession (bSSFP) CINE imaging. The primary goal is to mitigate off-resonance dark banding artifacts, which arise from local magnetic field inhomogeneities, while simultaneously preserving high signal-to-noise ratio (SNR) and blood-myocardium contrast. By selecting a frequency offset that minimizes intra-myocardial signal loss, the system ensures diagnostic image quality for the evaluation of cardiac wall motion and function.

## 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** -75.0 Hz
**Corresponding Image:** #4
**Composite Score:** 0.9247
**Confidence:** High (0.85)

Image #4 (-75.0 Hz) is designated as the optimal frequency offset. This selection is justified by its superior performance across all critical quality metrics:
*   **Minimal Banding:** It exhibits the lowest banded fraction in the myocardium (BFM) among viable candidates at 0.0385 (3.85%), well below the rejection threshold.
*   **Superior Homogeneity:** It possesses the highest signal homogeneity (1-CoV) score of 0.7456 across the entire stack, indicating uniform signal intensity within the myocardial tissue.
*   **Neural Consensus:** The CNN score of 0.0133 indicates strong classifier confidence in the absence of significant artifacts.
This offset represents the optimal balance between avoiding dark banding stripes and maintaining robust tissue contrast.

## 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The diagnostic operating window is centered around the primary recommendation of -75.0 Hz. The passband plateau includes offsets where the banded fraction remains low and homogeneity is preserved, allowing for reliable cine function evaluation.

**Acceptable Alternative:**
*   **Image #6 (-25.0 Hz):**
    *   **BFM:** 0.0237 (2.37%)
    *   **Homogeneity:** 0.7038
    *   **Justification:** This offset falls within the acceptable tolerance range. It demonstrates negligible banding and high homogeneity, making it a viable alternative if the primary offset is compromised by other factors (e.g., motion). The delta from the primary recommendation is 50.0 Hz, but it remains within the functional passband for diagnostic purposes.

Offsets outside this narrow window (e.g., -100.0 Hz, -50.0 Hz) show a significant degradation in homogeneity or an increase in banding, reducing their diagnostic reliability.

## 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
Eleven out of thirteen offsets were rejected due to severe off-resonance dark banding intersecting the myocardium. The specific rejections and anatomical impacts are as follows:

*   **Image #1 (-150.0 Hz):** Rejected. BFM 0.123 (12.3%). **Affected Sector: Anterior.** Severe dark banding observed in the anterior wall.
*   **Image #2 (-125.0 Hz):** Rejected. BFM 0.0904 (9.04%). **Affected Sector: None explicitly localized, but borderline banding present.** Although no specific sector was flagged in the band_sectors field, the BFM is borderline high, and homogeneity is lower (0.663) than the primary recommendation.
*   **Image #3 (-100.0 Hz):** Rejected. BFM 0.1156 (11.6%). **Affected Sector: Anterior.** Significant signal loss in the anterior wall.
*   **Image #5 (-50.0 Hz):** Rejected. BFM 0.0993 (9.93%). **Affected Sector: None explicitly localized.** Homogeneity is notably lower (0.5765) compared to the passband.
*   **Image #7 (+0.0 Hz):** Rejected. BFM 0.1793 (17.9%). **Affected Sector: Anterior.** Severe banding in the anterior wall.
*   **Image #8 (+25.0 Hz):** Rejected. BFM 0.2326 (23.3%). **Affected Sector: None explicitly localized.** High banding fraction despite high CNR.
*   **Image #9 (+50.0 Hz):** Rejected. BFM 0.275