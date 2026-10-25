**CLINICAL FREQUENCY SELECTION REPORT**
**Patient ID:** Patient-OAF
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13 slices, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging to mitigate off-resonance artifacts. At 3.0T, magnetic susceptibility variations and B0 inhomogeneities are amplified, leading to characteristic dark banding stripes that can obscure myocardial tissue and compromise the assessment of wall motion and ventricular function. The primary goal is to select a frequency offset that minimizes intra-myocardial signal dropout (dark banding) while preserving high signal homogeneity and adequate blood-myocardium contrast, thereby ensuring diagnostic image quality for clinical evaluation.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** **+25.0 Hz**
**Corresponding Image:** **#8**

**Justification:**
Image #8 represents the global multi-metric peak for this patient, achieving the highest composite score of **0.9881**. This offset provides the optimal balance of diagnostic metrics:
*   **Intra-myocardial Banding:** The banded fraction of the myocardium is negligible at **0.0056** (0.56%), effectively eliminating dark banding artifacts within the heart muscle.
*   **Signal Homogeneity:** The signal homogeneity (1-CoV) reaches its maximum value of **0.8579**, indicating uniform signal intensity across the myocardium, which is critical for accurate tissue characterization.
*   **Temporal Stability:** The temporal standard deviation sweep is at its minimum (**0.0027**), suggesting stable signal acquisition over the cardiac cycle.
*   **Neural Confidence:** The CNN score is high (**0.0894**), confirming that the image quality meets the criteria for diagnostic utility without significant artifact interference.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The analysis identifies a clear **passband plateau** centered around the recommended offset. This plateau indicates a range of frequencies where image quality remains diagnostically viable, offering flexibility in case of minor patient motion or shimming drift.

**Acceptable Alternative Offsets:**
1.  **Image #6 (-25.0 Hz):**
    *   **Banded Fraction:** 0.0% (Zero intra-myocardial banding).
    *   **Homogeneity:** 0.8314.
    *   **Viability:** This offset is highly viable, offering zero banding and high homogeneity. It is the primary alternative if the +25 Hz offset is compromised by external factors.
2.  **Image #9 (+50.0 Hz):**
    *   **Banded Fraction:** 0.0359 (3.59%).
    *   **Homogeneity:** 0.7738.
    *   **Viability:** This offset remains within the diagnostic passband. While homogeneity is slightly lower than the primary recommendation, the banding fraction is still negligible, and the composite score (0.9008) supports its use for cine function evaluation.

The consensus between the CNN (favoring #7 at 0 Hz) and Physics metrics (favoring #6 at -25 Hz) aligns within the ±25 Hz clinical tolerance window, confirming the robustness of the passband from -25 Hz to +50 Hz.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
Four offsets are strictly rejected due to severe off-resonance dark banding that intersects the myocardium, rendering them non-diagnostic for wall motion assessment.

*   **Image #1 (-150.0 Hz):**
    *   **Severity:** Severe. Banded fraction of **18.5%**.
    *   **Location:** No specific sector identified ("none"), indicating diffuse or non-sector-specific severe artifact.
*   **Image #2 (-125.0 Hz):**
    *   **Severity:** Severe. Banded fraction of **13.56%**.
    *   **Location:** No specific sector identified ("none").
*   **Image #4 (-75.0 Hz):**
    *   **Severity:** Severe. Banded fraction of **17.38%**.
    *   **Location:** **Anterior** wall. This offset causes significant signal loss specifically in the anterior myocardial segment, which would critically impair assessment of anterior wall motion and