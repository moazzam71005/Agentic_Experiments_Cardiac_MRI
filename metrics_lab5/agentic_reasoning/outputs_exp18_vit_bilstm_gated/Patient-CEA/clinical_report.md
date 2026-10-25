# Clinical Frequency Selection Report
**Patient ID:** Patient-CEA
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13-slice stack, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this automatic frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging. At 3.0T, the increased susceptibility to magnetic field inhomogeneities significantly elevates the risk of off-resonance artifacts, specifically dark banding stripes, which can obscure myocardial borders and compromise the assessment of wall motion and ejection fraction. The primary goal is to mitigate these dark banding artifacts while preserving high signal-to-noise ratio (SNR) and robust blood-myocardium contrast, ensuring diagnostic image quality for functional cardiac evaluation.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** **+0.0 Hz**
**Corresponding Image:** **#7**

**Justification:**
Image #7 (+0.0 Hz) is designated as the optimal frequency setting based on a composite score of **0.9915**, reflecting a global multi-metric peak. This offset demonstrates the best balance of critical imaging parameters:
*   **Intra-myocardial Banding:** Minimal banding fraction of **1.57%**, indicating negligible interference with myocardial signal.
*   **Signal Homogeneity:** High homogeneity index of **0.7667** (1-CoV), ensuring uniform signal distribution across the myocardium.
*   **Neural Confidence:** The highest CNN score in the stack (**0.0919**), indicating strong classifier confidence in the absence of significant artifacts.
*   **Consensus:** Both the Deep Learning classifier and physics-based artifact metrics independently converge on this slice, providing high confidence in the selection.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The frequency scout reveals a stable **passband plateau** centered at 0.0 Hz, characterized by consistent diagnostic quality across a ±25 Hz range. This plateau indicates that minor deviations from the exact center frequency will not result in clinically significant degradation of image quality.

**Acceptable Alternatives:**
*   **Image #6 (-25.0 Hz):** Composite score 0.9358. Banded fraction 2.01%, Homogeneity 0.7685.
*   **Image #8 (+25.0 Hz):** Composite score 0.9272. Banded fraction 1.57%, Homogeneity 0.7548.

**Viability Explanation:**
Both alternatives maintain a banded fraction below 2.1% and homogeneity indices above 0.75. These metrics confirm that the myocardial signal remains sufficiently uniform and free from severe dark banding to allow for accurate cine function evaluation. The slight variation in homogeneity between #6 and #8 is within clinical tolerance, making either offset a viable backup if the primary +0.0 Hz setting encounters minor drift or if operator preference dictates a slight shift to optimize temporal stability (noting Image #6 has a higher temporal standard deviation of 0.0501 vs. 0.0346 for #8).

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
Seven offsets were strictly rejected due to severe off-resonance dark banding that compromises diagnostic utility.

**Severe Rejections (Banded Fraction > 10%):**
*   **Image #12 (+125.0 Hz):** Highest artifact severity with a banded fraction of **38.48%**. Homogeneity drops to 0.2537.
*   **Image #2 (-125.0 Hz):** Banded fraction of **32.44%**. **Anatomical Localization:** Dark banding explicitly intersects the **anterior** myocardial wall.
*   **Image #11 (+100.0 Hz):** Banded fraction of **26.62%**.
*   **Image #13 (+150.0 Hz):** Banded fraction of **26.40%**.
*   **Image #1 (-150.0 Hz):** Banded fraction of **24.61%**.
*   **Image #3 (-100.0 Hz):** Banded fraction of **23.04%**.
*   **Image #10 (+75.0 Hz):** Banded fraction of **11.86%**.

**Suboptimal Offsets (Early Roll-off):**