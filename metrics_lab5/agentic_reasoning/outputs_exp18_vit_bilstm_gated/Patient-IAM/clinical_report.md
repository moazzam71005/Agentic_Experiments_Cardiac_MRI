**CLINICAL FREQUENCY SELECTION REPORT**
**Patient ID:** Patient-IAM
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13 slices, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging to mitigate off-resonance dark banding artifacts. At 3.0T, the increased susceptibility to magnetic field inhomogeneities necessitates precise frequency tuning to prevent signal voids from intersecting the myocardium. The primary goal is to preserve high blood-myocardium contrast and signal homogeneity while ensuring that no dark banding stripes obscure the endocardial or epicardial borders, thereby enabling accurate assessment of ventricular function and wall motion.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** **+25.0 Hz**
**Corresponding Image:** **#8**

**Justification:**
Image #8 (+25.0 Hz) is designated as the optimal frequency offset based on a composite score of **0.9678**, which represents the global peak across the scout stack. This offset demonstrates the best balance of critical imaging metrics:
*   **Intra-myocardial Banding:** The banded fraction of the myocardium is minimal at **0.0071** (0.71%), effectively eliminating diagnostic interference from dark stripes.
*   **Signal Homogeneity:** The myocardial signal homogeneity (1-CoV) is **0.7248**, indicating uniform signal distribution within the myocardial wall.
*   **Neural Confidence:** The CNN classifier assigns the highest confidence score (**0.0938**) to this slice, corroborating the physics-based metrics.
*   **Temporal Stability:** The temporal standard deviation sweep is low (**0.0106**), suggesting stable signal acquisition over the cardiac cycle.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The scout stack exhibits a clear **passband plateau** centered around the isocenter, spanning from **-25.0 Hz to +50.0 Hz**. Within this window, all slices maintain negligible intra-myocardial banding and acceptable signal homogeneity, making them viable for diagnostic CINE acquisition.

**Acceptable Alternative Offsets:**
1.  **Image #7 (+0.0 Hz):**
    *   *Metrics:* Banded fraction **0.0%**, Homogeneity **0.7933**.
    *   *Viability:* This slice offers the highest signal homogeneity in the stack and zero banding. It is a strong alternative if the operator prefers the exact isocenter frequency.
2.  **Image #9 (+50.0 Hz):**
    *   *Metrics:* Banded fraction **0.0018** (0.18%), Homogeneity **0.7446**.
    *   *Viability:* This slice maintains near-zero banding and high homogeneity. It is diagnostically viable and offers a slightly higher CNR (1.65) compared to the primary recommendation.
3.  **Image #6 (-25.0 Hz):**
    *   *Metrics:* Banded fraction **0.0266** (2.66%), Homogeneity **0.6707**.
    *   *Viability:* While the banding fraction is slightly higher than the primary recommendation, it remains below the threshold for significant diagnostic interference. It is acceptable if the +25.0 Hz offset is technically unattainable due to scanner constraints.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
Offsets outside the passband plateau are strictly rejected due to severe off-resonance dark banding that compromises diagnostic utility.

**Negative Frequency Rejections:**
*   **Image #5 (-50.0 Hz):** Rejected as **suboptimal**. Early peripheral phase roll-off is observed, with banding affecting the **septal** sector. Banded fraction is **0.0657** (6.57%).
*   **Images #1–#4 (-150.0 Hz to -75.0 Hz):** Rejected due to severe banding. Banded fractions range from **9.77% to 29.49%**. No specific sectors were flagged in the evidence matrix for these slices, but the global signal loss is significant.

**Positive Frequency Rejections:**
*   **Image #10 (+75.0 Hz):** Rejected. Severe dark band