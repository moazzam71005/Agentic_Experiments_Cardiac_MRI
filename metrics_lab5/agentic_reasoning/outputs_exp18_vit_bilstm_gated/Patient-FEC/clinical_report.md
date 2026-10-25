# Clinical Frequency Selection Report

**Patient ID:** Patient-FEC
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13-slice stack, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE

The objective of this automatic frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging to mitigate off-resonance dark banding artifacts while preserving high blood-myocardium contrast. At 3.0T, the increased chemical shift and susceptibility effects necessitate precise frequency alignment to prevent signal voids within the myocardium, which would otherwise compromise the assessment of wall motion and global cardiac function. This report evaluates a 13-slice frequency scout stack to determine the diagnostic passband where intra-myocardial signal homogeneity is maximized and artifact burden is minimized.

### 2. PRIMARY FREQUENCY RECOMMENDATION

**Recommended Offset:** +50.0 Hz
**Corresponding Image:** #9
**Composite Score:** 0.9677
**Confidence:** High (0.82)

The offset of **+50.0 Hz** (Image #9) is designated as the optimal center frequency for this patient. This selection represents the global multi-metric peak across the scout stack. Specifically, Image #9 demonstrates the lowest intra-myocardial banded fraction (0.56%) and the highest signal homogeneity (1-CoV: 0.845) among all tested slices. While the CNN score (0.0861) is slightly lower than the peak at +25 Hz, the physics-based metrics strongly favor +50 Hz due to the near-elimination of dark banding artifacts. The high composite score (0.9677) confirms that this offset provides the best balance between artifact suppression and signal quality, ensuring a clear delineation of the myocardial wall for functional evaluation.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)

The diagnostic operating window for this patient is defined as a **passband plateau** spanning from **-25.0 Hz to +100.0 Hz**. Within this range, the myocardial signal remains diagnostically viable, with banded fractions generally below 4.0% and homogeneity indices above 0.70.

The following offsets are identified as acceptable alternatives should the primary recommendation be technically constrained:

*   **#8 (+25.0 Hz):** Banded fraction 2.07%, Homogeneity 0.7208. This slice offers the highest CNN confidence (0.0877) and is a strong secondary option.
*   **#7 (0.0 Hz):** Banded fraction 3.58%, Homogeneity 0.7370. Provides a balanced profile with moderate artifact levels.
*   **#6 (-25.0 Hz):** Banded fraction 3.96%, Homogeneity 0.7540. Acceptable, though homogeneity is slightly lower than the primary recommendation.
*   **#10 (+75.0 Hz):** Banded fraction 3.58%, Homogeneity 0.7108. Viable, but shows a slight decline in homogeneity compared to the peak.
*   **#11 (+100.0 Hz):** Banded fraction 3.58%, Homogeneity 0.7011. The upper limit of the passband; acceptable but with reduced signal uniformity.

All listed alternatives maintain negligible banding (<4.0%) and sufficient homogeneity for cine function evaluation, allowing for minor adjustments if necessary.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION

Offsets outside the passband plateau are strictly rejected due to severe off-resonance dark banding that intersects the myocardium, compromising diagnostic utility.

**Negative Frequency Rejections:**
*   **#1 (-150.0 Hz):** Banded fraction 12.43%. Severe dark banding intersecting the myocardium. No specific sector localization provided in evidence, but global impact is significant.
*   **#2 (-125.0 Hz):** Banded fraction 24.48%. Severe artifact burden.
*   **#3 (-100.0 Hz):** Banded fraction 22.41%. Severe artifact burden.
*   **#4 (-75.0 Hz):** Banded fraction 18.08%. Severe artifact burden.
*   **#5 (-50.0 Hz):** Banded fraction 13.56%. Severe artifact burden.

**Positive Frequency Rejections:**