# Clinical Frequency Selection Report
**Patient ID:** Patient-HMJ
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-Metric Frequency Scout (13 Slices, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this automatic frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging. At high field strengths (3.0T), magnetic susceptibility variations and B0 inhomogeneities cause off-resonance effects that manifest as dark banding artifacts. These artifacts can obscure myocardial tissue, degrade blood-myocardium contrast, and compromise the quantitative assessment of cardiac function. The primary goal is to select a frequency offset that minimizes intra-myocardial dark banding while preserving signal homogeneity and contrast-to-noise ratio (CNR) across the cardiac cycle.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** +25.0 Hz
**Corresponding Image:** #8
**Composite Score:** 0.9709
**Confidence:** High (0.82)

**Justification:**
Image #8 (+25.0 Hz) represents the global multi-metric peak for this patient. It achieves the highest composite score (0.9709) in the scout stack, driven by a superior balance of three critical factors:
1.  **Minimal Intra-Myocardial Banding:** The banded fraction of myocardium is 0.0520 (5.2%), which is significantly lower than the rejected offsets and comparable to the lowest observed value in the stack (Image #9 at 0.0409).
2.  **High Signal Homogeneity:** The signal homogeneity index (1-CoV) is 0.6708, the highest value in the entire 13-slice stack, indicating uniform signal distribution within the myocardium.
3.  **Strong Neural Confidence:** The CNN score is 0.6927, the highest in the stack, indicating strong classifier agreement that this slice is free of significant diagnostic artifacts.

Although Image #9 (+50.0 Hz) exhibits a slightly lower banded fraction (0.0409), its significantly lower CNN score (0.0136) and reduced homogeneity (0.6263) render it suboptimal compared to Image #8.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The system identifies a "passband plateau" status, indicating a region of acceptable image quality surrounding the primary recommendation.

*   **Primary Recommendation:** Image #8 (+25.0 Hz)
*   **Acceptable Alternatives:** None explicitly listed in the `acceptable_alternatives` array.
*   **Tolerance Analysis:** While the consistency summary notes that the CNN pick (#8) and Physics pick (#9) agree within a clinical tolerance window of ±25 Hz, the deterministic decision source has strictly designated Image #8 as the sole optimal choice. Image #9 (+50.0 Hz) is classified as "suboptimal" due to reduced signal homogeneity (0.6263) and early peripheral phase roll-off, despite having the lowest banded fraction. Therefore, no alternative offsets are formally recommended for clinical use; the operator should adhere strictly to the +25.0 Hz offset to ensure maximum homogeneity and neural confidence.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
Ten out of thirteen scout images were rejected due to severe off-resonance dark banding intersecting the myocardium. The specific rejections and anatomical impacts are detailed below:

**Negative Frequency Offsets (Left Side of Spectrum):**
*   **Image #1 (-150.0 Hz):** Rejected. Banded fraction: 16.4%. No specific sector identified, but severe global banding.
*   **Image #2 (-125.0 Hz):** Rejected. Banded fraction: 29.4%. No specific sector identified.
*   **Image #3 (-100.0 Hz):** Rejected. Banded fraction: 13.4%. No specific sector identified.
*   **Image #4 (-75.0 Hz):** Rejected. Banded fraction: 10.8%. **Affected Sector: Anterior.**
*   **Image #5 (-50.0 Hz):** Rejected. Banded fraction: 17.5%. **Affected Sector: Anterior.**
*   **Image #6 (-25.0 Hz):** Rejected. Banded fraction: 24.5%.