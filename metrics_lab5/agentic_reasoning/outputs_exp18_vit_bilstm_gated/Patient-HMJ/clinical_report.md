# Clinical Frequency Selection Report
**Patient ID:** Patient-HMJ
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13 slices, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this automatic frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging. At 3.0T, the increased magnetic field strength amplifies susceptibility-induced off-resonance effects, leading to characteristic dark banding artifacts (stripes) that can obscure myocardial tissue and compromise the assessment of wall motion and ejection fraction. The primary goal is to mitigate these dark banding stripes while simultaneously preserving high blood-myocardium contrast (CNR) and signal homogeneity, ensuring diagnostic image quality for functional cardiac evaluation.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** **+25.0 Hz**
**Corresponding Image:** **#8**

**Justification:**
Image #8 (+25.0 Hz) is designated as the optimal frequency offset based on a composite score of 0.9341 and high confidence (0.85). This selection represents the superior balance between artifact suppression and signal integrity:
*   **Banding Suppression:** It exhibits a low Banded Fraction of Myocardium (BFM) of **0.052**, well below the critical rejection threshold of 0.08, indicating minimal intra-myocardial dark banding.
*   **Signal Homogeneity:** It achieves the highest signal homogeneity (1-CoV) in the stack at **0.6708**, indicating uniform signal distribution across the myocardium.
*   **Neural Confidence:** The CNN score is robust at **0.0861**, reflecting strong classifier consensus on image quality.
*   **Arbitration Basis:** While Image #7 (+0.0 Hz) had a slightly higher CNN score, its BFM of 0.0781 was deemed dangerously close to the 0.08 rejection threshold. Image #8 provides a safer margin against myocardial dark bands while maintaining superior homogeneity.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
**Status:** No acceptable alternatives identified.

The diagnostic operating window for this patient is narrow. According to the Stage-A findings, there are **no acceptable alternative offsets** listed within the ±25 Hz tolerance of the primary recommendation.
*   **Image #7 (+0.0 Hz):** Although adjacent to the primary recommendation, it is classified as "suboptimal" due to reduced signal homogeneity (0.6306) and a BFM (0.0781) that approaches the rejection threshold. It does not meet the criteria for a viable alternative for routine cine function evaluation.
*   **Image #9 (+50.0 Hz):** Similarly classified as "suboptimal" due to reduced signal homogeneity (0.6263) and early peripheral phase roll-off.

Consequently, the passband plateau is effectively singular, centered strictly at **+25.0 Hz**. Operators should not substitute adjacent frequencies without re-evaluation, as both neighboring slices exhibit compromised homogeneity or borderline banding risks.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
Ten out of thirteen scanned offsets were strictly rejected due to severe off-resonance dark banding intersecting the myocardium. The specific rejections and anatomical localizations are as follows:

*   **Negative Offsets (Left Side of Spectrum):**
    *   **#1 (-150.0 Hz):** Rejected. BFM 0.1636. No specific sector identified.
    *   **#2 (-125.0 Hz):** Rejected. BFM 0.2937. No specific sector identified.
    *   **#3 (-100.0 Hz):** Rejected. BFM 0.1338. No specific sector identified.
    *   **#4 (-75.0 Hz):** Rejected. BFM 0.1078. **Affected Sector: Anterior.**
    *   **#5 (-50.0 Hz):** Rejected. BFM 0.1747. **Affected Sector: Anterior.**
    *   **#6 (-25.0 Hz):** Rejected. BFM 0.2454. No specific sector identified.

*   **Positive Offsets (Right Side of Spectrum):**
    *   **#10 (+75.0 Hz):** Rejected. BFM 0.1301. No specific