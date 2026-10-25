**CLINICAL FREQUENCY SELECTION REPORT**
**Patient ID:** Patient-BEV
**Modality:** 3.0T Cardiac MRI – Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-metric Frequency Scout (13-slice stack, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging. At high field strengths (3.0T), bSSFP sequences are highly sensitive to magnetic field inhomogeneities, which manifest as dark banding artifacts (off-resonance stripes). These artifacts can obscure myocardial tissue, degrade the blood-myocardium contrast, and compromise the assessment of wall motion and ejection fraction. This report aims to select a frequency offset that minimizes intra-myocardial banding while preserving signal homogeneity and contrast-to-noise ratio (CNR) for diagnostic-grade cine evaluation.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** **+0.0 Hz**
**Corresponding Image:** **#7**
**Composite Score:** 0.9972
**Confidence:** High (0.82)

**Justification:**
Image #7 (+0.0 Hz) represents the global multi-metric peak for this patient. It demonstrates the lowest intra-myocardial banded fraction (0.37%) across the entire scout stack, indicating near-complete avoidance of off-resonance dark banding within the myocardium. Concurrently, it maintains high signal homogeneity (1-CoV = 0.830) and achieves the highest CNN confidence score (0.0999), reflecting strong neural consensus for artifact-free image quality. The alignment of physics-based metrics (minimal banding) and deep learning classification confirms this offset as the optimal balance for preserving diagnostic contrast without signal loss.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The scout data reveals a clear **passband plateau** centered at 0.0 Hz, extending from -25.0 Hz to +25.0 Hz. Within this window, the myocardial banded fraction remains negligible (<1.0%), and signal homogeneity exceeds 0.82.

**Acceptable Alternatives:**
*   **Image #6 (-25.0 Hz):**
    *   *Banded Fraction:* 0.61%
    *   *Homogeneity:* 0.8322
    *   *Viability:* This offset is diagnostically viable. It exhibits slightly higher homogeneity than the primary recommendation but with a marginally higher banded fraction. It serves as a robust backup if minor temporal instability is observed at 0.0 Hz.
*   **Image #8 (+25.0 Hz):**
    *   *Banded Fraction:* 0.98%
    *   *Homogeneity:* 0.8212
    *   *Viability:* This offset remains within the clinical tolerance window. The banded fraction is still negligible (<1.0%), and homogeneity is acceptable for cine function evaluation.

Offsets outside this ±25 Hz window (e.g., -50 Hz, +50 Hz) show a progressive decline in homogeneity and an increase in banded fraction, marking the boundaries of the usable passband.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
Six offsets were strictly rejected due to severe off-resonance dark banding that compromises diagnostic utility.

*   **Image #10 (+75.0 Hz):**
    *   *Severity:* Severe (Banded Fraction = 18.06%)
    *   *Affected Sector:* **Anterior**
    *   *Note:* This is the only offset in the stack with explicitly localized sectoral involvement. The anterior wall is significantly obscured by dark banding, rendering assessment of anterior wall motion unreliable.
*   **Image #11 (+100.0 Hz):**
    *   *Severity:* Critical (Banded Fraction = 31.70%)
    *   *Affected Sector:* None specified (Global/Diffuse)
    *   *Note:* Represents the worst-case artifact severity in the stack, with over 30% of the myocardium affected by banding.
*   **Image #12 (+125.0 Hz):**
    *   *Severity:* Severe (Banded Fraction = 19.66%)
    *   *Affected Sector:* None specified
*   **Image #13 (+150.0 Hz):**
    *   *Severity:* Se