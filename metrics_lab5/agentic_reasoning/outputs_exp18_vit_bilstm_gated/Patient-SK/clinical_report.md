**CLINICAL FREQUENCY SELECTION REPORT**
**Patient ID:** Patient-SK
**Modality:** 3.0T Balanced Steady-State Free Precession (bSSFP) CINE
**Protocol:** Multi-Metric Frequency Scout (13 Slices, -150 Hz to +150 Hz)

---

### 1. CLINICAL INDICATION & OBJECTIVE
The objective of this frequency scouting procedure is to identify the optimal center frequency offset for 3.0T bSSFP CINE imaging to mitigate off-resonance dark banding artifacts. At 3.0T, the increased chemical shift and susceptibility effects necessitate precise frequency alignment to preserve high signal-to-noise ratio (SNR) and blood-myocardium contrast. The primary goal is to select a frequency offset that minimizes intra-myocardial signal dropout (dark banding) while maintaining uniform signal homogeneity across the cardiac chambers and myocardial walls, thereby ensuring diagnostic quality for functional assessment.

### 2. PRIMARY FREQUENCY RECOMMENDATION
**Recommended Offset:** **+0.0 Hz**
**Corresponding Image:** **#7**
**Composite Score:** 0.9955
**Confidence:** High (0.82)

**Justification:**
Image #7 (+0.0 Hz) represents the global multi-metric peak for this patient. It demonstrates a **banded fraction of 0.0%**, indicating a complete absence of off-resonance dark banding intersecting the myocardium. This offset achieves a high signal homogeneity index of **0.7244** (1-CoV), ensuring uniform tissue characterization. Furthermore, the Deep Learning classifier (EfficientNet-B0) assigns the highest confidence score (**0.0834**) to this slice, corroborating the physics-based metrics. The consensus between the neural network and the physical artifact metrics confirms that +0.0 Hz provides the optimal balance of artifact avoidance and signal integrity.

### 3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
The diagnostic operating window for Patient-SK is defined by a **passband plateau** centered around the primary recommendation. This plateau indicates a range of frequencies where image quality remains diagnostically viable for cine function evaluation.

**Acceptable Alternatives:**
1.  **Image #8 (+25.0 Hz):**
    *   **Banded Fraction:** 0.0261 (2.6%)
    *   **Homogeneity:** 0.7115
    *   **Rationale:** This offset remains within the clinical tolerance window (±25 Hz). The negligible banding fraction and high homogeneity support its use as a viable alternative if +0.0 Hz is compromised by external factors.
2.  **Image #11 (+100.0 Hz):**
    *   **Banded Fraction:** 0.0294 (2.9%)
    *   **Homogeneity:** 0.7314
    *   **Rationale:** Although further from the primary peak, this offset exhibits negligible banding and the highest homogeneity score in the stack (0.7314). It is classified as an acceptable alternative within the broader passband consensus.

**Note:** Offsets #5 (-50 Hz), #6 (-25 Hz), #10 (+75 Hz), and #12 (+125 Hz) are classified as "suboptimal" due to reduced signal homogeneity or early peripheral phase roll-off, and are not recommended for primary diagnostic use.

### 4. REJECTED OFFSETS & ANATOMICAL ARTIFACT LOCALIZATION
The following offsets are strictly rejected due to severe off-resonance dark banding that compromises diagnostic utility:

*   **Image #1 (-150.0 Hz):**
    *   **Severity:** Severe (Banded Fraction: 24.1%)
    *   **Affected Sectors:** None explicitly localized (global banding)
    *   **Impact:** Significant signal loss across the myocardium.
*   **Image #2 (-125.0 Hz):**
    *   **Severity:** Severe (Banded Fraction: 27.4%)
    *   **Affected Sectors:** None explicitly localized (global banding)
    *   **Impact:** Highest banding fraction in the stack; unacceptable for diagnostic imaging.
*   **Image #3 (-100.0 Hz):**
    *   **Severity:** Severe (Banded Fraction: 20.1%)
    *   **Affected Sectors:** None explicitly localized (global banding)
    *   **Impact:** Substantial myocardial signal dropout.
*   **Image #4 (-75.0 Hz):**
    *   **