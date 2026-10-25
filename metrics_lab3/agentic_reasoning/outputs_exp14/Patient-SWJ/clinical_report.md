1. CLINICAL INDICATION & OBJECTIVE
Automated frequency-scout optimization for Patient Patient-SWJ at 3.0T cardiac bSSFP CINE MRI. The diagnostic goal is to select the radiofrequency offset that eliminates off-resonance banding artifacts within the left ventricular myocardium while maintaining high myocardium-to-blood contrast across the cardiac cycle.

2. PRIMARY FREQUENCY RECOMMENDATION
RECOMMENDED OFFSET: +25.0 Hz (Image #8)
- Composite Score: 0.9880
- Clinical Confidence: MEDIUM (Rating: 0.65)
- Rationale: Image #8 occupies the global optimum of the myocardial passband. It demonstrates near-zero intra-myocardial banded fraction, optimal signal homogeneity, and strong agreement across multi-agent physics and contrast specialists.

3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
A stable passband plateau exists around the primary recommendation. The following adjacent offsets lie within the acceptable clinical tolerance window (±25 Hz to ±50 Hz) and retain diagnostic image quality:
  * #6 (-25.0 Hz, Δ=-50 Hz): banded fraction=2.2%, homogeneity=0.832
  * #7 (+0.0 Hz, Δ=-25 Hz): banded fraction=0.6%, homogeneity=0.849
These alternative offsets provide acceptable blood-myocardium boundary delineation if minor shimming adjustments are necessitated during the scan session.

4. EXCLUDED OFFSETS & ARTIFACT LOCALIZATION
The following frequency offsets were vetoed due to significant off-resonance banding (BFM > 5%):
  * #1 (-150.0 Hz): banded fraction=9.5%, affected AHA wall sectors: none
  * #2 (-125.0 Hz): banded fraction=9.5%, affected AHA wall sectors: none
  * #3 (-100.0 Hz): banded fraction=10.6%, affected AHA wall sectors: anterior
  * #4 (-75.0 Hz): banded fraction=16.1%, affected AHA wall sectors: septal
  * #10 (+75.0 Hz): banded fraction=21.5%, affected AHA wall sectors: septal
  * #11 (+100.0 Hz): banded fraction=31.8%, affected AHA wall sectors: anterior
Severe signal nulls in these frames obscure cardiac wall motion and preclude accurate ventricular analysis.

5. METHODOLOGICAL RATIONALE & METRIC EVIDENCE
- Multi-Agent Agreement: passband_plateau
- Confidence Basis: Quantitative consensus
- Banding Burden at Selected Offset: 0.00%
- Homogeneity Index: 0.000

6. RISK OF SUBOPTIMAL OFFSET SELECTION
Operating at an incorrect frequency offset in 3.0T bSSFP introduces dark band artifacts traversing the myocardium. This mimics focal wall thinning, impairs regional ejection fraction assessment, and risks false-positive diagnosis of myocardial infarction or scar.

7. FINAL SCANNER CONFIGURATION INSTRUCTION
Apply center frequency offset of +25.0 Hz (Image #8) on the scanner console. Proceed with diagnostic SAX CINE acquisition.