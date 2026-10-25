1. CLINICAL INDICATION & OBJECTIVE
Automated frequency-scout optimization for Patient Patient-MJ at 3.0T cardiac bSSFP CINE MRI. The diagnostic goal is to select the radiofrequency offset that eliminates off-resonance banding artifacts within the left ventricular myocardium while maintaining high myocardium-to-blood contrast across the cardiac cycle.

2. PRIMARY FREQUENCY RECOMMENDATION
RECOMMENDED OFFSET: -50.0 Hz (Image #5)
- Composite Score: 0.9612
- Clinical Confidence: HIGH (Rating: 0.90)
- Rationale: Image #5 occupies the global optimum of the myocardial passband. Visual inspection confirmed complete absence of intra-myocardial dark bands and sharp blood-pool contrast.

3. DIAGNOSTIC OPERATING WINDOW (PASSBAND PLATEAU)
A stable passband plateau exists around the primary recommendation. The following adjacent offsets lie within the acceptable clinical tolerance window (±25 Hz to ±50 Hz) and retain diagnostic image quality:
  * #4 (-75.0 Hz, Δ=-25 Hz): banded fraction=0.9%, homogeneity=0.650
These alternative offsets provide acceptable blood-myocardium boundary delineation if minor shimming adjustments are necessitated during the scan session.

4. EXCLUDED OFFSETS & ARTIFACT LOCALIZATION
The following frequency offsets were vetoed due to significant off-resonance banding (BFM > 5%):
  * #1 (-150.0 Hz): banded fraction=18.3%, affected AHA wall sectors: none
  * #2 (-125.0 Hz): banded fraction=12.6%, affected AHA wall sectors: none
  * #6 (-25.0 Hz): banded fraction=9.1%, affected AHA wall sectors: none
  * #7 (+0.0 Hz): banded fraction=8.3%, affected AHA wall sectors: none
  * #8 (+25.0 Hz): banded fraction=14.6%, affected AHA wall sectors: none
  * #9 (+50.0 Hz): banded fraction=12.0%, affected AHA wall sectors: none
Severe signal nulls in these frames obscure cardiac wall motion and preclude accurate ventricular analysis.

5. METHODOLOGICAL RATIONALE & METRIC EVIDENCE
- Visual Decision Status: passband_plateau
- Confidence Basis: Multimodal visual consensus
- Banding Burden at Selected Offset: 0.00%
- Homogeneity Index: 0.000

6. RISK OF SUBOPTIMAL OFFSET SELECTION
Operating at an incorrect frequency offset in 3.0T bSSFP introduces dark band artifacts traversing the myocardium. This mimics focal wall thinning, impairs regional ejection fraction assessment, and risks false-positive diagnosis of myocardial infarction or scar.

7. FINAL SCANNER CONFIGURATION INSTRUCTION
Apply center frequency offset of -50.0 Hz (Image #5) on the scanner console. Proceed with diagnostic SAX CINE acquisition.