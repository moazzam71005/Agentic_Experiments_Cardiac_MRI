# Clinical Specification: bSSFP CINE Frequency-Offset Selection at 3.0T

## 1. Background Physics

### bSSFP and Banding Artifacts
Balanced steady-state free precession (bSSFP) CINE is the clinical gold standard for
cardiac function assessment. At 1.5T it provides high SNR and excellent blood-myocardium
contrast. At 3.0T, the doubled static magnetic field (B0) causes two key problems:

1. **Greater B0 inhomogeneity** — local field variations across the heart are larger.
2. **Narrower frequency passband** — the bSSFP stop-band (where signal drops to zero)
   is at ±1/(2·TR) from resonance, typically ±125–167 Hz for cardiac TR ≈ 3–4 ms.
   At 3T the absolute inhomogeneity (in Hz) is twice that at 1.5T, so the passband is
   easily violated.

**Banding artifacts** appear as dark (near-zero signal) bands in the image wherever
local off-resonance equals ±1/(2·TR). The bands are spatially fixed in the anatomy but
shift when the transmitter frequency (center frequency / RF offset) changes. Moving the
RF offset by +Δf shifts all bands in the direction that displaces higher-frequency spins.

### The Frequency Scout
The clinical workflow at 3.0T includes a short "frequency scout" acquisition:
- 13 images of the same short-axis (SAX) cardiac slice are acquired.
- Each image uses a different RF transmitter offset, stepped from –150 Hz to +150 Hz
  in 25 Hz increments (patient KP uses 50 Hz steps with 7 images due to wider B0 spread).
- Total acquisition: seconds.
- The clinician reviews all 13 images and selects the offset where banding artifacts
  are pushed out of the myocardium and blood pool — the "best" or "optimal" offset.
- This offset is then used for the full CINE acquisition.

## 2. Anatomy Reference

### Short-Axis View
The frequency scout is acquired in the SAX plane, which shows:
- **Left ventricle (LV)**: circular, centre of image; the primary region of interest.
  - **Blood pool**: high signal in bSSFP (bright).
  - **Myocardium**: intermediate signal ring surrounding the blood pool.
- **Right ventricle (RV)**: crescent-shaped, anterior-right of LV.
- **Pericardial fat / epicardial fat**: thin bright rim outside the myocardium.
- **Lung fields**: low signal (air), outside the cardiac silhouette.
- **Background**: chest wall, ribs, subcutaneous fat.

### AHA 17-Segment Model (simplified for frequency scouting)
Banding location is reported in 4 broad sectors of the LV myocardium (clock positions):
- **Anterior** (12 o'clock)
- **Lateral** (3 o'clock for standard orientation / 9 o'clock)
- **Inferior** (6 o'clock)
- **Septal** (9 o'clock / towards RV)

## 3. What Makes an Offset Optimal

An offset is **optimal** if ALL of the following hold:
1. No dark banding artifact overlaps the **LV myocardium** in any sector.
2. No dark band crosses the **LV blood pool**.
3. The **myocardial signal is uniform** (high homogeneity — low coefficient of variation).
4. **Blood-myocardium contrast** is preserved (the blood pool remains bright).
5. Any residual bands are displaced to: epicardial fat, lung, background, or the RV
   (where some artifact is clinically tolerable).

An offset is **acceptable** (suboptimal but usable) if:
- A mild band crosses the RV (not myocardium) or the far periphery.
- Homogeneity is slightly reduced but the myocardium is still assessable.

An offset is **suboptimal / reject** if:
- A band crosses any sector of the LV myocardium (dark stripe through the heart wall).
- The blood pool shows a dark band (impairs cavity delineation).
- Multiple bands overlap the cardiac ROI.

### Multiple Optimal Offsets
In practice, there is often a **range of 1–4 consecutive offsets** that are all acceptable.
The central offset of this range is the most robust choice. The system should identify:
- The single best offset (highest quality, most central in the acceptable plateau).
- The full set of acceptable alternatives.
- The confidence/trust score (width of the acceptable plateau and evidence agreement).

## 4. Frequency-Scouter Vocabulary

The following terms are used by radiologists and MRI physicists when reviewing frequency scouts:

| Term | Definition |
|------|-----------|
| **off-resonance** | Deviation from the optimal RF center frequency; causes banding |
| **banding artifact** | Dark stripe due to bSSFP signal null at ±1/(2TR) |
| **frequency offset** | RF transmitter shift in Hz applied to this image |
| **center frequency** | 0 Hz reference; nominal Larmor frequency |
| **transmitter offset** | Synonym for frequency offset |
| **passband** | Range of frequencies with high bSSFP signal |
| **stop-band** | Range of frequencies with near-zero bSSFP signal (banding) |
| **B0 inhomogeneity** | Spatial variation in static magnetic field |
| **shimming** | Active/passive correction of B0 inhomogeneity |
| **residual off-resonance** | Remaining B0 inhomogeneity after shimming |
| **band displacement** | How shifting the offset moves the band spatially |
| **myocardial homogeneity** | Uniformity of LV wall signal intensity (high = good) |
| **blood-myocardium CNR** | Contrast-to-noise ratio between LV blood pool and wall |
| **optimal offset** | Frequency offset at which the heart is free of banding |
| **frequency scout** | Series of images at different offsets for offset selection |
| **plateau** | Consecutive range of acceptable offsets |

## 5. Expected Reasoning Template

A clinician reviewing a frequency scout produces reasoning such as:

> "At the reference offset (0 Hz), a moderate dark banding artifact traverses the
> inferolateral wall of the left ventricle, degrading myocardial delineation. Moving
> to +25 Hz displaces this band into the epicardial fat and lung, leaving the
> myocardium uniform and blood-pool contrast well preserved. This offset is therefore
> selected as optimal. The adjacent +50 Hz offset remains acceptable, with the band
> at the lung periphery. Beyond +75 Hz, a new band enters the anterior wall.
> The acceptable plateau spans +25 to +50 Hz; +25 Hz is recommended as the most
> central and robust choice. Confidence: high — neural ranking, quantitative
> homogeneity (CoV 0.94), and visual banding assessment concur."

## 6. Key Clinical References

The knowledge base is grounded in the following literature:

1. **Scheffler & Lehnhardt (2003)** — Principles and applications of balanced SSFP.
   European Radiology 13(11):2409–2418. Covers bSSFP physics and banding theory.

2. **Herzka et al. (2002)** — Myocardial tagging with SSFP. MRM 48:783–789.
   Frequency-offset sensitivity at high field.

3. **Markl et al. (2003)** — Balanced SSFP myocardial perfusion imaging.
   JMRI 17:553–561. Frequency offset selection protocol.

4. **Ferreira et al. (2013)** — Cardiovascular magnetic resonance artefacts.
   JCMR 15:41. Comprehensive banding artifact description and clinical management.

5. **Kramer et al. / SCMR (2013, updated 2020)** — Society for Cardiovascular Magnetic
   Resonance standardized protocols. J Cardiovasc Magn Reson.
   Quality and acquisition standards for cardiac MRI.

6. **Bieri & Scheffler (2013)** — Fundamentals of balanced SSFP MRI.
   J Magn Reson Imaging 38:2–11. Comprehensive review of bSSFP including off-resonance.

## 7. Physics-Derived Quantities Used by the Agent

Given TR (repetition time), the following are computed per patient:
- **Banding period** = 1/TR (Hz) — distance between successive dark bands.
- **Offset step** = 25 Hz (or 50 Hz for KP).
- **Sweep range** = –150 to +150 Hz.
- **Expected band position** for slice i at offset f_i: band at ±(1/(2·TR) – f_i) Hz
  from the local B0 field (depends on shimming residuals, not directly measurable from
  images alone without B0 mapping).

The banding detector treats band position empirically from image intensities.
