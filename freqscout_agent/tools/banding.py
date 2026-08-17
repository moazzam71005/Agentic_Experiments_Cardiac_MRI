"""
T2 Banding Detector
-------------------
Detects dark off-resonance banding artifacts in bSSFP frequency-scout images.
Uses deterministic computer-vision (no learning required):
  1. Identify "dark band" candidates: intensity valleys below a threshold relative
     to the local myocardial median.
  2. Locate connected dark regions in the whole image.
  3. Intersect with the myocardial / LV-pool masks from the segmentation tool.
  4. Report band presence, location (AHA sector), severity, and myocardial overlap.

All outputs are provenance-tagged (patient_id, slice_idx, offset_hz, tool).
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .segment import SegmentationResult

logger = logging.getLogger(__name__)

TOOL_NAME = "banding"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BandingResult:
    """Banding evidence for a single offset image."""
    patient_id: str
    slice_idx: int
    offset_hz: float

    band_present: bool              # True if any band detected anywhere
    bands_in_myocardium: bool       # True if a band overlaps the LV myocardium
    bands_in_lv_pool: bool          # True if a band overlaps the LV blood pool
    severity: float                 # 0–1 global severity score
    myo_overlap_fraction: float     # fraction of myocardium pixels that are dark
    band_sectors: List[str]         # AHA sectors affected ("anterior","lateral",...)
    band_mask: np.ndarray           # H×W bool of detected band pixels
    band_pixel_fraction: float      # fraction of whole image that is banding

    provenance: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------

class BandingDetector:
    """
    Parameters
    ----------
    dark_fraction : float
        Pixels below (image_median * dark_fraction) are considered "dark band" candidates.
    min_band_area : int
        Minimum connected-component area (pixels) to be counted as a band.
    myo_overlap_threshold : float
        Fraction of myocardium pixels that must be dark to flag bands_in_myocardium.
    """

    TOOL_NAME = TOOL_NAME

    def __init__(
        self,
        dark_fraction: float = 0.35,
        min_band_area: int = 40,
        myo_overlap_threshold: float = 0.05,
    ):
        self.dark_fraction = dark_fraction
        self.min_band_area = min_band_area
        self.myo_overlap_threshold = myo_overlap_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        images: List[np.ndarray],
        offsets_hz: List[float],
        seg_result: SegmentationResult,
    ) -> List[BandingResult]:
        """
        Run banding detection on all slices for one patient.

        Parameters
        ----------
        images : list of H×W float32 arrays (values normalised 0-1)
        offsets_hz : list of floats
        seg_result : SegmentationResult from SegmentationTool

        Returns
        -------
        List of BandingResult, one per slice.
        """
        results = []
        for idx, (img, hz) in enumerate(zip(images, offsets_hz)):
            result = self._detect_one(img, hz, idx, seg_result)
            results.append(result)
            logger.debug(
                "Banding[%s][%d @ %.0f Hz]: present=%s myo=%s severity=%.3f",
                seg_result.patient_id, idx, hz,
                result.band_present, result.bands_in_myocardium, result.severity,
            )
        return results

    # ------------------------------------------------------------------
    # Per-slice detection
    # ------------------------------------------------------------------

    def _detect_one(
        self,
        img: np.ndarray,
        offset_hz: float,
        slice_idx: int,
        seg: SegmentationResult,
    ) -> BandingResult:
        from scipy import ndimage as ndi

        if img.ndim == 3:
            gray = img.mean(axis=2)
        else:
            gray = img.copy()

        # Ensure float32 0-1
        if gray.max() > 1.0:
            gray = gray / gray.max()

        # Compute threshold relative to the overall image median
        median_val = float(np.median(gray))
        dark_thresh = median_val * self.dark_fraction

        # Binary dark mask
        dark = gray < dark_thresh

        # Label connected components; filter by minimum area
        labeled, n = ndi.label(dark)
        band_mask = np.zeros_like(dark)
        for lb in range(1, n + 1):
            cc = labeled == lb
            if cc.sum() >= self.min_band_area:
                band_mask |= cc

        band_present = bool(band_mask.any())
        band_pixel_fraction = float(band_mask.sum()) / (gray.shape[0] * gray.shape[1])

        # Intersect with cardiac ROI (resize/transpose mask if shapes differ)
        myo_mask = self._align_mask(seg.myocardium_mask, gray.shape)
        pool_mask = self._align_mask(seg.lv_pool_mask, gray.shape)

        myo_area = myo_mask.sum()
        pool_area = pool_mask.sum()

        myo_dark = band_mask & myo_mask
        pool_dark = band_mask & pool_mask

        myo_overlap_frac = float(myo_dark.sum()) / max(myo_area, 1)
        pool_overlap_frac = float(pool_dark.sum()) / max(pool_area, 1)

        bands_in_myo = bool(myo_overlap_frac >= self.myo_overlap_threshold)
        bands_in_pool = bool(pool_overlap_frac >= self.myo_overlap_threshold)

        # Severity: weighted combination of band_pixel_fraction and myo_overlap
        severity = float(
            0.4 * min(band_pixel_fraction / 0.15, 1.0)
            + 0.6 * min(myo_overlap_frac / 0.25, 1.0)
        )
        severity = min(severity, 1.0)

        # Which AHA sectors are affected?
        band_sectors: List[str] = []
        for sector_name, sector_mask in seg.sectors.items():
            sector_aligned = self._align_mask(sector_mask, gray.shape)
            overlap = (band_mask & sector_aligned).sum()
            if overlap >= max(5, self.min_band_area // 4):
                band_sectors.append(sector_name)

        return BandingResult(
            patient_id=seg.patient_id,
            slice_idx=slice_idx,
            offset_hz=offset_hz,
            band_present=band_present,
            bands_in_myocardium=bands_in_myo,
            bands_in_lv_pool=bands_in_pool,
            severity=severity,
            myo_overlap_fraction=myo_overlap_frac,
            band_sectors=band_sectors,
            band_mask=band_mask,
            band_pixel_fraction=band_pixel_fraction,
            provenance={
                "tool": TOOL_NAME,
                "slice_idx": slice_idx,
                "offset_hz": offset_hz,
                "dark_fraction_threshold": self.dark_fraction,
            },
        )

    @staticmethod
    def _align_mask(mask: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Make mask match target HxW via transpose or nearest resize."""
        th, tw = int(target_shape[0]), int(target_shape[1])
        mh, mw = mask.shape[:2]
        if (mh, mw) == (th, tw):
            return mask.astype(bool)
        if (mh, mw) == (tw, th):
            return mask.T.astype(bool)
        from PIL import Image as PILImage
        pil = PILImage.fromarray(mask.astype(np.uint8) * 255)
        pil = pil.resize((tw, th), PILImage.NEAREST)
        return (np.asarray(pil) > 127)

    # ------------------------------------------------------------------
    # Utility: generate overlay image for visualisation
    # ------------------------------------------------------------------

    def overlay(self, img: np.ndarray, band_mask: np.ndarray) -> np.ndarray:
        """Return RGB array with band pixels highlighted in red."""
        if img.max() > 1.0:
            img = img / img.max()
        rgb = np.stack([img, img, img], axis=-1)
        rgb[band_mask, 0] = 1.0  # red channel
        rgb[band_mask, 1] = 0.0
        rgb[band_mask, 2] = 0.0
        return (rgb * 255).astype(np.uint8)
