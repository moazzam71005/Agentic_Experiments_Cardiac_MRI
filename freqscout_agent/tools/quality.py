"""
T3 Quality Metrics
------------------
Computes per-slice, per-offset image quality metrics standard in bSSFP cardiac MRI:
  - CoV  (Coefficient of Variation) of myocardial intensity: lower = more uniform
  - CNR  (Contrast-to-Noise Ratio) between LV blood pool and myocardium
  - SNR  (Signal-to-Noise Ratio) of the myocardium
  - homogeneity  = 1 - CoV  (higher = better)

All outputs are provenance-tagged.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .segment import SegmentationResult

logger = logging.getLogger(__name__)

TOOL_NAME = "quality"


@dataclass
class QualityResult:
    patient_id: str
    slice_idx: int
    offset_hz: float

    cov: float              # coefficient of variation (myocardium intensity), lower = better
    homogeneity: float      # 1 - CoV, higher = better
    cnr: float              # (mu_blood - mu_myo) / sigma_background
    snr: float              # mu_myo / sigma_background
    mu_myo: float           # mean myocardial intensity
    sigma_myo: float        # std of myocardial intensity
    mu_blood: float         # mean blood pool intensity
    sigma_bg: float         # estimate of background noise std

    # --- New intra-myocardial banding metrics (require real mask) ---
    banded_fraction_myo: float  # fraction of myo pixels below 60% of mu_myo (banding threshold)
    myo_min_signal: float       # min myocardial intensity (lowest band trough)
    band_trough_depth: float    # (mu_myo - myo_min_signal) / mu_myo; 0–1, higher = deeper band

    provenance: Dict = field(default_factory=dict)


class QualityMetrics:

    TOOL_NAME = TOOL_NAME

    def __init__(self, background_percentile: float = 5.0):
        """
        background_percentile : float
            Percentile used to estimate background noise region
            (lowest intensity pixels in the image corners).
        """
        self.background_percentile = background_percentile

    # ------------------------------------------------------------------

    def run(
        self,
        images: List[np.ndarray],
        offsets_hz: List[float],
        seg_result: SegmentationResult,
    ) -> List[QualityResult]:
        """
        Compute quality metrics for every slice in the patient's frequency scout.
        """
        results = []
        for idx, (img, hz) in enumerate(zip(images, offsets_hz)):
            result = self._compute_one(img, hz, idx, seg_result)
            results.append(result)
            logger.debug(
                "Quality[%s][%d @ %.0f Hz]: CoV=%.3f CNR=%.2f SNR=%.2f",
                seg_result.patient_id, idx, hz,
                result.cov, result.cnr, result.snr,
            )
        return results

    # ------------------------------------------------------------------

    def _compute_one(
        self,
        img: np.ndarray,
        offset_hz: float,
        slice_idx: int,
        seg: SegmentationResult,
    ) -> QualityResult:
        if img.ndim == 3:
            gray = img.mean(axis=2).astype(np.float32)
        else:
            gray = img.astype(np.float32)

        if gray.max() > 1.0:
            gray = gray / gray.max()

        myo_mask = self._align_mask(seg.myocardium_mask, gray.shape)
        pool_mask = self._align_mask(seg.lv_pool_mask, gray.shape)

        myo_pixels  = gray[myo_mask]
        pool_pixels = gray[pool_mask]
        bg_pixels   = self._background_pixels(gray)

        mu_myo    = float(myo_pixels.mean())   if myo_pixels.size  > 0 else 0.0
        sigma_myo = float(myo_pixels.std())    if myo_pixels.size  > 0 else 1e-6
        mu_blood  = float(pool_pixels.mean())  if pool_pixels.size > 0 else 0.0
        sigma_bg  = float(bg_pixels.std())     if bg_pixels.size   > 0 else 1e-6
        sigma_bg  = max(sigma_bg, 1e-6)

        cov         = sigma_myo / max(mu_myo, 1e-6)
        homogeneity = max(0.0, 1.0 - cov)
        cnr         = (mu_blood - mu_myo) / sigma_bg
        snr         = mu_myo / sigma_bg

        # --- Intra-myocardial banding metrics ---
        # Banding threshold: pixels below 60% of mean myocardial intensity are "banded"
        # Trough depth uses the 5th percentile (not min) so a single dark pixel
        # cannot saturate the metric at 1.0.
        band_thresh = mu_myo * 0.60
        if myo_pixels.size > 0 and mu_myo > 1e-6:
            banded_fraction_myo = float((myo_pixels < band_thresh).mean())
            myo_min_signal = float(np.percentile(myo_pixels, 5.0))
            band_trough_depth = max(0.0, (mu_myo - myo_min_signal) / mu_myo)
        else:
            banded_fraction_myo = 0.0
            myo_min_signal = 0.0
            band_trough_depth = 0.0

        return QualityResult(
            patient_id=seg.patient_id,
            slice_idx=slice_idx,
            offset_hz=offset_hz,
            cov=cov,
            homogeneity=homogeneity,
            cnr=cnr,
            snr=snr,
            mu_myo=mu_myo,
            sigma_myo=sigma_myo,
            mu_blood=mu_blood,
            sigma_bg=sigma_bg,
            banded_fraction_myo=banded_fraction_myo,
            myo_min_signal=myo_min_signal,
            band_trough_depth=band_trough_depth,
            provenance={
                "tool": TOOL_NAME,
                "slice_idx": slice_idx,
                "offset_hz": offset_hz,
                "seg_method": seg.provenance.get("method", "unknown"),
            },
        )

    # ------------------------------------------------------------------

    def _background_pixels(self, gray: np.ndarray) -> np.ndarray:
        """
        Sample background from image corners (assumed to be outside the body).
        """
        H, W = gray.shape
        corner_size = max(10, H // 8)
        corners = np.concatenate([
            gray[:corner_size, :corner_size].ravel(),
            gray[:corner_size, -corner_size:].ravel(),
            gray[-corner_size:, :corner_size].ravel(),
            gray[-corner_size:, -corner_size:].ravel(),
        ])
        return corners

    @staticmethod
    def _align_mask(mask: np.ndarray, target_shape) -> np.ndarray:
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
