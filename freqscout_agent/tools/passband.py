"""
Myocardial Passband Model
--------------------------
Uses per-offset quality and banding metrics (already computed from the
segmentation mask) to build a physics-grounded passband model for a single
patient's frequency scout.

In bSSFP, the myocardial signal follows a characteristic profile:
  - Passband: high, uniform signal; banding nulls far from heart
  - Null zone: dark bands enter / cross the myocardium

Key output:
  passband_peak_idx      — offset index maximising the passband quality score
  acceptable_set         — contiguous or close-to-peak indices with no myo banding
  rejected_set           — indices where banding enters the myocardium
  per_offset_proximity   — 0–1 Gaussian score relative to passband peak
  per_offset_reason      — human-readable physical reason per offset

This module is a stateless helper; it is called directly from StageAGenerator
rather than being an agentic tool in the reasoning loop.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PassbandResult:
    passband_peak_idx: int
    passband_peak_hz: float
    passband_quality: List[float]          # 0–1 composite quality per slice
    acceptable_set: List[int]              # indices where myo banding is absent/minor
    rejected_set: List[int]               # indices where banding overlaps myo/pool
    per_offset_proximity: List[float]     # Gaussian distance from passband peak (0–1)
    per_offset_reason: List[str]          # physics reason string per offset


# ---------------------------------------------------------------------------
# Passband estimator
# ---------------------------------------------------------------------------

class PassbandEstimator:
    """
    Parameters
    ----------
    bfm_accept_thresh : float
        Max banded_fraction_myo allowed for a slice to be in the acceptable set.
    bfm_reject_thresh : float
        Min banded_fraction_myo that forces rejection (even without overlap flag).
    proximity_sigma_hz : float
        σ of the Gaussian used for proximity scoring (Hz).
    """

    def __init__(
        self,
        bfm_accept_thresh: float = 0.15,
        bfm_reject_thresh: float = 0.25,
        proximity_sigma_hz: float = 75.0,
    ):
        self.bfm_accept_thresh = bfm_accept_thresh
        self.bfm_reject_thresh = bfm_reject_thresh
        self.proximity_sigma_hz = proximity_sigma_hz

    # ------------------------------------------------------------------

    def estimate(
        self,
        offsets_hz: List[float],
        quality_data: List[Dict],   # from StageAGenerator._quality_data()
        banding_data: List[Dict],   # from StageAGenerator._banding_data()
    ) -> PassbandResult:
        n = len(offsets_hz)
        hz = np.array(offsets_hz, dtype=float)

        # --- Extract per-slice physics signals ---
        bfm     = np.array([q.get("banded_fraction_myo", 0.0) for q in quality_data])
        homog   = np.array([q.get("homogeneity", 0.5) for q in quality_data])
        cnr     = np.array([q.get("cnr", 0.0) for q in quality_data])
        mu_myo  = np.array([q.get("mu_myo", 0.0) for q in quality_data])
        in_myo  = np.array([b.get("overlaps_myo", False) for b in banding_data], dtype=bool)
        in_pool = np.array([b.get("overlaps_pool", False) for b in banding_data], dtype=bool)
        burden  = np.array([b.get("banding_burden", 0.0) for b in banding_data])

        # --- Passband quality score: high when banding is absent from myocardium ---
        # Combines (1 - bfm), homogeneity, CNR, and relative myocardial brightness.
        cnr_norm = np.clip(cnr / (cnr.max() + 1e-9), 0.0, 1.0)
        mu_norm  = np.clip(mu_myo / (mu_myo.max() + 1e-9), 0.0, 1.0)
        passband_quality = (
            0.40 * (1.0 - np.clip(bfm / 0.3, 0.0, 1.0))
            + 0.25 * homog
            + 0.20 * cnr_norm
            + 0.15 * mu_norm
        )
        # Hard penalty: any banding inside myo/pool → zero quality
        passband_quality[in_myo | in_pool] = 0.0
        passband_quality = np.clip(passband_quality, 0.0, 1.0)

        passband_peak_idx = int(np.argmax(passband_quality))
        passband_peak_hz  = float(hz[passband_peak_idx])

        # --- Acceptable set: CONTIGUOUS window around the passband peak ---
        # Extreme offsets (±150 Hz) often have low banded_fraction because the
        # null has not yet entered the myocardium — do not treat them as acceptable.
        quality_floor = 0.70 * float(passband_quality[passband_peak_idx])
        acceptable_set = [passband_peak_idx]
        # grow left
        i = passband_peak_idx - 1
        while i >= 0:
            if in_myo[i] or in_pool[i] or bfm[i] > self.bfm_accept_thresh:
                break
            if passband_quality[i] < quality_floor:
                break
            acceptable_set.append(i)
            i -= 1
        # grow right
        i = passband_peak_idx + 1
        while i < n:
            if in_myo[i] or in_pool[i] or bfm[i] > self.bfm_accept_thresh:
                break
            if passband_quality[i] < quality_floor:
                break
            acceptable_set.append(i)
            i += 1
        # Cap at peak ± 1 step (25 Hz): matches the clinical "neighbours can
        # both be good" guidance without declaring half the sweep acceptable.
        acceptable_set = sorted(
            i for i in acceptable_set if abs(i - passband_peak_idx) <= 1
        )

        # --- Rejected set: banding inside myo/pool or very high bfm ---
        rejected_set = [
            i for i in range(n)
            if in_myo[i] or in_pool[i] or bfm[i] > self.bfm_reject_thresh
        ]

        # --- Proximity: Gaussian distance from passband peak (over acceptable_set) ---
        peak_hz = hz[passband_peak_idx]
        proximity = np.exp(-0.5 * ((hz - peak_hz) / self.proximity_sigma_hz) ** 2)
        # Zero out rejected slices
        for i in rejected_set:
            proximity[i] = 0.0
        if proximity.max() > 0:
            proximity = proximity / proximity.max()

        # --- Per-offset reasons ---
        reasons = self._build_reasons(
            n, hz, passband_peak_idx, passband_quality,
            bfm, homog, cnr_norm, in_myo, in_pool, burden,
            banding_data,
        )

        return PassbandResult(
            passband_peak_idx=passband_peak_idx,
            passband_peak_hz=passband_peak_hz,
            passband_quality=passband_quality.tolist(),
            acceptable_set=acceptable_set,
            rejected_set=rejected_set,
            per_offset_proximity=proximity.tolist(),
            per_offset_reason=reasons,
        )

    # ------------------------------------------------------------------

    def _build_reasons(
        self,
        n: int,
        hz: np.ndarray,
        peak_idx: int,
        quality: np.ndarray,
        bfm: np.ndarray,
        homog: np.ndarray,
        cnr_norm: np.ndarray,
        in_myo: np.ndarray,
        in_pool: np.ndarray,
        burden: np.ndarray,
        banding_data: List[Dict],
    ) -> List[str]:
        reasons = []
        for i in range(n):
            hz_i   = float(hz[i])
            bfm_i  = float(bfm[i])
            h_i    = float(homog[i])
            b_i    = float(burden[i])
            loc    = banding_data[i].get("location", "none") or "none"

            if in_myo[i] or in_pool[i]:
                target = "myocardium" if in_myo[i] else "LV blood pool"
                reasons.append(
                    f"bSSFP null crosses {target}"
                    + (f" ({loc} sector)" if loc not in ("none", "unknown") else "")
                    + f": banding burden {b_i:.0%} — REJECT"
                )
            elif bfm_i > self.bfm_reject_thresh:
                reasons.append(
                    f"high intra-myocardial banded fraction ({bfm_i:.0%} > {self.bfm_reject_thresh:.0%}): "
                    f"probable null approaching myocardial wall — suboptimal"
                )
            elif i == peak_idx:
                reasons.append(
                    f"passband peak: myocardial signal most uniform "
                    f"(banded fraction {bfm_i:.1%}, homogeneity {h_i:.2f}); "
                    f"banding null outside cardiac ROI — OPTIMAL"
                )
            elif bfm_i <= self.bfm_accept_thresh:
                dist_hz = abs(hz_i - float(hz[peak_idx]))
                reasons.append(
                    f"near-passband: banded fraction {bfm_i:.1%}, homogeneity {h_i:.2f}"
                    + (f"; banding {loc}" if loc not in ("none", "unknown") else "")
                    + f" — acceptable ({dist_hz:.0f} Hz from peak)"
                )
            else:
                reasons.append(
                    f"marginal: banded fraction {bfm_i:.1%} above acceptable threshold "
                    f"({self.bfm_accept_thresh:.0%}); "
                    + (f"banding in {loc}" if loc not in ("none", "unknown") else "some peripheral banding")
                    + " — suboptimal"
                )

        return reasons
