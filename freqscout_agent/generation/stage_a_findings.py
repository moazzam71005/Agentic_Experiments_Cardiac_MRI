"""
Stage A: Structured Findings Generator (T=0, deterministic JSON)
----------------------------------------------------------------
Produces a schema-constrained JSON document from the evidence state.
No LLM required – values are assembled directly from the evidence packets.
The LLM is invoked only to produce a concise per-slice "verdict" string that
synthesises multiple evidence sources.

Output schema
-------------
{
  "patient_id": str,
  "per_slice": [
    {
      "slice_idx": int,
      "offset_hz": float,
      "banding": {
        "present": bool,
        "location": str,         # AHA sector(s) or "none" / "peripheral"
        "severity": str,         # "none"|"mild"|"moderate"|"severe"
        "overlaps_myocardium": bool,
        "overlaps_lv_pool": bool
      },
      "homogeneity": float,      # 0–1 (= 1 - CoV)
      "cnr": float,
      "neural_score": float,     # softmax-normalised confidence for this slice
      "vlm_verdict": str | null, # "optimal"|"acceptable"|"suboptimal"|"reject"|null
      "composite_score": float,  # 0–1 combined ranking score
      "verdict": str             # "optimal"|"acceptable"|"suboptimal"|"reject"
    }
  ],
  "recommended_offset": {
    "slice_idx": int,
    "offset_hz": float,
    "confidence": float,
    "confidence_label": str
  },
  "alternative_offsets": [
    {"slice_idx": int, "offset_hz": float, "verdict": str}
  ],
  "evidence_agreement": str,     # "full"|"partial"|"conflicted"
  "disagreements": [str],
  "evidence_sources": [str]
}
"""

from __future__ import annotations

import json
import logging
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..agent.evidence_state import EvidenceState
from ..agent.consistency import ConsistencyReport
from ..tools.passband import PassbandEstimator, PassbandResult

logger = logging.getLogger(__name__)


class StageAGenerator:

    # Composite score weights — Exp 10 (passband + myocardial banded fraction).
    # VLM remains at 0: Exp9 showed 7B/2B are center-biased and 0/20 accurate.
    # Weights may be overwritten by LOOCV fusion (see eval.fusion).
    # Deploy weights from Exp10 full-data grid search (2*Top-1 + ±1 objective):
    # banded_fraction_myo is the strongest physics signal (38.5% Top-1 alone).
    W_NEURAL     = 0.20
    W_HOMOG      = 0.00
    W_NO_BURDEN  = 0.00
    W_NO_BFM     = 0.70
    W_PASSBAND   = 0.00
    W_CENTER     = 0.10
    W_VLM        = 0.00

    CENTER_SIGMA_HZ = 100.0

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        if weights:
            self.W_NEURAL   = float(weights.get("neural", self.W_NEURAL))
            self.W_HOMOG    = float(weights.get("homog", self.W_HOMOG))
            self.W_NO_BURDEN = float(weights.get("no_burden", self.W_NO_BURDEN))
            self.W_NO_BFM   = float(weights.get("no_bfm", self.W_NO_BFM))
            self.W_PASSBAND = float(weights.get("passband", self.W_PASSBAND))
            self.W_CENTER   = float(weights.get("center", self.W_CENTER))
            self.W_VLM      = float(weights.get("vlm", self.W_VLM))
        self._passband = PassbandEstimator()

    def generate(
        self,
        state: EvidenceState,
        consistency: ConsistencyReport,
    ) -> Dict:
        """
        Assemble Stage-A structured findings from the evidence state.
        Returns a plain dict (JSON-serialisable).
        """
        n = state.n_slices
        offsets = state.offsets_hz

        # --- Collect evidence per slice ---
        neural_conf   = self._neural_conf(state, n)
        banding_data  = self._banding_data(state, n)
        quality_data  = self._quality_data(state, n)
        vlm_data      = self._vlm_data(state, n)

        # --- Myocardial passband model (Exp 10) ---
        pb: PassbandResult = self._passband.estimate(offsets, quality_data, banding_data)
        accept_set = set(pb.acceptable_set)
        reject_set = set(pb.rejected_set)

        # --- Composite score ---
        composite = self._composite(
            neural_conf, banding_data, quality_data, vlm_data, n, offsets,
            passband_prox=np.array(pb.per_offset_proximity),
        )
        max_comp = float(composite.max()) if n else 0.0

        # --- Per-slice findings (verdict relative to this patient's best) ---
        per_slice = []
        for i in range(n):
            b = banding_data[i]
            q = quality_data[i]
            v = vlm_data[i]
            verdict = self._slice_verdict(
                b, q, composite[i], max_comp,
                in_acceptable=(i in accept_set),
                in_rejected=(i in reject_set),
            )
            ps = {
                "slice_idx": int(i),
                "offset_hz": float(offsets[i]),
                "banding": {
                    "present":             bool(b["present"]),
                    "location":            str(b["location"]),
                    "severity":            str(b["severity"]),
                    "overlaps_myocardium": bool(b["overlaps_myo"]),
                    "overlaps_lv_pool":    bool(b["overlaps_pool"]),
                },
                "homogeneity":           round(float(q["homogeneity"]), 4),
                "cnr":                   round(float(q["cnr"]), 3),
                "mu_myo":                round(float(q.get("mu_myo", 0.0)), 4),
                "banded_fraction_myo":   round(float(q.get("banded_fraction_myo", 0.0)), 4),
                "band_trough_depth":     round(float(q.get("band_trough_depth", 0.0)), 4),
                "banding_burden":        round(float(b.get("banding_burden", 0.0)), 4),
                "neural_score":          round(float(neural_conf[i]), 4),
                "passband_quality":      round(float(pb.passband_quality[i]), 4),
                "passband_proximity":    round(float(pb.per_offset_proximity[i]), 4),
                "vlm_verdict":           v["verdict"],
                "composite_score":       round(float(composite[i]), 4),
                "verdict":               verdict,
                "reason":                pb.per_offset_reason[i],
            }
            per_slice.append(ps)

        # --- Recommendation ---
        best_idx   = int(np.argmax(composite))

        # Exp11: passband-margin confidence (doctor-trustable rules)
        rec_conf, rec_label, conf_basis = self._passband_confidence(
            best_idx=best_idx,
            peak_idx=pb.passband_peak_idx,
            quality_data=quality_data,
            banding_data=banding_data,
            accept_set=accept_set,
            consistency=consistency,
        )

        # Graded sets — neighbours STRICTLY from passband acceptable window
        # (do not promote extreme offsets just because verdict is "acceptable")
        acceptable_neighbors = []
        rejected_offsets = []
        for i, ps in enumerate(per_slice):
            entry = {
                "slice_idx": i,
                "offset_hz": float(offsets[i]),
                "verdict": ps["verdict"],
                "reason": ps["reason"],
                "composite_score": ps["composite_score"],
            }
            if i == best_idx:
                continue
            if i in accept_set:
                acceptable_neighbors.append(entry)
            if ps["verdict"] == "reject" or i in reject_set:
                rejected_offsets.append(entry)

        # Legacy alternative_offsets = acceptable neighbors ∪ neural top-3
        neural_pkts = state.get_for_tool("neural")
        neural_top3 = list(neural_pkts[0].data.top3_idx) if neural_pkts else []
        alts = list(acceptable_neighbors)
        seen = {best_idx} | {a["slice_idx"] for a in alts}
        for i in neural_top3:
            if i in seen or i < 0 or i >= n:
                continue
            alts.append({
                "slice_idx": int(i),
                "offset_hz": float(offsets[i]),
                "verdict": per_slice[i]["verdict"],
                "reason": per_slice[i]["reason"],
                "composite_score": per_slice[i]["composite_score"],
            })
            seen.add(i)

        if len(consistency.disagreements) == 0:
            agreement = "full"
        elif len(consistency.disagreements) <= 1:
            agreement = "partial"
        else:
            agreement = "conflicted"

        rec_reason = per_slice[best_idx]["reason"]
        doc = {
            "patient_id": state.patient_id,
            "per_slice": per_slice,
            "recommended_offset": {
                "slice_idx": best_idx,
                "offset_hz": float(offsets[best_idx]),
                "confidence": round(rec_conf, 3),
                "confidence_label": rec_label,
                "confidence_basis": conf_basis,
                "reason": rec_reason,
                "verdict": per_slice[best_idx]["verdict"],
            },
            "acceptable_neighbors": acceptable_neighbors,
            "rejected_offsets": rejected_offsets,
            "passband": {
                "peak_idx": pb.passband_peak_idx,
                "peak_hz": pb.passband_peak_hz,
                "acceptable_set": list(pb.acceptable_set),
                "rejected_set": list(pb.rejected_set),
            },
            "alternative_offsets": alts,
            "neural_top3": [int(i) for i in neural_top3],
            "evidence_agreement": agreement,
            "disagreements": consistency.disagreements,
            "evidence_sources": state.tools_present(),
        }
        logger.info(
            "Stage-A done: recommend slice %d (%+.0f Hz), conf=%.2f, "
            "acceptable=%d rejected=%d",
            best_idx, offsets[best_idx], rec_conf,
            len(acceptable_neighbors), len(rejected_offsets),
        )
        return doc

    # ------------------------------------------------------------------
    # Evidence extraction
    # ------------------------------------------------------------------

    def _neural_conf(self, state: EvidenceState, n: int) -> np.ndarray:
        pkts = state.get_for_tool("neural")
        if pkts and hasattr(pkts[0].data, "softmax_confidence"):
            return np.array(pkts[0].data.softmax_confidence[:n])
        return np.ones(n) / n

    def _banding_data(self, state: EvidenceState, n: int) -> List[Dict]:
        default = {
            "present": False, "location": "unknown", "severity": "none",
            "severity_float": 0.0, "overlaps_myo": False, "overlaps_pool": False,
            "banding_burden": 0.0, "myo_overlap_fraction": 0.0, "band_depth_in_myo": 0.0,
        }
        lookup = {}
        for pkt in state.get_for_tool("banding"):
            b = pkt.data
            lookup[b.slice_idx] = {
                "present":            b.band_present,
                "location":           ", ".join(b.band_sectors) if b.band_sectors else "none",
                "severity":           self._severity_label(b.severity),
                "severity_float":     b.severity,
                "overlaps_myo":       b.bands_in_myocardium,
                "overlaps_pool":      b.bands_in_lv_pool,
                "banding_burden":     getattr(b, "banding_burden", b.severity),
                "myo_overlap_fraction": b.myo_overlap_fraction,
                "band_depth_in_myo":  getattr(b, "band_depth_in_myo", 0.0),
            }
        return [lookup.get(i, dict(default)) for i in range(n)]

    def _quality_data(self, state: EvidenceState, n: int) -> List[Dict]:
        default = {"homogeneity": 0.5, "cnr": 5.0, "snr": 10.0,
                   "banded_fraction_myo": 0.0, "band_trough_depth": 0.0,
                   "mu_myo": 0.0}
        lookup = {}
        for pkt in state.get_for_tool("quality"):
            q = pkt.data
            lookup[q.slice_idx] = {
                "homogeneity":        q.homogeneity,
                "cnr":                q.cnr,
                "snr":                q.snr,
                "banded_fraction_myo": getattr(q, "banded_fraction_myo", 0.0),
                "band_trough_depth":  getattr(q, "band_trough_depth", 0.0),
                "mu_myo":             getattr(q, "mu_myo", 0.0),
            }
        return [lookup.get(i, dict(default)) for i in range(n)]

    def _vlm_data(self, state: EvidenceState, n: int) -> List[Dict]:
        default = {"verdict": None, "severity": "unknown", "uniformity": "unknown"}
        lookup = {}
        for pkt in state.get_for_tool("vlm"):
            v = pkt.data
            lookup[v.slice_idx] = {
                "verdict":    v.overall_verdict,
                "severity":   v.banding_severity,
                "uniformity": v.myocardial_uniformity,
            }
        return [lookup.get(i, dict(default)) for i in range(n)]

    # ------------------------------------------------------------------
    # Exp11 confidence
    # ------------------------------------------------------------------

    def _passband_confidence(
        self,
        best_idx: int,
        peak_idx: int,
        quality_data: List[Dict],
        banding_data: List[Dict],
        accept_set: set,
        consistency: ConsistencyReport,
    ) -> tuple:
        """
        High   — primary == passband peak, bfm margin ≥ 15%, no myo overlap
        Medium — primary within ±1 step of peak OR in acceptable set
        Low    — flat passband (bfm margin < 5%), no mask signal, or large disagreement
        """
        bfms = [float(q.get("banded_fraction_myo", 1.0)) for q in quality_data]
        if len(bfms) >= 2:
            sorted_bfm = sorted(bfms)
            bfm_margin = float(sorted_bfm[1] - sorted_bfm[0])
        else:
            bfm_margin = 0.0

        no_overlap = not (
            banding_data[best_idx].get("overlaps_myo", False)
            or banding_data[best_idx].get("overlaps_pool", False)
        )
        at_peak = best_idx == peak_idx
        near_peak = abs(best_idx - peak_idx) <= 1
        flat = bfm_margin < 0.05
        n_disagree = len(consistency.disagreements)

        if flat or (not no_overlap):
            label, conf = "low", 0.25
            basis = (
                f"flat passband (bfm margin {bfm_margin:.1%})"
                if flat else "primary overlaps myocardium/pool"
            )
        elif at_peak and bfm_margin >= 0.15 and no_overlap and n_disagree <= 1:
            label = "high"
            conf = float(np.clip(0.70 + min(0.25, bfm_margin), 0.70, 0.95))
            basis = (
                f"primary is passband peak; bfm margin {bfm_margin:.1%}; "
                f"sources {'agree' if n_disagree == 0 else 'mostly agree'}"
            )
        elif near_peak or best_idx in accept_set:
            label, conf = "medium", 0.48
            basis = (
                f"primary within ±1 step of passband peak "
                f"(peak idx {peak_idx}, primary {best_idx}); bfm margin {bfm_margin:.1%}"
            )
        else:
            label, conf = "low", 0.28
            basis = (
                f"primary distant from passband peak (peak {peak_idx}, "
                f"primary {best_idx}); {n_disagree} source disagreements"
            )

        # Blend lightly with consistency score so agreement still matters
        conf = float(np.clip(0.80 * conf + 0.20 * consistency.confidence, 0.10, 0.95))
        return conf, label, basis

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _composite(
        self,
        neural: np.ndarray,
        banding: List[Dict],
        quality: List[Dict],
        vlm: List[Dict],
        n: int,
        offsets_hz: Optional[List[float]] = None,
        passband_prox: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        scores = np.zeros(n)

        if offsets_hz is not None and len(offsets_hz) == n:
            hz_arr = np.array([float(h) for h in offsets_hz])
            center_prior = np.exp(-0.5 * (hz_arr / self.CENTER_SIGMA_HZ) ** 2)
            center_prior = center_prior / (center_prior.max() + 1e-9)
        else:
            center_prior = np.ones(n)

        if passband_prox is None or len(passband_prox) != n:
            passband_prox = np.ones(n)

        for i in range(n):
            b = banding[i]
            q = quality[i]
            v = vlm[i]

            burden     = float(b.get("banding_burden", b["severity_float"]))
            no_burden  = 1.0 - burden
            homog      = float(q.get("homogeneity", 0.5))
            no_bfm     = 1.0 - float(q.get("banded_fraction_myo", 0.0))
            vlm_score  = {"optimal": 1.0, "acceptable": 0.7, "suboptimal": 0.3,
                          "reject": 0.0, None: 0.5, "unknown": 0.5}.get(v["verdict"], 0.5)

            scores[i] = (
                self.W_NEURAL      * neural[i]
                + self.W_HOMOG     * homog
                + self.W_NO_BURDEN * no_burden
                + self.W_NO_BFM    * no_bfm
                + self.W_PASSBAND  * float(passband_prox[i])
                + self.W_CENTER    * center_prior[i]
                + self.W_VLM       * vlm_score
            )
        for i in range(n):
            if banding[i]["overlaps_myo"] or banding[i]["overlaps_pool"]:
                scores[i] *= 0.25
        return scores

    def _slice_verdict(
        self,
        b: Dict,
        q: Dict,
        composite: float,
        max_comp: float,
        in_acceptable: bool,
        in_rejected: bool,
    ) -> str:
        # Hard veto: banding physically inside the myocardium.
        if b["overlaps_myo"] or b["overlaps_pool"] or in_rejected:
            return "reject"
        burden = float(b.get("banding_burden", b["severity_float"]))
        bfm = float(q.get("banded_fraction_myo", 0.0))
        if burden >= 0.60 or bfm >= 0.25:
            return "suboptimal"
        # Relative to this patient's best composite (absolute 0.60 never fired).
        rel = composite / (max_comp + 1e-9)
        if in_acceptable and rel >= 0.95:
            return "optimal"
        if in_acceptable or rel >= 0.70:
            return "acceptable"
        return "suboptimal"

    @staticmethod
    def _severity_label(sev: float) -> str:
        if sev < 0.1:   return "none"
        if sev < 0.35:  return "mild"
        if sev < 0.65:  return "moderate"
        return "severe"
