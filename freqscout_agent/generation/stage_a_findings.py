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

logger = logging.getLogger(__name__)


class StageAGenerator:

    # Composite score weights — myocardium-centric (Exp 8).
    # Primary signal is myocardial banding burden (physiologically grounded).
    # Neural ranker (Exp5 LOOCV) provides the learned prior.
    # VLM only contributes once it shows real discrimination (currently 5%).
    W_NEURAL     = 0.40  # Exp5 LOOCV out-of-fold score
    W_HOMOG      = 0.25  # 1 - intra-myocardial CoV  (real mask required)
    W_NO_BURDEN  = 0.30  # 1 - banding_burden  (intra-myocardial, bSSFP-physics-based)
    W_VLM        = 0.05  # VLM verdict  (until it discriminates; see Exp8 ablation)

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

        # --- Composite score ---
        composite = self._composite(neural_conf, banding_data, quality_data, vlm_data, n)

        # --- Per-slice findings ---
        per_slice = []
        for i in range(n):
            b = banding_data[i]
            q = quality_data[i]
            v = vlm_data[i]
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
                "banded_fraction_myo":   round(float(q.get("banded_fraction_myo", 0.0)), 4),
                "band_trough_depth":     round(float(q.get("band_trough_depth", 0.0)), 4),
                "banding_burden":        round(float(b.get("banding_burden", 0.0)), 4),
                "neural_score":          round(float(neural_conf[i]), 4),
                "vlm_verdict":           v["verdict"],
                "composite_score":       round(float(composite[i]), 4),
                "verdict":               self._slice_verdict(b, q, v, composite[i]),
            }
            per_slice.append(ps)

        # --- Recommendation ---
        best_idx   = int(np.argmax(composite))
        rec_conf   = float(consistency.confidence)
        rec_label  = consistency.confidence_label

        # Alternative offsets: high composite OR neural top-3 (so Top-3 eval is meaningful)
        threshold = composite[best_idx] * 0.80
        alts = []
        seen = {best_idx}
        for i, ps in enumerate(per_slice):
            if i in seen:
                continue
            if composite[i] >= threshold and ps["verdict"] in ("optimal", "acceptable"):
                alts.append({
                    "slice_idx": i,
                    "offset_hz": float(offsets[i]),
                    "verdict": ps["verdict"],
                })
                seen.add(i)

        neural_pkts = state.get_for_tool("neural")
        neural_top3 = list(neural_pkts[0].data.top3_idx) if neural_pkts else []
        for i in neural_top3:
            if i in seen or i < 0 or i >= n:
                continue
            alts.append({
                "slice_idx": int(i),
                "offset_hz": float(offsets[i]),
                "verdict": per_slice[i]["verdict"],
            })
            seen.add(i)

        # Evidence agreement
        if len(consistency.disagreements) == 0:
            agreement = "full"
        elif len(consistency.disagreements) <= 1:
            agreement = "partial"
        else:
            agreement = "conflicted"

        doc = {
            "patient_id": state.patient_id,
            "per_slice": per_slice,
            "recommended_offset": {
                "slice_idx": best_idx,
                "offset_hz": float(offsets[best_idx]),
                "confidence": round(rec_conf, 3),
                "confidence_label": rec_label,
            },
            "alternative_offsets": alts,
            "neural_top3": [int(i) for i in neural_top3],
            "evidence_agreement": agreement,
            "disagreements": consistency.disagreements,
            "evidence_sources": state.tools_present(),
        }
        logger.info(
            "Stage-A done: recommend slice %d (%+.0f Hz), conf=%.2f, alts=%d",
            best_idx, offsets[best_idx], rec_conf, len(alts),
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
                   "banded_fraction_myo": 0.0, "band_trough_depth": 0.0}
        lookup = {}
        for pkt in state.get_for_tool("quality"):
            q = pkt.data
            lookup[q.slice_idx] = {
                "homogeneity":        q.homogeneity,
                "cnr":                q.cnr,
                "snr":                q.snr,
                "banded_fraction_myo": getattr(q, "banded_fraction_myo", 0.0),
                "band_trough_depth":  getattr(q, "band_trough_depth", 0.0),
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
    # Scoring
    # ------------------------------------------------------------------

    def _composite(
        self, neural: np.ndarray, banding: List[Dict], quality: List[Dict], vlm: List[Dict], n: int
    ) -> np.ndarray:
        scores = np.zeros(n)
        for i in range(n):
            b = banding[i]
            q = quality[i]
            v = vlm[i]

            # Use intra-myocardial banding_burden if available; fall back to severity
            burden     = float(b.get("banding_burden", b["severity_float"]))
            no_burden  = 1.0 - burden
            homog      = float(q.get("homogeneity", 0.5))
            vlm_score  = {"optimal": 1.0, "acceptable": 0.7, "suboptimal": 0.3,
                          "reject": 0.0, None: 0.5, "unknown": 0.5}.get(v["verdict"], 0.5)

            scores[i] = (
                self.W_NEURAL    * neural[i]
                + self.W_HOMOG   * homog
                + self.W_NO_BURDEN * no_burden
                + self.W_VLM     * vlm_score
            )
        # Penalise slices where banding overlaps myocardium or LV pool
        for i in range(n):
            if banding[i]["overlaps_myo"] or banding[i]["overlaps_pool"]:
                scores[i] *= 0.25   # stronger penalty with real mask
        return scores

    def _slice_verdict(self, b: Dict, q: Dict, v: Dict, composite: float) -> str:
        # Hard veto: banding physically inside the myocardium — always reject.
        if b["overlaps_myo"] or b["overlaps_pool"]:
            return "reject"
        # Soft veto: high intra-myocardial banding burden even without overlap
        burden = float(b.get("banding_burden", b["severity_float"]))
        if burden >= 0.60:
            return "suboptimal"
        # Grade by composite score
        if composite >= 0.60:
            return "optimal"
        if composite >= 0.35:
            return "acceptable"
        return "suboptimal"

    @staticmethod
    def _severity_label(sev: float) -> str:
        if sev < 0.1:   return "none"
        if sev < 0.35:  return "mild"
        if sev < 0.65:  return "moderate"
        return "severe"
