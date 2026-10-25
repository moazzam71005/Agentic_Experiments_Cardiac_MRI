"""
Stage-A Generator (Exp 13 — LLM-Arbitration Aware)
---------------------------------------------------
Identical to Exp 12 except:
  - Accepts an optional ArbitrationResult from the LLM Arbitrator.
  - If arbitration_result is present and audit_passed=True, the LLM's
    recommended image overrides the composite-score winner.
  - The final decision source ('deterministic' vs 'llm_arbitrated') is
    logged in the output JSON for full provenance.
  - Composite scores are returned as a list for use by the LLM Arbitrator.
"""

from __future__ import annotations

import numpy as np
from dataclasses import asdict
from typing import Any, Dict, List, Optional
from .evidence_store import EvidenceStore
from .consistency_checker import ConsistencyReport


class StageAGenerator:

    # Validated weights from Metrics Lab and Exp10/12:
    W_CNN    = 0.20
    W_NO_BFM = 0.50
    W_HOMOG  = 0.20
    W_CENTER = 0.10

    def compute_composite_scores(self, store: EvidenceStore) -> List[float]:
        """
        Compute and return composite scores for all slices.
        Called separately so the LLM Arbitrator can include scores in its prompt.
        """
        n = store.n_slices
        offsets = store.offsets_hz

        cnn_map = store.get_metric_map("cnn_score")
        bfm_map = store.get_metric_map("banded_fraction_myo")
        hom_map = store.get_metric_map("homogeneity")

        cnn_vals = np.array([cnn_map.get(i + 1, 0.0) for i in range(n)], dtype=float)
        bfm_vals = np.array([bfm_map.get(i + 1, 0.0) for i in range(n)], dtype=float)
        hom_vals = np.array([hom_map.get(i + 1, 0.0) for i in range(n)], dtype=float)
        offsets_arr = np.array(offsets, dtype=float)

        c_min, c_max = cnn_vals.min(), cnn_vals.max()
        cnn_norm = (
            (cnn_vals - c_min) / (c_max - c_min + 1e-8)
            if c_max > c_min else np.ones(n) * 0.5
        )
        no_bfm = 1.0 - bfm_vals

        h_min, h_max = hom_vals.min(), hom_vals.max()
        hom_norm = (
            (hom_vals - h_min) / (h_max - h_min + 1e-8)
            if h_max > h_min else np.ones(n) * 0.5
        )

        center_prior = np.exp(-0.5 * (offsets_arr / 100.0) ** 2)

        composite = (
            self.W_CNN    * cnn_norm
            + self.W_NO_BFM * no_bfm
            + self.W_HOMOG  * hom_norm
            + self.W_CENTER * center_prior
        )
        return composite.tolist()

    def generate(
        self,
        store: EvidenceStore,
        consistency: ConsistencyReport,
        arbitration_result=None,   # Optional[ArbitrationResult]
    ) -> Dict[str, Any]:
        n = store.n_slices
        offsets = store.offsets_hz

        cnn_map  = store.get_metric_map("cnn_score")
        bfm_map  = store.get_metric_map("banded_fraction_myo")
        hom_map  = store.get_metric_map("homogeneity")
        cnr_map  = store.get_metric_map("cnr")
        temp_map = store.get_metric_map("temporal_std_sweep")
        sec_map  = store.get_metric_map("band_sectors")
        depth_map= store.get_metric_map("band_trough_depth")

        composite = self.compute_composite_scores(store)
        composite_arr = np.array(composite)

        bfm_vals = np.array([bfm_map.get(i + 1, 0.0) for i in range(n)], dtype=float)
        hom_vals = np.array([hom_map.get(i + 1, 0.0) for i in range(n)], dtype=float)
        cnn_vals = np.array([cnn_map.get(i + 1, 0.0) for i in range(n)], dtype=float)

        # ── Determine final best index ────────────────────────────────────────
        decision_source = "deterministic"
        arbitration_note = None

        if (
            arbitration_result is not None
            and arbitration_result.audit_passed
        ):
            # LLM arbitrator succeeded — use its recommendation
            best_img = arbitration_result.recommended_image
            best_idx = best_img - 1
            best_hz  = float(arbitration_result.offset_hz)
            decision_source = "llm_arbitrated"
            arbitration_note = {
                "confidence": arbitration_result.confidence,
                "rationale":  arbitration_result.rationale,
                "audit_notes": arbitration_result.audit_notes,
            }
        else:
            # Standard deterministic composite pick
            best_idx = int(np.argmax(composite_arr))
            best_img = best_idx + 1
            best_hz  = float(offsets[best_idx])

        max_comp = float(composite_arr[best_idx])

        # ── Per-slice categorisation ──────────────────────────────────────────
        per_slice: List[Dict[str, Any]] = []
        acceptable_offsets: List[Dict[str, Any]] = []
        rejected_offsets: List[Dict[str, Any]] = []

        for i in range(n):
            img_no = i + 1
            hz     = float(offsets[i])
            bfm    = float(bfm_vals[i])
            hom    = float(hom_vals[i])
            cnn_sc = float(cnn_vals[i])
            comp_sc= float(composite_arr[i])
            sectors = str(sec_map.get(img_no, "none")).strip()
            if not sectors or sectors.lower() in ("nan", "none", "[]"):
                sectors = "none"

            is_rejected  = bfm > 0.08
            is_optimal   = (i == best_idx)
            is_acceptable = (
                not is_optimal and not is_rejected
                and comp_sc >= max_comp - 0.18
                and bfm <= 0.04
            )

            if is_optimal:
                verdict = "optimal"
                reason = (
                    f"{'LLM-arbitrated' if decision_source == 'llm_arbitrated' else 'Global multi-metric'} "
                    f"peak: minimal intra-myocardial banding, high homogeneity, and strong CNN consensus."
                )
            elif is_rejected:
                verdict = "rejected"
                reason = (
                    f"Severe off-resonance dark banding intersecting myocardium "
                    f"(banded fraction={bfm:.1%}); affected sectors: {sectors}."
                )
            elif is_acceptable:
                verdict = "acceptable"
                reason = (
                    f"Diagnostic passband plateau: acceptable myocardium homogeneity "
                    f"({hom:.3f}) and negligible banding ({bfm:.1%})."
                )
            else:
                verdict = "suboptimal"
                reason = f"Reduced signal homogeneity ({hom:.3f}) or early peripheral phase roll-off."

            slice_dict = {
                "image_no":            img_no,
                "offset_hz":           hz,
                "composite_score":     round(comp_sc, 4),
                "cnn_score":           round(cnn_sc, 4),
                "banded_fraction_myo": round(bfm, 4),
                "homogeneity":         round(hom, 4),
                "cnr":  round(float(cnr_map.get(img_no, 0.0)), 2),
                "temporal_std_sweep":  round(float(temp_map.get(img_no, 0.0)), 4),
                "band_sectors":        sectors,
                "verdict":             verdict,
                "reason":              reason,
            }
            per_slice.append(slice_dict)

            if is_acceptable:
                acceptable_offsets.append({
                    "image_no":            img_no,
                    "offset_hz":           hz,
                    "delta_hz":            hz - best_hz,
                    "banded_fraction_myo": round(bfm, 4),
                    "homogeneity":         round(hom, 4),
                })
            elif is_rejected:
                rejected_offsets.append({
                    "image_no":            img_no,
                    "offset_hz":           hz,
                    "banded_fraction_myo": round(bfm, 4),
                    "affected_sectors":    sectors,
                })

        # ── Build output ──────────────────────────────────────────────────────
        # Confidence: if LLM arbitrated, use its confidence label and map to score
        if decision_source == "llm_arbitrated" and arbitration_result:
            arb_conf_label = arbitration_result.confidence
            arb_conf_score = {"high": 0.85, "medium": 0.70, "low": 0.55}.get(arb_conf_label, 0.70)
            conf_score = arb_conf_score
            conf_label = arb_conf_label
            conf_basis = (
                f"LLM arbitration (Exp13): {arbitration_result.rationale}"
            )
        else:
            conf_score = consistency.confidence_score
            conf_label = consistency.confidence_label
            conf_basis = consistency.confidence_basis

        output = {
            "patient_id": store.patient_id,
            "decision_source": decision_source,   # NEW: "deterministic" | "llm_arbitrated"
            "recommended_offset": {
                "image_no":        best_img,
                "offset_hz":       best_hz,
                "composite_score": round(max_comp, 4),
                "confidence":      round(conf_score, 2),
                "confidence_label": conf_label,
                "confidence_basis": conf_basis,
            },
            "acceptable_alternatives": acceptable_offsets,
            "rejected_offsets":        rejected_offsets,
            "consistency_summary": {
                "status":           consistency.agreement_status,
                "requires_arbitration": consistency.requires_arbitration,
                "cnn_pick":         f"#{consistency.cnn_top1_img} ({consistency.cnn_top1_hz:+.0f} Hz)",
                "physics_pick":     f"#{consistency.physics_top1_img} ({consistency.physics_top1_hz:+.0f} Hz)",
                "homogeneity_pick": f"#{consistency.homog_top1_img} ({consistency.homog_top1_hz:+.0f} Hz)",
                "disagreements":    consistency.disagreements,
            },
            "per_slice": per_slice,
        }

        # Attach arbitration detail if applicable
        if arbitration_note:
            output["arbitration_detail"] = arbitration_note

        return output
