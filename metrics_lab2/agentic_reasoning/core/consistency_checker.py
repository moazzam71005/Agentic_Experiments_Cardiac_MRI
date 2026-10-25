"""
Consistency Checker (Exp 13 — Extended for Arbitration Routing)
---------------------------------------------------------------
Adds a new 'borderline_veto' status for BFM in the 0.06-0.10 range where
the hard veto may be too aggressive. Both 'divergent' and 'borderline_veto'
cases are routed to the LLM Arbitrator in Exp 13.

Agreement status values:
  full_consensus   - CNN, physics, homogeneity all agree on same image
  passband_plateau - CNN and physics agree within ±25 Hz (clinical tolerance)
  physics_veto     - BFM > 0.10: hard veto, no arbitration needed
  borderline_veto  - BFM in (0.06, 0.10]: route to LLM arbitrator
  divergent        - CNN and physics disagree by > 25 Hz: route to LLM arbitrator
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .evidence_store import EvidenceStore


@dataclass
class ConsistencyReport:
    patient_id: str
    cnn_top1_img: int
    cnn_top1_hz: float
    physics_top1_img: int
    physics_top1_hz: float
    homog_top1_img: int
    homog_top1_hz: float
    agreement_status: str       # "full_consensus" | "passband_plateau" | "physics_veto"
                                # | "borderline_veto" | "divergent"
    is_reconciled: bool
    confidence_score: float     # 0.0 to 1.0
    confidence_label: str       # "high" | "medium" | "low"
    confidence_basis: str
    requires_arbitration: bool  # True if LLM arbitrator should be invoked
    disagreements: List[str] = field(default_factory=list)


class ConsistencyChecker:

    # Hard veto threshold: BFM > this is always rejected, no arbitration
    HARD_VETO_BFM = 0.10
    # Borderline veto zone: BFM in (BORDERLINE_BFM, HARD_VETO_BFM] → route to arbitrator
    BORDERLINE_BFM = 0.06

    def check(self, store: EvidenceStore) -> ConsistencyReport:
        cnn_map = store.get_metric_map("cnn_score")
        bfm_map = store.get_metric_map("banded_fraction_myo")
        hom_map = store.get_metric_map("homogeneity")

        offsets = store.offsets_hz

        # 1. Neural Top-1
        if cnn_map:
            cnn_top1_img = max(cnn_map.keys(), key=lambda k: cnn_map[k])
        else:
            cnn_top1_img = 7
        cnn_top1_hz = offsets[cnn_top1_img - 1] if cnn_top1_img - 1 < len(offsets) else 0.0

        # 2. Physics Top-1 (argmin bfm; tie-break with highest homogeneity)
        if bfm_map:
            min_bfm = min(bfm_map.values())
            candidates = [k for k, v in bfm_map.items() if abs(v - min_bfm) < 1e-4]
            if hom_map and len(candidates) > 1:
                physics_top1_img = max(candidates, key=lambda k: hom_map.get(k, 0.0))
            else:
                physics_top1_img = candidates[0]
        else:
            physics_top1_img = 7
        physics_top1_hz = offsets[physics_top1_img - 1] if physics_top1_img - 1 < len(offsets) else 0.0

        # 3. Homogeneity Top-1
        if hom_map:
            homog_top1_img = max(hom_map.keys(), key=lambda k: hom_map[k])
        else:
            homog_top1_img = 7
        homog_top1_hz = offsets[homog_top1_img - 1] if homog_top1_img - 1 < len(offsets) else 0.0

        disagreements = []

        # Analyse alignment
        cnn_bfm = bfm_map.get(cnn_top1_img, 0.0)
        delta_cnn_phys_hz = abs(cnn_top1_hz - physics_top1_hz)

        # ── Decision tree ──────────────────────────────────────────────────────

        # Case A: Hard veto — CNN picked a severely banded image
        if cnn_bfm > self.HARD_VETO_BFM:
            agreement_status = "physics_veto"
            requires_arbitration = False
            disagreements.append(
                f"CNN selected image #{cnn_top1_img} ({cnn_top1_hz:+.0f} Hz), "
                f"but it has severe myocardial banding (BFM={cnn_bfm:.3f} > {self.HARD_VETO_BFM}). "
                f"Hard-overruled by physics detector."
            )
            confidence_score = 0.65
            confidence_label = "medium"
            confidence_basis = (
                f"Physics hard-veto applied: CNN favoured an offset with >{self.HARD_VETO_BFM:.0%} "
                f"myocardial banding. Physics/BFM pick adopted unconditionally."
            )

        # Case B: Borderline veto — CNN picked a borderline-banded image → arbitrate
        elif self.BORDERLINE_BFM < cnn_bfm <= self.HARD_VETO_BFM:
            agreement_status = "borderline_veto"
            requires_arbitration = True
            disagreements.append(
                f"CNN selected image #{cnn_top1_img} ({cnn_top1_hz:+.0f} Hz) which has borderline "
                f"myocardial banding (BFM={cnn_bfm:.3f}, threshold={self.HARD_VETO_BFM}). "
                f"Routed to LLM arbitrator for context-sensitive resolution."
            )
            confidence_score = 0.60
            confidence_label = "medium"
            confidence_basis = (
                f"Borderline physics conflict: CNN BFM={cnn_bfm:.3f} is in grey zone. "
                f"LLM arbitration invoked."
            )

        # Case C: Full consensus — all three tools agree
        elif cnn_top1_img == physics_top1_img and cnn_top1_img == homog_top1_img:
            agreement_status = "full_consensus"
            requires_arbitration = False
            confidence_score = 0.92
            confidence_label = "high"
            confidence_basis = (
                "Full consensus: CNN, BFM detector, and signal homogeneity all independently "
                "select the identical offset."
            )

        # Case D: Passband plateau — CNN and physics agree within ±25 Hz
        elif delta_cnn_phys_hz <= 25.0:
            agreement_status = "passband_plateau"
            requires_arbitration = False
            confidence_score = 0.82
            confidence_label = "high"
            confidence_basis = (
                f"Passband plateau consensus: CNN (#{cnn_top1_img}) and Physics (#{physics_top1_img}) "
                f"agree within clinical tolerance window (±25 Hz)."
            )

        # Case E: Divergent — tools significantly disagree → route to LLM arbitrator
        else:
            agreement_status = "divergent"
            requires_arbitration = True
            disagreements.append(
                f"CNN favours #{cnn_top1_img} ({cnn_top1_hz:+.0f} Hz), "
                f"Physics favours #{physics_top1_img} ({physics_top1_hz:+.0f} Hz), "
                f"Homogeneity favours #{homog_top1_img} ({homog_top1_hz:+.0f} Hz)."
            )
            confidence_score = 0.48
            confidence_label = "low"
            confidence_basis = (
                "Divergent evidence: CNN and quantitative metrics prefer distinct frequency zones. "
                "LLM arbitration invoked for context-sensitive resolution."
            )

        return ConsistencyReport(
            patient_id=store.patient_id,
            cnn_top1_img=cnn_top1_img,
            cnn_top1_hz=cnn_top1_hz,
            physics_top1_img=physics_top1_img,
            physics_top1_hz=physics_top1_hz,
            homog_top1_img=homog_top1_img,
            homog_top1_hz=homog_top1_hz,
            agreement_status=agreement_status,
            is_reconciled=agreement_status not in ("divergent", "borderline_veto"),
            confidence_score=round(confidence_score, 2),
            confidence_label=confidence_label,
            confidence_basis=confidence_basis,
            requires_arbitration=requires_arbitration,
            disagreements=disagreements,
        )
