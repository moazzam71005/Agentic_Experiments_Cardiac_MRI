"""
Consistency Checker (Exp 14 — ADRE Wide-Plateau & Conflict Detection)
----------------------------------------------------------------------
Identifies cases that require multi-agent deliberation:
  1. divergent: CNN and physics disagree by > 25 Hz
  2. borderline_veto: candidate chosen by CNN has BFM in (0.05, 0.10]
  3. wide_plateau: passband spans >= 50 Hz with multiple valid candidates,
     creating ambiguity where choosing an edge offset causes clinical errors.
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
                                # | "borderline_veto" | "divergent" | "wide_plateau"
    is_reconciled: bool
    confidence_score: float     # 0.0 to 1.0
    confidence_label: str       # "high" | "medium" | "low"
    confidence_basis: str
    requires_arbitration: bool  # True if ADRE debate loop should be invoked
    candidate_images: List[int] = field(default_factory=list) # Candidate frames for debate
    disagreements: List[str] = field(default_factory=list)


class ConsistencyChecker:

    HARD_VETO_BFM = 0.10
    BORDERLINE_BFM = 0.05
    PASSBAND_BFM_THRESHOLD = 0.035

    def check(self, store: EvidenceStore) -> ConsistencyReport:
        cnn_map = store.get_metric_map("cnn_score")
        bfm_map = store.get_metric_map("banded_fraction_myo")
        hom_map = store.get_metric_map("homogeneity")
        cnr_map = store.get_metric_map("cnr")

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
        candidate_images = list({cnn_top1_img, physics_top1_img, homog_top1_img})

        cnn_bfm = bfm_map.get(cnn_top1_img, 0.0)
        delta_cnn_phys_hz = abs(cnn_top1_hz - physics_top1_hz)

        # ── Check for Wide Plateau ─────────────────────────────────────────────
        # Find all frames that meet passband criteria (low BFM and acceptable homogeneity)
        passband_frames = [
            img for img in sorted(bfm_map.keys())
            if bfm_map.get(img, 1.0) <= self.PASSBAND_BFM_THRESHOLD
        ]
        is_wide_plateau = len(passband_frames) >= 3  # Spans >= 50 Hz (3 * 25 Hz steps)

        # ── Decision Tree ──────────────────────────────────────────────────────

        # Case A: Hard veto — CNN picked a severely banded image
        if cnn_bfm > self.HARD_VETO_BFM:
            agreement_status = "physics_veto"
            requires_arbitration = False
            disagreements.append(
                f"CNN selected image #{cnn_top1_img} ({cnn_top1_hz:+.0f} Hz), "
                f"which has severe myocardial banding (BFM={cnn_bfm:.3f} > {self.HARD_VETO_BFM}). "
                f"Hard-overruled by physics detector."
            )
            confidence_score = 0.70
            confidence_label = "medium"
            confidence_basis = (
                f"Physics hard-veto applied: CNN favoured an offset with >{self.HARD_VETO_BFM:.0%} "
                f"myocardial banding. Physics/BFM pick adopted unconditionally."
            )

        # Case B: Borderline veto — CNN picked a borderline-banded image → debate
        elif self.BORDERLINE_BFM < cnn_bfm <= self.HARD_VETO_BFM:
            agreement_status = "borderline_veto"
            requires_arbitration = True
            disagreements.append(
                f"CNN selected image #{cnn_top1_img} ({cnn_top1_hz:+.0f} Hz) which has borderline "
                f"myocardial banding (BFM={cnn_bfm:.3f}, threshold={self.HARD_VETO_BFM}). "
                f"Routed to ADRE Multi-Agent Debate."
            )
            confidence_score = 0.60
            confidence_label = "medium"
            confidence_basis = (
                f"Borderline physics conflict: CNN BFM={cnn_bfm:.3f} is in grey zone. "
                f"Multi-agent debate invoked."
            )

        # Case C: Divergent — CNN and physics disagree by > 25 Hz → debate
        elif delta_cnn_phys_hz > 25.0:
            agreement_status = "divergent"
            requires_arbitration = True
            disagreements.append(
                f"CNN favours image #{cnn_top1_img} ({cnn_top1_hz:+.0f} Hz) "
                f"while Physics favours image #{physics_top1_img} ({physics_top1_hz:+.0f} Hz) "
                f"(disagreement = {delta_cnn_phys_hz:.0f} Hz > 25 Hz tolerance). "
                f"Homogeneity favours image #{homog_top1_img} ({homog_top1_hz:+.0f} Hz). "
                f"Routed to ADRE Multi-Agent Debate."
            )
            confidence_score = 0.50
            confidence_label = "low"
            confidence_basis = (
                f"Severe metric divergence ({delta_cnn_phys_hz:.0f} Hz gap). "
                f"Requires multi-agent clinical consensus."
            )

        # Case D: Wide Plateau — Multiple viable frames on passband plateau → debate
        elif is_wide_plateau and (cnn_top1_img != physics_top1_img or cnn_top1_img != homog_top1_img):
            agreement_status = "wide_plateau"
            requires_arbitration = True
            for pf in passband_frames:
                if pf not in candidate_images:
                    candidate_images.append(pf)
            disagreements.append(
                f"Wide passband plateau identified across {len(passband_frames)} offsets "
                f"({[offsets[f-1] for f in passband_frames]} Hz). "
                f"CNN (#{cnn_top1_img}), Physics (#{physics_top1_img}), and Homogeneity (#{homog_top1_img}) "
                f"occupy different positions on the plateau. Multi-agent debate invoked to isolate true plateau center."
            )
            confidence_score = 0.65
            confidence_label = "medium"
            confidence_basis = (
                f"Wide passband plateau ({len(passband_frames)} candidate offsets). "
                f"ADRE debate invoked for plateau center calibration."
            )

        # Case E: Full consensus — all metrics agree on the exact same frame
        elif cnn_top1_img == physics_top1_img == homog_top1_img:
            agreement_status = "full_consensus"
            requires_arbitration = False
            confidence_score = 0.95
            confidence_label = "high"
            confidence_basis = (
                f"Full consensus across CNN, Physics (BFM={bfm_map.get(physics_top1_img, 0.0):.3f}), "
                f"and Homogeneity (1-CoV={hom_map.get(homog_top1_img, 0.0):.3f}) on image #{cnn_top1_img} "
                f"({cnn_top1_hz:+.0f} Hz)."
            )

        # Case F: Narrow Passband plateau — metrics within ±25 Hz
        else:
            agreement_status = "passband_plateau"
            requires_arbitration = False
            confidence_score = 0.85
            confidence_label = "high"
            confidence_basis = (
                f"Metrics align within clinical tolerance ({delta_cnn_phys_hz:.0f} Hz gap <= 25 Hz). "
                f"CNN=#{cnn_top1_img}, Physics=#{physics_top1_img}."
            )

        is_reconciled = not requires_arbitration

        return ConsistencyReport(
            patient_id=store.patient_id,
            cnn_top1_img=cnn_top1_img,
            cnn_top1_hz=cnn_top1_hz,
            physics_top1_img=physics_top1_img,
            physics_top1_hz=physics_top1_hz,
            homog_top1_img=homog_top1_img,
            homog_top1_hz=homog_top1_hz,
            agreement_status=agreement_status,
            is_reconciled=is_reconciled,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
            confidence_basis=confidence_basis,
            requires_arbitration=requires_arbitration,
            candidate_images=sorted(candidate_images),
            disagreements=disagreements,
        )
