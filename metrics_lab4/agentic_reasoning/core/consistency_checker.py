"""
Consistency Checker (Exp 15 — VLM Visual Scouting Gatekeeper)
--------------------------------------------------------------
Determines when visual inspection by the VLM is necessary:
  1. wide_plateau: Multiple frames in passband plateau -> VLM visual confirmation
  2. borderline_veto: Borderline banding -> VLM visual inspection of myocardial wall
  3. divergent: Neural CNN vs Physics disagreement -> VLM arbitration
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
    confidence_score: float
    confidence_label: str
    confidence_basis: str
    requires_vlm_inspection: bool
    candidate_images: List[int] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)


class ConsistencyChecker:

    HARD_VETO_BFM = 0.10
    BORDERLINE_BFM = 0.05
    PASSBAND_BFM_THRESHOLD = 0.035

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

        # 2. Physics Top-1 (argmin BFM; tie-break with highest homogeneity)
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

        # Detect passband plateau
        passband_frames = [
            img for img in sorted(bfm_map.keys())
            if bfm_map.get(img, 1.0) <= self.PASSBAND_BFM_THRESHOLD
        ]
        is_wide_plateau = len(passband_frames) >= 3

        # Decision tree
        if cnn_bfm > self.HARD_VETO_BFM:
            agreement_status = "physics_veto"
            requires_vlm = False
            disagreements.append(
                f"CNN selected image #{cnn_top1_img} with severe banding (BFM={cnn_bfm:.3f} > {self.HARD_VETO_BFM})."
            )
            confidence_score = 0.70
            confidence_label = "medium"
            confidence_basis = "Physics hard-veto applied."

        elif self.BORDERLINE_BFM < cnn_bfm <= self.HARD_VETO_BFM:
            agreement_status = "borderline_veto"
            requires_vlm = True
            disagreements.append(
                f"Borderline banding on CNN selection #{cnn_top1_img} (BFM={cnn_bfm:.3f}). Routed to VLM visual inspection."
            )
            confidence_score = 0.60
            confidence_label = "medium"
            confidence_basis = "Borderline banding conflict -> VLM inspection."

        elif delta_cnn_phys_hz > 25.0:
            agreement_status = "divergent"
            requires_vlm = True
            disagreements.append(
                f"Metric divergence: CNN favours #{cnn_top1_img} ({cnn_top1_hz:+.0f} Hz), "
                f"Physics favours #{physics_top1_img} ({physics_top1_hz:+.0f} Hz). Routed to VLM visual inspection."
            )
            confidence_score = 0.50
            confidence_label = "low"
            confidence_basis = "Severe metric divergence -> VLM inspection."

        elif is_wide_plateau and (cnn_top1_img != physics_top1_img or cnn_top1_img != homog_top1_img):
            agreement_status = "wide_plateau"
            requires_vlm = True
            for pf in passband_frames:
                if pf not in candidate_images:
                    candidate_images.append(pf)
            disagreements.append(
                f"Wide passband plateau across {len(passband_frames)} offsets. Routed to VLM visual inspection."
            )
            confidence_score = 0.65
            confidence_label = "medium"
            confidence_basis = "Wide plateau ambiguity -> VLM inspection."

        elif cnn_top1_img == physics_top1_img == homog_top1_img:
            agreement_status = "full_consensus"
            requires_vlm = False
            confidence_score = 0.95
            confidence_label = "high"
            confidence_basis = f"Full consensus on #{cnn_top1_img} ({cnn_top1_hz:+.0f} Hz)."

        else:
            agreement_status = "passband_plateau"
            requires_vlm = False
            confidence_score = 0.85
            confidence_label = "high"
            confidence_basis = f"Metrics agree within tolerance (CNN=#{cnn_top1_img}, Physics=#{physics_top1_img})."

        is_reconciled = not requires_vlm

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
            requires_vlm_inspection=requires_vlm,
            candidate_images=sorted(candidate_images),
            disagreements=disagreements,
        )
