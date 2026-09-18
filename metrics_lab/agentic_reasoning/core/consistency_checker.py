"""
Consistency Checker (ADRE-style multi-source agreement reconciliation)
-----------------------------------------------------------------------
Cross-examines decisions from:
1. Deep Learning Classifier (EfficientNet-B0 LOOCV)
2. Quantitative Physics / Artifact Detector (argmin banded_fraction_myo)
3. Signal Uniformity Metric (argmax homogeneity = 1 - CoV)
4. Passband Plateau Model

Reconciles agreements, detects hard conflicts (e.g. CNN selecting an offset with severe banding),
and issues calibrated confidence ratings.
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
    agreement_status: str       # "full_consensus" | "passband_plateau" | "physics_veto" | "divergent"
    is_reconciled: bool
    confidence_score: float     # 0.0 to 1.0
    confidence_label: str       # "high" | "medium" | "low"
    confidence_basis: str       # Human-readable justification of confidence
    disagreements: List[str] = field(default_factory=list)


class ConsistencyChecker:

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

        # Analyze alignment
        cnn_bfm = bfm_map.get(cnn_top1_img, 0.0)
        delta_cnn_phys_hz = abs(cnn_top1_hz - physics_top1_hz)

        # Check for hard veto condition: CNN chose an offset with significant banding
        if cnn_bfm > 0.08:
            agreement_status = "physics_veto"
            disagreements.append(
                f"CNN selected image #{cnn_top1_img} ({cnn_top1_hz:+.0f} Hz), but it suffers from severe "
                f"myocardial banding (banded fraction = {cnn_bfm:.3f}). Overruled by physics detector."
            )
            confidence_score = 0.65
            confidence_label = "medium"
            confidence_basis = "Physics veto applied: CNN favored an offset with dark banding in myocardium."

        elif cnn_top1_img == physics_top1_img and cnn_top1_img == homog_top1_img:
            agreement_status = "full_consensus"
            confidence_score = 0.92
            confidence_label = "high"
            confidence_basis = "Full consensus: CNN, intra-myocardial banding, and signal homogeneity all independently select the identical offset."

        elif delta_cnn_phys_hz <= 25.0:
            agreement_status = "passband_plateau"
            confidence_score = 0.82
            confidence_label = "high"
            confidence_basis = f"Passband plateau consensus: CNN (#{cnn_top1_img}) and Physics (#{physics_top1_img}) agree within clinical tolerance window (±25 Hz)."

        else:
            agreement_status = "divergent"
            disagreements.append(
                f"CNN favors #{cnn_top1_img} ({cnn_top1_hz:+.0f} Hz), Physics favors #{physics_top1_img} "
                f"({physics_top1_hz:+.0f} Hz), and Homogeneity favors #{homog_top1_img} ({homog_top1_hz:+.0f} Hz)."
            )
            confidence_score = 0.48
            confidence_label = "low"
            confidence_basis = "Divergent evidence: CNN and quantitative artifact metrics prefer distinct frequency zones."

        return ConsistencyReport(
            patient_id=store.patient_id,
            cnn_top1_img=cnn_top1_img,
            cnn_top1_hz=cnn_top1_hz,
            physics_top1_img=physics_top1_img,
            physics_top1_hz=physics_top1_hz,
            homog_top1_img=homog_top1_img,
            homog_top1_hz=homog_top1_hz,
            agreement_status=agreement_status,
            is_reconciled=agreement_status != "divergent",
            confidence_score=round(confidence_score, 2),
            confidence_label=confidence_label,
            confidence_basis=confidence_basis,
            disagreements=disagreements
        )
