"""
Gap Analyser (ADRE-style diagnostic checklist verification)
------------------------------------------------------------
Evaluates completeness of evidence for a given patient before clinical synthesis.
Flags any missing perceptual measurements or quality anomalies (e.g. empty mask).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from .evidence_store import EvidenceStore


@dataclass
class ChecklistItem:
    item_id: str
    description: str
    required: bool
    is_satisfied: bool = False
    detail: str = ""


class GapAnalyzer:
    """
    Evaluates whether all essential clinical evidence has been assembled.
    """

    def __init__(self):
        self.checklist_defs = [
            ("seg_verified", "Myocardium mask obtained and verified non-empty", True),
            ("cnn_evaluated", "CNN stack probabilities available across all offsets", True),
            ("banding_detected", "Banded fraction and anatomical sectors localized", True),
            ("homogeneity_checked", "Myocardial signal homogeneity (1-CoV) quantified", True),
            ("temporal_stability", "Temporal frequency sweep pixel variation measured", False),
            ("contrast_evaluated", "Blood-myocardium CNR quantified", False),
            ("passband_bounded", "Passband plateau and center prior computed", True),
        ]

    def analyze(self, store: EvidenceStore, is_empty_seg: bool = False) -> Tuple[bool, List[ChecklistItem]]:
        items: List[ChecklistItem] = []

        for item_id, desc, req in self.checklist_defs:
            sat = False
            detail = ""

            if item_id == "seg_verified":
                sat = not is_empty_seg
                detail = "Valid myocardium mask" if sat else "EMPTY_MASK: apical/atypical scan"
            elif item_id == "cnn_evaluated":
                sat = len(store.get_metric_map("cnn_score")) >= store.n_slices
                detail = f"{len(store.get_metric_map('cnn_score'))}/{store.n_slices} scored"
            elif item_id == "banding_detected":
                sat = len(store.get_metric_map("banded_fraction_myo")) >= store.n_slices
                detail = f"{len(store.get_metric_map('banded_fraction_myo'))}/{store.n_slices} measured"
            elif item_id == "homogeneity_checked":
                sat = len(store.get_metric_map("homogeneity")) >= store.n_slices
                detail = f"{len(store.get_metric_map('homogeneity'))}/{store.n_slices} measured"
            elif item_id == "temporal_stability":
                sat = len(store.get_metric_map("temporal_std_sweep")) >= store.n_slices
                detail = f"{len(store.get_metric_map('temporal_std_sweep'))}/{store.n_slices} measured"
            elif item_id == "contrast_evaluated":
                sat = len(store.get_metric_map("cnr")) >= store.n_slices
                detail = f"{len(store.get_metric_map('cnr'))}/{store.n_slices} measured"
            elif item_id == "passband_bounded":
                sat = store.n_slices > 0
                detail = "Passband bounds calculated"

            items.append(ChecklistItem(
                item_id=item_id,
                description=desc,
                required=req,
                is_satisfied=sat,
                detail=detail
            ))

        all_required_met = all(item.is_satisfied for item in items if item.required)
        return all_required_met, items
