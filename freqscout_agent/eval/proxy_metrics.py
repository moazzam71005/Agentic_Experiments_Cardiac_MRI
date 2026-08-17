"""
Proxy Evaluation Metrics (GT we already have: Excel best-slice ground truth)
-----------------------------------------------------------------------------
Computes metrics that can be evaluated NOW (before doctor reasoning GT arrives):

1. Selection accuracy
   - Top-1 accuracy      : recommended offset == GT offset
   - Top-3 accuracy      : GT offset is in the top-3 alternatives or recommendation
   - Within-acceptable   : recommended offset is within ±1 step of GT
   - MAE                 : |recommended_Hz – GT_Hz|

2. Confidence calibration
   - Correct confidence  : mean confidence when Top-1 is correct
   - Wrong confidence    : mean confidence when Top-1 is wrong
   - ECE                 : Expected Calibration Error (bucketed)

3. Faithfulness / hallucination trace
   - n_issues            : total repair issues flagged across all patients
   - hallucination_rate  : fraction of patients with ≥1 hallucination flag

4. Evidence agreement distribution
   - n_full, n_partial, n_conflicted

5. ADRE-style success/failure cases
   - Saved as JSON for qualitative inspection.
"""

from __future__ import annotations

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SinglePatientResult:
    patient_id: str
    gt_slice: Optional[int]
    gt_hz: Optional[float]
    pred_slice: int
    pred_hz: float
    top3_slices: List[int]
    confidence: float
    confidence_label: str
    agreement: str              # "full"|"partial"|"conflicted"
    repair_issues: int
    hallucinated: bool


@dataclass
class ProxyResult:
    n_patients: int
    top1_acc: float
    top3_acc: float
    within1_acc: float          # within ±1 step from GT
    mean_mae_hz: float
    mean_conf_correct: float
    mean_conf_wrong: float
    ece: float
    hallucination_rate: float
    n_full_agree: int
    n_partial_agree: int
    n_conflicted: int
    per_patient: List[SinglePatientResult] = field(default_factory=list)


class ProxyMetrics:

    def compute(self, patient_results: List[SinglePatientResult]) -> ProxyResult:
        """Aggregate per-patient results into summary metrics."""
        n = len(patient_results)
        if n == 0:
            return ProxyResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        evaluated = [r for r in patient_results if r.gt_slice is not None]
        n_eval = len(evaluated)

        # Selection accuracy
        top1_correct = [r for r in evaluated if r.pred_slice == r.gt_slice]
        top3_correct = [r for r in evaluated if r.gt_slice in r.top3_slices]
        within1 = [
            r for r in evaluated
            if r.gt_hz is not None and abs(r.pred_hz - r.gt_hz) <= 26
        ]
        mae_vals = [
            abs(r.pred_hz - r.gt_hz) for r in evaluated if r.gt_hz is not None
        ]

        top1_acc   = len(top1_correct) / max(n_eval, 1)
        top3_acc   = len(top3_correct) / max(n_eval, 1)
        within1_acc= len(within1) / max(n_eval, 1)
        mean_mae   = float(np.mean(mae_vals)) if mae_vals else 0.0

        # Confidence calibration
        conf_correct = [r.confidence for r in evaluated if r.pred_slice == r.gt_slice]
        conf_wrong   = [r.confidence for r in evaluated if r.pred_slice != r.gt_slice]
        mean_conf_c  = float(np.mean(conf_correct)) if conf_correct else 0.0
        mean_conf_w  = float(np.mean(conf_wrong))   if conf_wrong   else 0.0

        ece = self._compute_ece(evaluated)

        # Hallucination
        hall_rate = float(sum(r.hallucinated for r in patient_results) / n)

        # Agreement distribution
        n_full      = sum(1 for r in patient_results if r.agreement == "full")
        n_partial   = sum(1 for r in patient_results if r.agreement == "partial")
        n_conflicted= sum(1 for r in patient_results if r.agreement == "conflicted")

        return ProxyResult(
            n_patients=n,
            top1_acc=top1_acc,
            top3_acc=top3_acc,
            within1_acc=within1_acc,
            mean_mae_hz=mean_mae,
            mean_conf_correct=mean_conf_c,
            mean_conf_wrong=mean_conf_w,
            ece=ece,
            hallucination_rate=hall_rate,
            n_full_agree=n_full,
            n_partial_agree=n_partial,
            n_conflicted=n_conflicted,
            per_patient=patient_results,
        )

    def save_adre_cases(self, result: ProxyResult, output_path: str):
        """Save success/failure case analysis in ADRE Table-5 style."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        evaluated = [r for r in result.per_patient if r.gt_slice is not None]
        successes = [r for r in evaluated if r.pred_slice == r.gt_slice]
        failures  = [r for r in evaluated if r.pred_slice != r.gt_slice]

        def r_to_dict(r):
            return {
                "patient_id": r.patient_id,
                "gt_slice": r.gt_slice,
                "gt_hz": r.gt_hz,
                "pred_slice": r.pred_slice,
                "pred_hz": r.pred_hz,
                "top3_slices": r.top3_slices,
                "confidence": r.confidence,
                "confidence_label": r.confidence_label,
                "agreement": r.agreement,
                "repair_issues": r.repair_issues,
                "hallucinated": r.hallucinated,
            }

        doc = {
            "summary": {
                "n_patients": result.n_patients,
                "top1_acc": result.top1_acc,
                "top3_acc": result.top3_acc,
                "within1_acc": result.within1_acc,
                "mean_mae_hz": result.mean_mae_hz,
                "confidence_calibration": {
                    "correct": result.mean_conf_correct,
                    "wrong": result.mean_conf_wrong,
                    "ece": result.ece,
                },
                "hallucination_rate": result.hallucination_rate,
                "agreement": {
                    "full": result.n_full_agree,
                    "partial": result.n_partial_agree,
                    "conflicted": result.n_conflicted,
                },
            },
            "success_cases": [r_to_dict(r) for r in successes[:5]],
            "failure_cases": [r_to_dict(r) for r in failures[:5]],
        }
        with open(out, "w") as f:
            json.dump(doc, f, indent=2)
        logger.info("ADRE case analysis saved to %s.", out)

    @staticmethod
    def _compute_ece(results: List[SinglePatientResult], n_bins: int = 5) -> float:
        """Expected Calibration Error with equal-width bins on [0, 1]."""
        if not results:
            return 0.0
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(results)
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_bin = [r for r in results if lo <= r.confidence < hi]
            if not in_bin:
                continue
            acc = sum(1 for r in in_bin if r.pred_slice == r.gt_slice) / len(in_bin)
            conf = np.mean([r.confidence for r in in_bin])
            ece += (len(in_bin) / n) * abs(conf - acc)
        return float(ece)

    def print_summary(self, result: ProxyResult):
        print("\n" + "=" * 55)
        print("  AGENTIC FREQUENCY SCOUT – PROXY EVALUATION")
        print("=" * 55)
        print(f"  Patients evaluated  : {result.n_patients}")
        print(f"  Top-1 accuracy      : {result.top1_acc:.3f}  ({result.top1_acc*100:.1f}%)")
        print(f"  Top-3 accuracy      : {result.top3_acc:.3f}  ({result.top3_acc*100:.1f}%)")
        print(f"  Within ±1 step      : {result.within1_acc:.3f}  ({result.within1_acc*100:.1f}%)")
        print(f"  Mean offset MAE     : {result.mean_mae_hz:.1f} Hz")
        print(f"  Conf (correct)      : {result.mean_conf_correct:.3f}")
        print(f"  Conf (wrong)        : {result.mean_conf_wrong:.3f}")
        print(f"  ECE                 : {result.ece:.3f}")
        print(f"  Hallucination rate  : {result.hallucination_rate:.3f}")
        print(f"  Evidence agreement  : full={result.n_full_agree} partial={result.n_partial_agree} conflicted={result.n_conflicted}")
        print("=" * 55)
