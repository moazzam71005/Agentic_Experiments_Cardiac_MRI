"""
Evaluator & Benchmark Comparator (Exp 13)
-----------------------------------------
Extended to also track arbitration-specific metrics:
  - arbitration_rate        : fraction of patients where LLM arbitrator was invoked
  - arbitration_hit_rate    : of arbitrated cases, fraction where arbitration improved vs deterministic
  - arbitration_pass_rate   : fraction of invocations where audit passed
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Any


class Evaluator:

    def evaluate_cohort(
        self,
        predictions: Dict[str, Dict[str, Any]],
        ground_truth: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        predictions: {pid: {
            "pred_image": int, "pred_hz": float,
            "decision_source": str,           # "deterministic" | "llm_arbitrated"
            "deterministic_image": int,       # what composite score would have picked
            "arbitration_invoked": bool,      # was arbitrator called (even if audit failed)?
            "arbitration_passed": bool,       # did arbitrator audit pass?
            ...
        }}
        ground_truth: {pid: {"image_no": int, "offset_hz": float}}
        """
        pids = sorted([p for p in predictions.keys() if p in ground_truth and p != "Patient-KP"])

        n = len(pids)
        if n == 0:
            return {"error": "No valid evaluated patients"}

        top1_correct = 0
        pm1_correct  = 0
        mae_hz_list  = []

        # Arbitration tracking
        arb_invoked_count  = 0
        arb_passed_count   = 0
        arb_improved_count = 0   # arbitration result better than deterministic
        arb_same_count     = 0   # same result
        arb_worse_count    = 0   # arbitration result worse than deterministic

        per_patient_rows = []

        for pid in pids:
            gt   = ground_truth[pid]
            pred = predictions[pid]

            gt_img  = int(gt["image_no"])
            gt_hz   = float(gt["offset_hz"])
            pred_img= int(pred["pred_image"])
            pred_hz = float(pred["pred_hz"])

            err_steps = abs(pred_img - gt_img)
            err_hz    = abs(pred_hz - gt_hz)

            is_top1 = (err_steps == 0)
            is_pm1  = (err_steps <= 1)

            if is_top1: top1_correct += 1
            if is_pm1:  pm1_correct  += 1
            mae_hz_list.append(err_hz)

            decision_source   = pred.get("decision_source", "deterministic")
            arb_invoked       = pred.get("arbitration_invoked", False)
            arb_passed        = pred.get("arbitration_passed", False)
            det_img           = pred.get("deterministic_image", pred_img)

            arb_outcome = "N/A"
            if arb_invoked:
                arb_invoked_count += 1
                if arb_passed:
                    arb_passed_count += 1
                    det_err = abs(det_img - gt_img)
                    arb_err = err_steps
                    if arb_err < det_err:
                        arb_improved_count += 1
                        arb_outcome = "improved"
                    elif arb_err == det_err:
                        arb_same_count += 1
                        arb_outcome = "same"
                    else:
                        arb_worse_count += 1
                        arb_outcome = "worse"
                else:
                    arb_outcome = "fallback"

            per_patient_rows.append({
                "patient_id":      pid,
                "gt_image":        gt_img,
                "gt_hz":           gt_hz,
                "pred_image":      pred_img,
                "pred_hz":         pred_hz,
                "err_steps":       err_steps,
                "err_hz":          err_hz,
                "is_top1":         is_top1,
                "is_pm1":          is_pm1,
                "decision_source": decision_source,
                "arb_invoked":     arb_invoked,
                "arb_passed":      arb_passed,
                "arb_outcome":     arb_outcome,
            })

        top1_acc  = (top1_correct / n) * 100.0
        pm1_acc   = (pm1_correct  / n) * 100.0
        mean_mae  = float(np.mean(mae_hz_list))

        df_breakdown = pd.DataFrame(per_patient_rows)

        # Comparative benchmark table
        benchmarks = [
            {"Method": "Center Slice (Naive 0 Hz)",            "Top-1 (%)": 26.9, "±1 Step (%)": 57.7, "MAE (Hz)": 32.8},
            {"Method": "Exp5 DINOv2-Large LOOCV",              "Top-1 (%)": 23.1, "±1 Step (%)": 57.7, "MAE (Hz)": 40.0},
            {"Method": "Standalone Banded Frac (argmin BFM)",  "Top-1 (%)": 40.0, "±1 Step (%)": 76.0, "MAE (Hz)": 25.0},
            {"Method": "Standalone Homogeneity (1-CoV)",       "Top-1 (%)": 44.0, "±1 Step (%)": 80.0, "MAE (Hz)": 27.0},
            {"Method": "EfficientNet-B0 CNN (LOOCV)",          "Top-1 (%)": 44.0, "±1 Step (%)": 72.0, "MAE (Hz)": 24.0},
            {"Method": "Exp10 Fused Agent",                    "Top-1 (%)": 42.3, "±1 Step (%)": 76.9, "MAE (Hz)": 22.1},
            {"Method": "Exp12 Agentic Multi-Metric",           "Top-1 (%)": 48.0, "±1 Step (%)": 76.0, "MAE (Hz)": 24.0},
            {"Method": "**Exp13 LLM Arbitration Loop**",
             "Top-1 (%)": round(top1_acc, 1),
             "±1 Step (%)": round(pm1_acc, 1),
             "MAE (Hz)": round(mean_mae, 1)},
        ]
        df_benchmarks = pd.DataFrame(benchmarks)

        # Arbitration summary
        arb_summary = {
            "total_patients":       n,
            "arb_invoked":          arb_invoked_count,
            "arb_invocation_rate":  f"{arb_invoked_count / n * 100:.1f}%",
            "arb_passed":           arb_passed_count,
            "arb_pass_rate":        f"{arb_passed_count / max(arb_invoked_count, 1) * 100:.1f}%",
            "arb_improved":         arb_improved_count,
            "arb_same":             arb_same_count,
            "arb_worse":            arb_worse_count,
            "arb_improvement_rate": f"{arb_improved_count / max(arb_passed_count, 1) * 100:.1f}%",
        }

        return {
            "n_patients":      n,
            "top1_correct":    top1_correct,
            "top1_acc":        round(top1_acc, 1),
            "pm1_correct":     pm1_correct,
            "pm1_acc":         round(pm1_acc, 1),
            "mean_mae_hz":     round(mean_mae, 1),
            "df_breakdown":    df_breakdown,
            "df_benchmarks":   df_benchmarks,
            "arbitration_summary": arb_summary,
        }
