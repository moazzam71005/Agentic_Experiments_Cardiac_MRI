"""
Evaluator & Benchmark Comparator (Exp 14 — ADRE-Debate)
--------------------------------------------------------
Tracks cohort accuracy, clinical window metrics, and multi-agent debate statistics.
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
            "decision_source": str,           # "deterministic" | "adre_debate"
            "deterministic_image": int,       # what composite score would have picked
            "debate_invoked": bool,           # was multi-agent debate called?
            "debate_passed": bool,            # did debate audit pass?
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

        # Debate tracking
        debate_invoked_count  = 0
        debate_passed_count   = 0
        debate_improved_count = 0   # debate result better than deterministic
        debate_same_count     = 0   # same result
        debate_worse_count    = 0   # debate result worse than deterministic

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

            decision_source = pred.get("decision_source", "deterministic")
            debate_invoked  = pred.get("debate_invoked", False)
            debate_passed   = pred.get("debate_passed", False)
            det_img         = pred.get("deterministic_image", pred_img)

            debate_outcome = "N/A"
            if debate_invoked:
                debate_invoked_count += 1
                if debate_passed:
                    debate_passed_count += 1
                    det_err = abs(det_img - gt_img)
                    deb_err = err_steps
                    if deb_err < det_err:
                        debate_improved_count += 1
                        debate_outcome = "improved"
                    elif deb_err == det_err:
                        debate_same_count += 1
                        debate_outcome = "same"
                    else:
                        debate_worse_count += 1
                        debate_outcome = "worse"
                else:
                    debate_outcome = "fallback"

            per_patient_rows.append({
                "patient_id":      pid,
                "gt_image":        gt_img,
                "gt_hz":           gt_hz,
                "pred_image":      pred_img,
                "pred_hz":         pred_hz,
                "error_hz":        err_hz,
                "error_steps":     err_steps,
                "is_top1":         is_top1,
                "is_pm1":          is_pm1,
                "decision_source": decision_source,
                "debate_invoked":  debate_invoked,
                "debate_passed":   debate_passed,
                "debate_outcome":  debate_outcome,
                "status":          pred.get("status", "unknown"),
                "confidence":      pred.get("confidence", 0.0),
            })

        df = pd.DataFrame(per_patient_rows)

        top1_acc = top1_correct / n
        pm1_acc  = pm1_correct  / n
        mae_hz   = float(np.mean(mae_hz_list))

        debate_rate = debate_invoked_count / n if n > 0 else 0.0
        debate_pass_rate = debate_passed_count / debate_invoked_count if debate_invoked_count > 0 else 0.0
        debate_improve_rate = debate_improved_count / debate_passed_count if debate_passed_count > 0 else 0.0

        return {
            "n_patients":            n,
            "top1_correct":          top1_correct,
            "top1_accuracy":         round(top1_acc, 4),
            "pm1_correct":           pm1_correct,
            "pm1_accuracy":          round(pm1_acc, 4),
            "mae_hz":                round(mae_hz, 2),
            "debate_invoked_count":  debate_invoked_count,
            "debate_passed_count":   debate_passed_count,
            "debate_improved_count": debate_improved_count,
            "debate_same_count":     debate_same_count,
            "debate_worse_count":    debate_worse_count,
            "debate_rate":           round(debate_rate, 4),
            "debate_pass_rate":      round(debate_pass_rate, 4),
            "debate_improve_rate":   round(debate_improve_rate, 4),
            "per_patient_df":        df,
        }

    def print_benchmark_table(self, exp14_results: Dict[str, Any]) -> str:
        lines = [
            "=" * 70,
            "      ADRE MULTI-AGENT DEBATE EXPERIMENT RESULTS  (EXP 14)",
            "=" * 70,
            f"Evaluated Patients                 : {exp14_results['n_patients']}",
            f"Top-1 Exact Accuracy               : {exp14_results['top1_correct']}/{exp14_results['n_patients']} = {exp14_results['top1_accuracy']:.1%}",
            f"Clinical Window (±1 Step / ±25 Hz) : {exp14_results['pm1_correct']}/{exp14_results['n_patients']} = {exp14_results['pm1_accuracy']:.1%}",
            f"Mean Absolute Error (MAE)          : {exp14_results['mae_hz']:.1f} Hz",
            "-" * 70,
            "ADRE Multi-Agent Debate Statistics:",
            f"  Invocation Rate                  : {exp14_results['debate_rate']:.1%} ({exp14_results['debate_invoked_count']}/{exp14_results['n_patients']} patients)",
            f"  Debate Audit Pass Rate           : {exp14_results['debate_pass_rate']:.1%}",
            f"  Improvement Rate (of passed)     : {exp14_results['debate_improve_rate']:.1%}",
            f"    Improved : {exp14_results['debate_improved_count']} | Same: {exp14_results['debate_same_count']} | Worse: {exp14_results['debate_worse_count']}",
            "-" * 70,
            "Comparative Benchmarks:",
            f"                             Method  Top-1 (%)  ±1 Step (%)  MAE (Hz)",
            f"          Center Slice (Naive 0 Hz)       26.9         57.7      32.8",
            f"            Exp5 DINOv2-Large LOOCV       23.1         57.7      40.0",
            f"Standalone Banded Frac (argmin BFM)       40.0         76.0      25.0",
            f"     Standalone Homogeneity (1-CoV)       44.0         80.0      27.0",
            f"        EfficientNet-B0 CNN (LOOCV)       44.0         72.0      24.0",
            f"                  Exp10 Fused Agent       42.3         76.9      22.1",
            f"         Exp12 Agentic Multi-Metric       48.0         76.0      24.0",
            f"          Exp13 Gated LLM Arbitrate       52.0         80.0      21.0",
            f"     **Exp14 ADRE Multi-Agent Debate**     {exp14_results['top1_accuracy']*100:.1f}         {exp14_results['pm1_accuracy']*100:.1f}      {exp14_results['mae_hz']:.1f}",
            "=" * 70,
        ]
        return "\n".join(lines)
