"""
Evaluator & Benchmark Comparator (Exp 15 — Multimodal VLM)
------------------------------------------------------------
Tracks cohort accuracy, clinical window metrics, and VLM visual inspection statistics.
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
        pids = sorted([p for p in predictions.keys() if p in ground_truth and p != "Patient-KP"])

        n = len(pids)
        if n == 0:
            return {"error": "No valid evaluated patients"}

        top1_correct = 0
        pm1_correct  = 0
        mae_hz_list  = []

        vlm_invoked_count  = 0
        vlm_passed_count   = 0
        vlm_improved_count = 0
        vlm_same_count     = 0
        vlm_worse_count    = 0

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
            vlm_invoked     = pred.get("vlm_invoked", False)
            vlm_passed      = pred.get("vlm_passed", False)
            det_img         = pred.get("deterministic_image", pred_img)

            vlm_outcome = "N/A"
            if vlm_invoked:
                vlm_invoked_count += 1
                if vlm_passed:
                    vlm_passed_count += 1
                    det_err = abs(det_img - gt_img)
                    vlm_err = err_steps
                    if vlm_err < det_err:
                        vlm_improved_count += 1
                        vlm_outcome = "improved"
                    elif vlm_err == det_err:
                        vlm_same_count += 1
                        vlm_outcome = "same"
                    else:
                        vlm_worse_count += 1
                        vlm_outcome = "worse"
                else:
                    vlm_outcome = "fallback"

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
                "vlm_invoked":     vlm_invoked,
                "vlm_passed":      vlm_passed,
                "vlm_outcome":     vlm_outcome,
                "status":          pred.get("status", "unknown"),
                "confidence":      pred.get("confidence", 0.0),
            })

        df = pd.DataFrame(per_patient_rows)

        top1_acc = top1_correct / n
        pm1_acc  = pm1_correct  / n
        mae_hz   = float(np.mean(mae_hz_list))

        vlm_rate = vlm_invoked_count / n if n > 0 else 0.0
        vlm_pass_rate = vlm_passed_count / vlm_invoked_count if vlm_invoked_count > 0 else 0.0
        vlm_improve_rate = vlm_improved_count / vlm_passed_count if vlm_passed_count > 0 else 0.0

        return {
            "n_patients":         n,
            "top1_correct":       top1_correct,
            "top1_accuracy":      round(top1_acc, 4),
            "pm1_correct":        pm1_correct,
            "pm1_accuracy":       round(pm1_acc, 4),
            "mae_hz":             round(mae_hz, 2),
            "vlm_invoked_count":  vlm_invoked_count,
            "vlm_passed_count":   vlm_passed_count,
            "vlm_improved_count": vlm_improved_count,
            "vlm_same_count":     vlm_same_count,
            "vlm_worse_count":    vlm_worse_count,
            "vlm_rate":           round(vlm_rate, 4),
            "vlm_pass_rate":      round(vlm_pass_rate, 4),
            "vlm_improve_rate":   round(vlm_improve_rate, 4),
            "per_patient_df":     df,
        }

    def print_benchmark_table(self, exp15_results: Dict[str, Any]) -> str:
        lines = [
            "=" * 70,
            "     MULTIMODAL VLM VISUAL SCOUTING EXPERIMENT RESULTS  (EXP 15)",
            "=" * 70,
            f"Evaluated Patients                 : {exp15_results['n_patients']}",
            f"Top-1 Exact Accuracy               : {exp15_results['top1_correct']}/{exp15_results['n_patients']} = {exp15_results['top1_accuracy']:.1%}",
            f"Clinical Window (±1 Step / ±25 Hz) : {exp15_results['pm1_correct']}/{exp15_results['n_patients']} = {exp15_results['pm1_accuracy']:.1%}",
            f"Mean Absolute Error (MAE)          : {exp15_results['mae_hz']:.1f} Hz",
            "-" * 70,
            "VLM Visual Inspection Statistics:",
            f"  Invocation Rate                  : {exp15_results['vlm_rate']:.1%} ({exp15_results['vlm_invoked_count']}/{exp15_results['n_patients']} patients)",
            f"  Visual Audit Pass Rate           : {exp15_results['vlm_pass_rate']:.1%}",
            f"  Improvement Rate (of passed)     : {exp15_results['vlm_improve_rate']:.1%}",
            f"    Improved : {exp15_results['vlm_improved_count']} | Same: {exp15_results['vlm_same_count']} | Worse: {exp15_results['vlm_worse_count']}",
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
            f"         Exp14 Tri-Agent Debate 27B       52.0         80.0      22.0",
            f"     **Exp15 Multimodal VLM Agent**        {exp15_results['top1_accuracy']*100:.1f}         {exp15_results['pm1_accuracy']*100:.1f}      {exp15_results['mae_hz']:.1f}",
            "=" * 70,
        ]
        return "\n".join(lines)
