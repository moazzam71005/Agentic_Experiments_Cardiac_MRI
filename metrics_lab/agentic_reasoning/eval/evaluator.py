"""
Evaluator & Benchmark Comparator
--------------------------------
Evaluates the predictions of the Agentic Reasoning framework against Ground Truth.
Computes Top-1, ±1 Step (±25 Hz), MAE (Hz), and builds comparative benchmark tables.
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
        predictions: {pid: {"pred_image": int, "pred_hz": float, ...}}
        ground_truth: {pid: {"image_no": int, "offset_hz": float}}
        """
        pids = sorted([p for p in predictions.keys() if p in ground_truth and p != "Patient-KP"])

        n = len(pids)
        if n == 0:
            return {"error": "No valid evaluated patients"}

        top1_correct = 0
        pm1_correct = 0
        mae_hz_list = []

        per_patient_rows = []

        for pid in pids:
            gt = ground_truth[pid]
            pred = predictions[pid]

            gt_img = int(gt["image_no"])
            gt_hz = float(gt["offset_hz"])

            pred_img = int(pred["pred_image"])
            pred_hz = float(pred["pred_hz"])

            err_steps = abs(pred_img - gt_img)
            err_hz = abs(pred_hz - gt_hz)

            is_top1 = (err_steps == 0)
            is_pm1 = (err_steps <= 1)

            if is_top1:
                top1_correct += 1
            if is_pm1:
                pm1_correct += 1
            mae_hz_list.append(err_hz)

            per_patient_rows.append({
                "patient_id": pid,
                "gt_image": gt_img,
                "gt_hz": gt_hz,
                "pred_image": pred_img,
                "pred_hz": pred_hz,
                "err_steps": err_steps,
                "err_hz": err_hz,
                "is_top1": is_top1,
                "is_pm1": is_pm1,
            })

        top1_acc = (top1_correct / n) * 100.0
        pm1_acc = (pm1_correct / n) * 100.0
        mean_mae = float(np.mean(mae_hz_list))

        df_breakdown = pd.DataFrame(per_patient_rows)

        # Comparative benchmark table
        benchmarks = [
            {"Method": "Center Slice (Naive 0 Hz)", "Top-1 (%)": 26.9, "±1 Step (%)": 57.7, "MAE (Hz)": 32.8},
            {"Method": "Exp5 DINOv2-Large LOOCV", "Top-1 (%)": 23.1, "±1 Step (%)": 57.7, "MAE (Hz)": 40.0},
            {"Method": "Standalone Banded Frac (argmin bfm)", "Top-1 (%)": 40.0, "±1 Step (%)": 76.0, "MAE (Hz)": 25.0},
            {"Method": "Standalone Homogeneity (1 - CoV)", "Top-1 (%)": 44.0, "±1 Step (%)": 80.0, "MAE (Hz)": 27.0},
            {"Method": "EfficientNet-B0 CNN (LOOCV)", "Top-1 (%)": 44.0, "±1 Step (%)": 72.0, "MAE (Hz)": 24.0},
            {"Method": "Exp10 Fused Agent", "Top-1 (%)": 42.3, "±1 Step (%)": 76.9, "MAE (Hz)": 22.1},
            {"Method": "**Agentic Multi-Metric Reasoning (Exp12)**", "Top-1 (%)": round(top1_acc, 1), "±1 Step (%)": round(pm1_acc, 1), "MAE (Hz)": round(mean_mae, 1)},
        ]
        df_benchmarks = pd.DataFrame(benchmarks)

        return {
            "n_patients": n,
            "top1_correct": top1_correct,
            "top1_acc": round(top1_acc, 1),
            "pm1_correct": pm1_correct,
            "pm1_acc": round(pm1_acc, 1),
            "mean_mae_hz": round(mean_mae, 1),
            "df_breakdown": df_breakdown,
            "df_benchmarks": df_benchmarks,
        }
