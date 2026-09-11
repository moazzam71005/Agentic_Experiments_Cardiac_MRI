"""
gt_loader.py
------------
Load ground-truth selections from the Excel file.

GT file: 1stcohortSAX (1).xlsx
  Row 1  = header: Anonymized ID, Plane, Max Freq, Freq Step, Min Freq,
                   Manual frequency (Hz), image no
  image no = 1-indexed image position in the sorted DICOM stack.

Returns
-------
    gt_dict : {patient_id: {"image_no": int, "offset_hz": float, "manual_hz": float}}

Usage
-----
    from utils.gt_loader import load_gt
    gt = load_gt("E:/CINE-Test1/1stcohortSAX (1).xlsx")
    print(gt["Patient-BEV"])   # {"image_no": 7, "offset_hz": 0.0, "manual_hz": 0.0}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


_OFFSETS_13 = [-150 + i * 25 for i in range(13)]   # 13-image patients
_OFFSETS_7  = [-150 + i * 50 for i in range(7)]    # Patient-KP only


def _image_no_to_hz(image_no: int, n_images: int) -> float:
    """Convert 1-indexed image_no to Hz for a patient with n_images."""
    step = 300.0 / (n_images - 1)
    return -150.0 + (image_no - 1) * step


def load_gt(
    xlsx_path: str | Path,
    n_images_by_patient: Dict[str, int] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Parse the GT Excel file.

    Parameters
    ----------
    xlsx_path : path to 1stcohortSAX (1).xlsx
    n_images_by_patient : optional dict {patient_id: n}; if provided, used to
        compute offset_hz precisely per patient.  Defaults to 13 for all except
        Patient-KP (7).

    Returns
    -------
    dict: {
        patient_id: {
            "image_no"  : int,        # 1-indexed
            "offset_hz" : float,      # Hz at GT image_no
            "manual_hz" : float,      # Manual frequency from Excel
            "n_images"  : int,        # number of images for that patient
        }
    }
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"GT file not found: {xlsx_path}")

    # The file has 2 blank header rows; row index 1 has the real column names
    df = pd.read_excel(xlsx_path, header=None)

    # Find the header row (contains "Anonymized ID")
    header_row = None
    for i, row in df.iterrows():
        if any("Anonymized ID" in str(v) for v in row.values):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not locate 'Anonymized ID' header in Excel file.")

    df.columns = df.iloc[header_row].tolist()
    df = df.iloc[header_row + 1:].reset_index(drop=True)
    df = df.dropna(subset=["Anonymized ID"]).copy()

    # Normalise column names
    col_map = {}
    for c in df.columns:
        cs = str(c).strip().lower()
        if "anonymized" in cs:
            col_map[c] = "patient_id"
        elif "image no" in cs:
            col_map[c] = "image_no"
        elif "manual" in cs and "freq" in cs:
            col_map[c] = "manual_hz"
        elif "max freq" in cs:
            col_map[c] = "max_freq_hz"
        elif "freq step" in cs:
            col_map[c] = "freq_step_hz"
        elif "min freq" in cs:
            col_map[c] = "min_freq_hz"
    df = df.rename(columns=col_map)

    gt_dict = {}
    for _, row in df.iterrows():
        pid = str(row["patient_id"]).strip()
        image_no = int(row["image_no"])
        manual_hz = float(row.get("manual_hz", 0.0)) if "manual_hz" in row else 0.0

        # Determine n_images for this patient
        if n_images_by_patient and pid in n_images_by_patient:
            n = n_images_by_patient[pid]
        elif pid == "Patient-KP":
            n = 7
        else:
            n = 13

        offset_hz = _image_no_to_hz(image_no, n)

        gt_dict[pid] = {
            "image_no": image_no,
            "offset_hz": offset_hz,
            "manual_hz": manual_hz,
            "n_images": n,
        }

    return gt_dict


def gt_as_dataframe(gt_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Convert GT dict to a tidy DataFrame."""
    rows = []
    for pid, info in gt_dict.items():
        rows.append({"patient_id": pid, **info})
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_predictions(
    pred_dict: Dict[str, int],
    gt_dict: Dict[str, Dict[str, Any]],
    exclude: list | None = None,
) -> Dict[str, Any]:
    """
    Evaluate predicted image_no against GT.

    Parameters
    ----------
    pred_dict : {patient_id: predicted_image_no_1indexed}
    gt_dict   : output of load_gt()
    exclude   : list of patient_ids to skip (e.g. ["Patient-KP"])

    Returns
    -------
    dict with keys: top1, pm1, mae_hz, results_df
    """
    exclude = set(exclude or [])
    rows = []
    for pid, pred_no in pred_dict.items():
        if pid in exclude:
            continue
        if pid not in gt_dict:
            continue
        info = gt_dict[pid]
        gt_no = info["image_no"]
        n = info["n_images"]
        step = 300.0 / (n - 1)

        correct = int(pred_no == gt_no)
        pm1     = int(abs(pred_no - gt_no) <= 1)
        mae_hz  = abs(pred_no - gt_no) * step

        rows.append({
            "patient_id": pid,
            "gt_no": gt_no,
            "pred_no": pred_no,
            "correct": correct,
            "pm1": pm1,
            "mae_hz": mae_hz,
            "gt_hz": info["offset_hz"],
            "pred_hz": _image_no_to_hz(pred_no, n),
        })

    df = pd.DataFrame(rows)
    n_valid = len(df)
    top1  = df["correct"].sum() / n_valid if n_valid else 0.0
    pm1   = df["pm1"].sum()     / n_valid if n_valid else 0.0
    mae   = df["mae_hz"].mean()  if n_valid else 0.0

    return {
        "top1": top1,
        "pm1": pm1,
        "mae_hz": mae,
        "n_valid": n_valid,
        "results_df": df,
    }
