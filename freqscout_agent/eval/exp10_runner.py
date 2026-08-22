"""
Exp10 evaluation engine
-----------------------
Loads GT, Exp5 scores, DICOMs, and consensus masks (from exp8_seg_qc/ if
present, otherwise runs the LVSA U-Net).  Computes per-offset physics
metrics, passband model, fused scores, LOOCV weights, and calibration.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .fusion import FEATURE_NAMES, build_feature_matrix, grid_search_weights, loocv_weights
from .calibration import loocv_calibrate
from ..tools.banding import BandingDetector
from ..tools.quality import QualityMetrics
from ..tools.passband import PassbandEstimator
from ..tools.segment import SegmentationResult

logger = logging.getLogger(__name__)

DEFAULT_OFFSETS = [-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125, 150]


def load_gt(excel_path: str) -> Dict[str, dict]:
    raw = pd.read_excel(excel_path, header=None, dtype=str)
    header_row = 0
    for i, row in raw.iterrows():
        if row.dropna().shape[0] >= 4:
            header_row = int(i)
            break
    df = pd.read_excel(excel_path, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(*keywords):
        for col in df.columns:
            if all(k.lower() in col.lower() for k in keywords):
                return col
        return None

    id_col = find_col("anonymized") or find_col("patient")
    ino_col = find_col("image", "no")
    man_col = find_col("manual")
    gt = {}
    for _, row in df.iterrows():
        pid = str(row.get(id_col, "")).strip()
        if not pid or pid.lower() in ("nan", "none"):
            continue
        if not pid.startswith("Patient-"):
            pid = "Patient-" + pid.lstrip("Patient").lstrip("-")
        ino = row.get(ino_col)
        if ino is None or (hasattr(ino, "__float__") and pd.isna(ino)):
            continue
        gt_slice = int(float(ino)) - 1
        gt_hz = None
        if man_col is not None:
            raw_hz = row.get(man_col)
            try:
                if raw_hz is not None and not pd.isna(raw_hz):
                    gt_hz = float(raw_hz)
            except Exception:
                pass
        gt[pid] = {"gt_slice": gt_slice, "gt_hz": gt_hz}
    return gt


def load_exp5(csv_path: str) -> Dict[str, np.ndarray]:
    exp5 = {}
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        pid = str(row["patient_id"]).strip()
        raw = row["slice_scores"]
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = ast.literal_eval(raw)
        keys = sorted(int(k) for k in parsed)
        exp5[pid] = np.array([parsed[str(k)] for k in keys], dtype=float)
    return exp5


def discover_patients(data_root: str) -> List[Tuple[str, Path]]:
    root = Path(data_root)
    return sorted((p.name, p) for p in root.rglob("Patient-*") if p.is_dir())


def load_dicoms(folder: Path) -> Tuple[List[np.ndarray], List[float]]:
    import pydicom
    import re

    files = sorted(folder.glob("*.dcm")) or sorted(folder.glob("*.DCM"))
    if not files:
        files = sorted(folder.rglob("*.dcm"))
    loaded = []
    for f in files:
        ds = pydicom.dcmread(str(f))
        m = re.search(r"img(\d+)", f.name, re.IGNORECASE)
        key = int(m.group(1)) if m else int(getattr(ds, "InstanceNumber", 0) or 0)
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim > 2:
            arr = arr[0] if arr.shape[0] <= 4 else arr.mean(axis=0)
        if arr.max() > 0:
            arr = arr / arr.max()
        loaded.append((key, arr))
    loaded.sort(key=lambda x: x[0])
    images = [a for _, a in loaded]
    n = len(images)
    if n == 13:
        offsets = list(DEFAULT_OFFSETS)
    else:
        # Linear sweep covering ±150 if possible
        if n > 1:
            offsets = list(np.linspace(-150, 150, n))
        else:
            offsets = [0.0]
    return images, offsets


def _seg_result(pid: str, mask: np.ndarray, offsets: List[float]) -> SegmentationResult:
    return SegmentationResult(
        patient_id=pid,
        source_slice_idx=0,
        source_offset_hz=float(offsets[0]) if offsets else 0.0,
        mask=mask.astype(np.uint8),
        lv_centroid=(mask.shape[0] / 2, mask.shape[1] / 2),
        sectors={},
        provenance={"method": "cached_or_unet"},
    )


def collect_patient_metrics(
    data_root: str = "Dr-Dataset",
    excel_path: str = "1stcohortSAX (1).xlsx",
    exp5_csv: str = "exp5_outputs/exp5_loocv_per_patient.csv",
    qc_dir: str = "exp8_seg_qc",
    checkpoint: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, dict]:
    """Return pid -> metrics dict used by fusion / plots / calibration."""
    gt = load_gt(excel_path)
    exp5 = load_exp5(exp5_csv)
    patients = discover_patients(data_root)
    banding = BandingDetector()
    quality = QualityMetrics()
    estimator = PassbandEstimator()
    adapter = None

    results = {}
    for pid, folder in patients:
        if pid not in gt:
            continue
        images, offsets = load_dicoms(folder)
        n = len(images)
        mask_path = Path(qc_dir) / pid / "mask_consensus.npy"
        if mask_path.exists():
            mask = np.load(mask_path)
        else:
            if adapter is None:
                from ..tools.seg_advchain.adapter import LVSASegAdapter
                ckpt = checkpoint or (
                    "E:/cardiotune/backend/ml/Cardiac_Multi_view_segmentation/"
                    "checkpoints/Unet_LVSA_best.pkl"
                )
                adapter = LVSASegAdapter(ckpt, device=device)
            mask = adapter.segment_consensus(images)
        seg = _seg_result(pid, mask, offsets)
        band_res = banding.run(images, offsets, seg)
        qual_res = quality.run(images, offsets, seg)

        q_data = [{
            "homogeneity": q.homogeneity,
            "cnr": q.cnr,
            "banded_fraction_myo": q.banded_fraction_myo,
            "band_trough_depth": q.band_trough_depth,
            "mu_myo": q.mu_myo,
        } for q in qual_res]
        b_data = [{
            "overlaps_myo": b.bands_in_myocardium,
            "overlaps_pool": b.bands_in_lv_pool,
            "banding_burden": b.banding_burden,
            "location": ", ".join(b.band_sectors) if b.band_sectors else "none",
            "present": b.band_present,
            "severity_float": b.severity,
        } for b in band_res]
        pb = estimator.estimate(offsets, q_data, b_data)

        nn = exp5.get(pid, np.zeros(n))
        if len(nn) < n:
            nn = np.concatenate([nn, np.full(n - len(nn), nn.min() - 1 if len(nn) else 0)])
        nn = nn[:n]
        homog = np.array([q.homogeneity for q in qual_res])
        burden = np.array([b.banding_burden for b in band_res])
        bfm = np.array([q.banded_fraction_myo for q in qual_res])
        feats = build_feature_matrix(
            nn, homog, burden, bfm, np.array(pb.per_offset_proximity), offsets
        )
        penalty = np.where(
            np.array([b.bands_in_myocardium or b.bands_in_lv_pool for b in band_res]),
            0.25, 1.0,
        )
        g = gt[pid]
        gt_slice = int(g["gt_slice"])
        if not (0 <= gt_slice < n):
            gt_slice = min(max(gt_slice, 0), n - 1)
        # Same convention as Exp1–9 / PhysicsDescriptor: GT Hz is the
        # scout-sweep value at `image no`, not the Excel "Manual frequency"
        # column (the two can disagree).
        gt_hz = float(offsets[gt_slice])

        results[pid] = {
            "pid": pid,
            "offsets": offsets,
            "offsets_hz": offsets,
            "gt_slice": gt_slice,
            "gt_hz": gt_hz,
            "feats": feats,
            "penalty": penalty,
            "acceptable": pb.acceptable_set,
            "rejected": pb.rejected_set,
            "passband_peak": pb.passband_peak_idx,
            "reasons": pb.per_offset_reason,
            "passband_quality": pb.passband_quality,
            "mu_myo": np.array([q.mu_myo for q in qual_res]),
            "bfm": bfm,
            "homog": homog,
            "cnr": np.array([q.cnr for q in qual_res]),
            "burden": burden,
            "neural": nn,
            "myo_px": int((mask == 2).sum()),
        }
    return results


def evaluate(results: Dict[str, dict], weights: Optional[Dict[str, float]] = None) -> dict:
    patients = list(results.values())
    default_w = weights or {
        "neural": 0.20, "homog": 0.00, "no_burden": 0.00,
        "no_bfm": 0.70, "passband": 0.00, "center": 0.10,
    }
    w = np.array([default_w[k] for k in FEATURE_NAMES], dtype=float)

    def _pred(p, ww):
        scores = (p["feats"] @ ww) * p["penalty"]
        return int(np.argmax(scores)), scores

    rows = []
    for p in patients:
        pred, scores = _pred(p, w)
        pred_hz = float(p["offsets"][pred])
        gt_hz = float(p["gt_hz"])
        sorted_s = np.sort(scores)[::-1]
        margin = float(sorted_s[0] - sorted_s[1]) if len(sorted_s) > 1 else 0.0
        accept = set(p["acceptable"]) | {pred}
        rows.append({
            "patient_id": p["pid"],
            "gt_slice": p["gt_slice"],
            "gt_hz": gt_hz,
            "pred_slice": pred,
            "pred_hz": pred_hz,
            "passband_peak": p["passband_peak"],
            "neural_pred": int(np.argmax(p["neural"])),
            "bfm_pred": int(np.argmin(p["bfm"])) if p["bfm"].size else 0,
            "top1": pred == p["gt_slice"],
            "within1": abs(pred_hz - gt_hz) <= 26,
            "accept_hit": p["gt_slice"] in accept,
            "mae_hz": abs(pred_hz - gt_hz),
            "margin": margin,
            "myo_px": p["myo_px"],
        })

    df = pd.DataFrame(rows)
    grid = grid_search_weights(patients, step=0.10, min_neural=0.20)
    loocv = loocv_weights(patients, step=0.10, min_neural=0.20)

    # Neural / passband / bfm selectors for ablation
    def _sel(key):
        t1 = w1 = ah = mae = 0
        n = len(patients)
        for p in patients:
            if key == "neural":
                pred = int(np.argmax(p["neural"]))
            elif key == "passband":
                pred = int(p["passband_peak"])
            else:
                pred = int(np.argmin(p["bfm"])) if p["bfm"].size else 0
            pred_hz = float(p["offsets"][pred])
            t1 += pred == p["gt_slice"]
            w1 += abs(pred_hz - p["gt_hz"]) <= 26
            ah += p["gt_slice"] in (set(p["acceptable"]) | {pred})
            mae += abs(pred_hz - p["gt_hz"])
        return {
            "Top-1": f"{t1}/{n} ({t1/n:.1%})",
            "±1 step": f"{w1}/{n} ({w1/n:.1%})",
            "Accept-hit": f"{ah}/{n} ({ah/n:.1%})",
            "MAE (Hz)": f"{mae/n:.1f}",
        }

    n = len(df)
    fused_row = {
        "Top-1": f"{df.top1.sum()}/{n} ({df.top1.mean():.1%})",
        "±1 step": f"{df.within1.sum()}/{n} ({df.within1.mean():.1%})",
        "Accept-hit": f"{df.accept_hit.sum()}/{n} ({df.accept_hit.mean():.1%})",
        "MAE (Hz)": f"{df.mae_hz.mean():.1f}",
    }
    loocv_row = {
        "Top-1": f"{loocv.loocv_top1:.1%}",
        "±1 step": f"{loocv.loocv_within1:.1%}",
        "Accept-hit": f"{loocv.loocv_accept_hit:.1%}",
        "MAE (Hz)": f"{loocv.loocv_mae_hz:.1f}",
    }

    margins = [r["margin"] / 0.08 for r in rows]
    correct = [r["top1"] for r in rows]
    cal_iso = loocv_calibrate(margins, correct, method="isotonic")
    cal_platt = loocv_calibrate(margins, correct, method="platt")

    # Miss categorization
    misses = []
    for r, p in zip(rows, patients):
        if r["top1"]:
            continue
        d = abs(r["pred_hz"] - r["gt_hz"])
        if p["myo_px"] < 100:
            cat = "segmentation_failure"
        elif d <= 26:
            cat = "clinical_neighbor"
        elif abs(int(p["passband_peak"]) - p["gt_slice"]) <= 1:
            cat = "neural_error_passband_ok"
        else:
            cat = "flat_or_ambiguous_passband"
        misses.append({**r, "category": cat, "reason_pred": p["reasons"][r["pred_slice"]],
                       "reason_gt": p["reasons"][p["gt_slice"]] if 0 <= p["gt_slice"] < len(p["reasons"]) else ""})

    return {
        "df": df,
        "weights_default": default_w,
        "weights_grid": grid.weights,
        "weights_loocv": loocv.weights,
        "ablation": {
            "neural_only": _sel("neural"),
            "bfm_only (argmin banded_frac)": _sel("bfm"),
            "passband_peak": _sel("passband"),
            "fused Exp10 (default w)": fused_row,
            "fused Exp10 (LOOCV w)": loocv_row,
        },
        "calibration_isotonic": cal_iso,
        "calibration_platt": cal_platt,
        "misses": misses,
        "loocv_detail": loocv.per_patient,
        "results": results,
    }
