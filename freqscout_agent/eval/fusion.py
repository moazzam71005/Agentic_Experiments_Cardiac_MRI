"""
Leave-one-out fusion weight search (Exp 10)
-------------------------------------------
Fits composite-score weights on n−1 patients and evaluates on the held-out
patient.  Objective is a blend of exact Top-1 and clinical (±1 step) hit.

Features (per slice, already 0–1):
  neural, homog, no_burden, no_bfm, passband, center
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


FEATURE_NAMES = ("neural", "homog", "no_burden", "no_bfm", "passband", "center")


@dataclass
class FusionResult:
    weights: Dict[str, float]
    loocv_top1: float
    loocv_within1: float
    loocv_accept_hit: float
    loocv_mae_hz: float
    per_patient: List[Dict] = field(default_factory=list)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


def _score(feats: np.ndarray, w: np.ndarray, penalty: Optional[np.ndarray] = None) -> np.ndarray:
    s = feats @ w
    if penalty is not None:
        s = s * penalty
    return s


def _eval_one(
    feats: np.ndarray,
    w: np.ndarray,
    gt_slice: int,
    offsets_hz: Sequence[float],
    penalty: Optional[np.ndarray],
    acceptable: Optional[Sequence[int]],
) -> Tuple[int, bool, bool, bool, float]:
    scores = _score(feats, w, penalty)
    pred = int(np.argmax(scores))
    gt_hz = float(offsets_hz[gt_slice]) if 0 <= gt_slice < len(offsets_hz) else 0.0
    pred_hz = float(offsets_hz[pred])
    top1 = pred == gt_slice
    within1 = abs(pred_hz - gt_hz) <= 26.0
    accept_hit = (gt_slice in acceptable) if acceptable is not None else within1
    mae = abs(pred_hz - gt_hz)
    return pred, top1, within1, accept_hit, mae


def grid_search_weights(
    patients: List[Dict],
    step: float = 0.10,
    min_neural: float = 0.20,
) -> FusionResult:
    """
    Exhaustive simplex grid (weights sum to 1, non-negative).

    Each patient dict:
      pid, feats (n×6), gt_slice, offsets_hz, penalty (n,), acceptable (list[int])
    """
    ticks = np.arange(0.0, 1.0 + 1e-9, step)
    best = None
    best_key = (-1.0, -1.0, 999.0)

    for combo in product(ticks, repeat=len(FEATURE_NAMES)):
        if abs(sum(combo) - 1.0) > 1e-9:
            continue
        if combo[0] < min_neural:
            continue
        w = np.array(combo, dtype=float)
        top1s, w1s, accs, maes = [], [], [], []
        for p in patients:
            _, t1, w1, ah, mae = _eval_one(
                p["feats"], w, p["gt_slice"], p["offsets_hz"],
                p.get("penalty"), p.get("acceptable"),
            )
            top1s.append(t1)
            w1s.append(w1)
            accs.append(ah)
            maes.append(mae)
        # Prefer exact Top-1, then clinical ±1, then low MAE. Discourage
        # collapsing onto a single feature (except a strong neural prior).
        n_active = int(np.sum(w > 1e-9))
        score = 2.0 * float(np.mean(top1s)) + 1.0 * float(np.mean(w1s))
        if n_active < 2:
            score -= 0.15
        key = (score, float(np.mean(top1s)), -float(np.mean(maes)))
        if key > best_key:
            best_key = key
            best = (w, top1s, w1s, accs, maes)

    if best is None:
        w = np.array([0.40, 0.10, 0.10, 0.20, 0.15, 0.05])
        return FusionResult(
            weights=dict(zip(FEATURE_NAMES, w.tolist())),
            loocv_top1=0.0, loocv_within1=0.0, loocv_accept_hit=0.0, loocv_mae_hz=0.0,
        )

    w, top1s, w1s, accs, maes = best
    return FusionResult(
        weights=dict(zip(FEATURE_NAMES, [round(float(x), 3) for x in w])),
        loocv_top1=float(np.mean(top1s)),
        loocv_within1=float(np.mean(w1s)),
        loocv_accept_hit=float(np.mean(accs)),
        loocv_mae_hz=float(np.mean(maes)),
    )


def loocv_weights(
    patients: List[Dict],
    step: float = 0.10,
    min_neural: float = 0.20,
) -> FusionResult:
    """
    True leave-one-out: for each held-out patient, grid-search on the rest,
    then evaluate the held-out patient with those weights.
    Also reports the weights from a full-data grid search (for deployment).
    """
    n = len(patients)
    rows = []
    for i in range(n):
        train = [p for j, p in enumerate(patients) if j != i]
        hold = patients[i]
        fit = grid_search_weights(train, step=step, min_neural=min_neural)
        w = np.array([fit.weights[k] for k in FEATURE_NAMES], dtype=float)
        pred, t1, w1, ah, mae = _eval_one(
            hold["feats"], w, hold["gt_slice"], hold["offsets_hz"],
            hold.get("penalty"), hold.get("acceptable"),
        )
        rows.append({
            "patient_id": hold["pid"],
            "gt_slice": hold["gt_slice"],
            "pred_slice": pred,
            "top1": t1,
            "within1": w1,
            "accept_hit": ah,
            "mae_hz": mae,
            "weights": fit.weights,
        })

    deploy = grid_search_weights(patients, step=step, min_neural=min_neural)
    return FusionResult(
        weights=deploy.weights,
        loocv_top1=float(np.mean([r["top1"] for r in rows])),
        loocv_within1=float(np.mean([r["within1"] for r in rows])),
        loocv_accept_hit=float(np.mean([r["accept_hit"] for r in rows])),
        loocv_mae_hz=float(np.mean([r["mae_hz"] for r in rows])),
        per_patient=rows,
    )


def build_feature_matrix(
    neural_raw: np.ndarray,
    homog: np.ndarray,
    burden: np.ndarray,
    bfm: np.ndarray,
    passband_prox: np.ndarray,
    offsets_hz: Sequence[float],
    center_sigma_hz: float = 100.0,
) -> np.ndarray:
    """Stack the 6 fusion features (n×6)."""
    nn = _softmax(np.asarray(neural_raw, dtype=float))
    hz = np.asarray(offsets_hz, dtype=float)
    center = np.exp(-0.5 * (hz / center_sigma_hz) ** 2)
    center = center / (center.max() + 1e-9)
    no_burden = 1.0 - np.clip(np.asarray(burden, dtype=float), 0.0, 1.0)
    no_bfm = 1.0 - np.clip(np.asarray(bfm, dtype=float), 0.0, 1.0)
    return np.stack([
        nn,
        np.clip(np.asarray(homog, dtype=float), 0.0, 1.0),
        no_burden,
        no_bfm,
        np.clip(np.asarray(passband_prox, dtype=float), 0.0, 1.0),
        center,
    ], axis=1)
