"""
Post-hoc confidence calibration (Exp 10)
----------------------------------------
Fits Platt (logistic) or isotonic regression on composite-score margins,
leave-one-out so the held-out patient is never in the fit.

Target: ECE < 0.15 with meaningful high / medium / low separation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CalibrationReport:
    method: str
    ece_before: float
    ece_after: float
    mean_conf_correct_before: float
    mean_conf_wrong_before: float
    mean_conf_correct_after: float
    mean_conf_wrong_after: float
    per_patient: List[dict] = field(default_factory=list)


def ece(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 5) -> float:
    conf = np.asarray(confidences, dtype=float)
    y = np.asarray(correct, dtype=bool)
    if conf.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        acc = float(y[mask].mean())
        c = float(conf[mask].mean())
        total += (mask.sum() / n) * abs(c - acc)
    return float(total)


def _platt_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Newton-style logistic fit: p = 1 / (1 + exp(-(a*x + b)))."""
    a, b = 1.0, 0.0
    yf = y.astype(float)
    for _ in range(40):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))
        err = p - yf
        # gradients
        da = float(np.dot(err, x) / len(x))
        db = float(err.mean())
        a -= 0.8 * da
        b -= 0.8 * db
    return a, b


def _platt_predict(x: np.ndarray, a: float, b: float) -> np.ndarray:
    z = a * x + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def _isotonic_fit(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """PAV isotonic regression. Returns (sorted_x, fitted_y)."""
    order = np.argsort(x)
    xs = x[order]
    ys = y[order].astype(float)
    n = len(ys)
    if n == 0:
        return xs, ys
    # Pool Adjacent Violators
    blocks = [[ys[i], 1] for i in range(n)]  # [sum, count]
    i = 0
    while i < len(blocks) - 1:
        m_i = blocks[i][0] / blocks[i][1]
        m_j = blocks[i + 1][0] / blocks[i + 1][1]
        if m_i > m_j + 1e-12:
            blocks[i][0] += blocks[i + 1][0]
            blocks[i][1] += blocks[i + 1][1]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted = []
    for s, c in blocks:
        fitted.extend([s / c] * c)
    return xs, np.array(fitted)


def _isotonic_predict(x: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if len(xs) == 0:
        return np.full_like(x, 0.5, dtype=float)
    return np.interp(x, xs, ys, left=ys[0], right=ys[-1])


def label_from_conf(c: float) -> str:
    if c >= 0.65:
        return "high"
    if c >= 0.35:
        return "medium"
    return "low"


def loocv_calibrate(
    margins: Sequence[float],
    correct: Sequence[bool],
    method: str = "isotonic",
) -> CalibrationReport:
    """
    Leave-one-out calibration on (margin, was_correct) pairs.

    ``margins`` should already be in [0, 1] (e.g. composite-margin / 0.08 clipped).
    """
    x = np.asarray(margins, dtype=float)
    y = np.asarray(correct, dtype=bool)
    n = len(x)
    raw = np.clip(x, 0.05, 1.0)
    cal = np.zeros(n)

    for i in range(n):
        xt, yt = np.delete(x, i), np.delete(y, i).astype(float)
        if method == "platt":
            a, b = _platt_fit(xt, yt)
            cal[i] = float(_platt_predict(np.array([x[i]]), a, b)[0])
        else:
            xs, ys = _isotonic_fit(xt, yt)
            cal[i] = float(_isotonic_predict(np.array([x[i]]), xs, ys)[0])
    cal = np.clip(cal, 0.05, 0.99)

    def _means(conf):
        cc = [c for c, ok in zip(conf, y) if ok]
        cw = [c for c, ok in zip(conf, y) if not ok]
        return (
            float(np.mean(cc)) if cc else 0.0,
            float(np.mean(cw)) if cw else 0.0,
        )

    mc_b, mw_b = _means(raw)
    mc_a, mw_a = _means(cal)
    rows = [
        {"idx": i, "raw": float(raw[i]), "calibrated": float(cal[i]),
         "correct": bool(y[i]), "label": label_from_conf(float(cal[i]))}
        for i in range(n)
    ]
    return CalibrationReport(
        method=method,
        ece_before=ece(raw, y),
        ece_after=ece(cal, y),
        mean_conf_correct_before=mc_b,
        mean_conf_wrong_before=mw_b,
        mean_conf_correct_after=mc_a,
        mean_conf_wrong_after=mw_a,
        per_patient=rows,
    )
