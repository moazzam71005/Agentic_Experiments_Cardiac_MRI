"""
T5 Physics Descriptor
---------------------
Reads the Excel ground-truth file and attaches physics-grounded metadata to each
frequency-scout offset:
  - offset_hz           : RF transmitter offset in Hz (from Min/Step/Max columns)
  - band_period_hz      : expected distance between successive banding nulls (≈ 1/TR)
  - position_in_sweep   : index in the sweep (0 = most negative offset)
  - is_centre           : True if offset ≈ 0 Hz
  - distance_from_centre: |offset_hz| in Hz

Ground truth (eval only):
  - image no            : 1-indexed best-slice index in the scout
  - Manual frequency    : GT offset in Hz
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TOOL_NAME = "physics"


@dataclass
class PhysicsResult:
    patient_id: str
    offsets_hz: List[float]
    band_period_hz: Optional[float]
    gt_best_slice: Optional[int]        # 0-indexed
    gt_offset_hz: Optional[float]
    per_slice: List[Dict]
    provenance: Dict = field(default_factory=dict)


class PhysicsDescriptor:

    TOOL_NAME = TOOL_NAME

    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self._df: Optional[pd.DataFrame] = None
        self._load_excel()

    # ------------------------------------------------------------------

    def run(self, patient_id: str, offsets_hz: List[float]) -> PhysicsResult:
        row = self._get_patient_row(patient_id)
        gt_slice, gt_hz, tr = None, None, None

        # Prefer Excel-derived offset sweep when available
        excel_offsets = self._extract_offsets(row, len(offsets_hz)) if row is not None else None
        if excel_offsets is not None and len(excel_offsets) == len(offsets_hz):
            offsets_hz = excel_offsets

        if row is not None:
            gt_slice = self._extract_gt_slice(row, n_slices=len(offsets_hz))
            # Prefer the Hz of the GT image number (same convention as Exp1–5).
            # Manual frequency is recorded separately and can disagree with image no.
            if gt_slice is not None and 0 <= gt_slice < len(offsets_hz):
                gt_hz = float(offsets_hz[gt_slice])
            else:
                gt_hz = self._extract_gt_hz(row)
            tr = self._extract_tr(row)

        band_period = (1000.0 / tr) if tr else None  # TR in ms → period in Hz

        per_slice = []
        for idx, hz in enumerate(offsets_hz):
            hz_f = float(hz)
            per_slice.append({
                "slice_idx": idx,
                "offset_hz": hz_f,
                "position_in_sweep": idx,
                "is_centre": abs(hz_f) < 13,
                "distance_from_centre_hz": abs(hz_f),
                "band_period_hz": band_period,
            })

        logger.info(
            "Physics[%s]: %d slices, GT_slice=%s, GT_Hz=%s, TR=%s ms",
            patient_id, len(offsets_hz), gt_slice, gt_hz, tr,
        )
        return PhysicsResult(
            patient_id=patient_id,
            offsets_hz=[float(h) for h in offsets_hz],
            band_period_hz=band_period,
            gt_best_slice=gt_slice,
            gt_offset_hz=float(gt_hz) if gt_hz is not None else None,
            per_slice=per_slice,
            provenance={"tool": TOOL_NAME, "excel_path": str(self.excel_path)},
        )

    # ------------------------------------------------------------------
    # Excel parsing
    # ------------------------------------------------------------------

    def _load_excel(self):
        if not self.excel_path.exists():
            logger.warning("Excel file not found: %s", self.excel_path)
            return
        try:
            raw = pd.read_excel(self.excel_path, header=None, dtype=str)
            header_row = 0
            for i, row in raw.iterrows():
                if row.dropna().shape[0] >= 4:
                    header_row = int(i)
                    break
            self._df = pd.read_excel(self.excel_path, header=header_row)
            self._df.columns = [str(c).strip() for c in self._df.columns]
            logger.info("Loaded Excel with %d rows, %d cols.", len(self._df), len(self._df.columns))
        except Exception as exc:
            logger.error("Failed to load Excel: %s", exc)

    def _find_col(self, *keywords: str) -> Optional[str]:
        if self._df is None:
            return None
        for col in self._df.columns:
            cl = col.lower()
            if all(k.lower() in cl for k in keywords):
                return col
        return None

    @staticmethod
    def _norm_pid(value: str) -> str:
        """Normalize IDs so Patient-MS and Patient-MSR never collide."""
        s = str(value).strip().lower()
        if s in ("nan", "none", ""):
            return ""
        s = s.replace("_", "").replace(" ", "")
        if s.startswith("patient-"):
            s = s[len("patient-"):]
        elif s.startswith("patient"):
            s = s[len("patient"):]
        return s.replace("-", "")

    def _get_patient_row(self, patient_id: str) -> Optional[pd.Series]:
        if self._df is None:
            return None
        pid_col = self._find_col("anonymized") or self._find_col("patient")
        target = self._norm_pid(patient_id)
        if not target:
            return None
        search_cols = [pid_col] if pid_col else list(self._df.columns)
        for col in search_cols:
            if col is None:
                continue
            for _, row in self._df.iterrows():
                cell = self._norm_pid(str(row[col]))
                if cell and cell == target:
                    return row
        logger.debug("Patient %s not found in Excel.", patient_id)
        return None

    def _extract_gt_slice(self, row: pd.Series, n_slices: int) -> Optional[int]:
        """Use 'image no' column (1-indexed) → 0-indexed slice."""
        col = self._find_col("image", "no") or self._find_col("image no")
        if col is None:
            # fallback: columns that literally contain 'image no'
            for c in (self._df.columns if self._df is not None else []):
                if "image" in c.lower() and "no" in c.lower():
                    col = c
                    break
        if col is None:
            return None
        try:
            val = int(float(str(row[col]).strip()))
        except (ValueError, TypeError):
            return None
        # Excel is 1-indexed
        idx = val - 1
        if 0 <= idx < n_slices:
            return idx
        logger.warning("GT image no=%s out of range for n_slices=%d.", val, n_slices)
        return None

    def _extract_gt_hz(self, row: pd.Series) -> Optional[float]:
        col = self._find_col("manual", "frequency") or self._find_col("manual frequency")
        if col is None:
            return None
        try:
            return float(str(row[col]).strip())
        except (ValueError, TypeError):
            return None

    def _extract_offsets(self, row: pd.Series, n_slices: int) -> Optional[List[float]]:
        """Build offset list from Min Freq / Freq Step / Max Freq columns."""
        min_col = self._find_col("min", "freq")
        step_col = self._find_col("freq", "step") or self._find_col("step")
        max_col = self._find_col("max", "freq")
        if not (min_col and step_col):
            return None
        try:
            fmin = float(str(row[min_col]).strip())
            step = float(str(row[step_col]).strip())
            if step == 0:
                return None
            if max_col is not None:
                fmax = float(str(row[max_col]).strip())
                offs = list(np.arange(fmin, fmax + step * 0.5, step, dtype=float))
            else:
                offs = [fmin + i * step for i in range(n_slices)]
            if len(offs) != n_slices:
                # Truncate or pad to match loaded frames
                if len(offs) > n_slices:
                    offs = offs[:n_slices]
                else:
                    while len(offs) < n_slices:
                        offs.append(offs[-1] + step)
            return [float(x) for x in offs]
        except (ValueError, TypeError):
            return None

    def _extract_tr(self, row: pd.Series) -> Optional[float]:
        for col in self._df.columns if self._df is not None else []:
            cl = col.lower()
            if cl == "tr" or (cl.startswith("tr ") and "time" not in cl):
                try:
                    return float(str(row[col]).strip())
                except (ValueError, TypeError):
                    pass
        return None

    # ------------------------------------------------------------------

    @staticmethod
    def infer_offsets(n_slices: int) -> List[float]:
        """Return standard offset sweep for n_slices scout images."""
        if n_slices == 13:
            return list(np.arange(-150, 175, 25, dtype=float))
        if n_slices == 7:
            return list(np.arange(-150, 200, 50, dtype=float))
        step = 300.0 / max(n_slices - 1, 1)
        return [round(-150.0 + i * step, 1) for i in range(n_slices)]
