"""
dicom_loader.py
---------------
Load a patient's 13-image bSSFP frequency-scout DICOM stack.

Returns:
    images      : list of H×W float32 numpy arrays, normalised 0–1
    offsets_hz  : list of float (frequency offset per image, derived from
                  image index: Hz = -150 + (idx) * step_hz)
    dcm_list    : list of pydicom Dataset objects (for tag inspection)
    patient_id  : str
    step_hz     : float (25 Hz for most patients; 50 Hz for Patient-KP)

Usage
-----
    from utils.dicom_loader import load_patient
    images, offsets_hz, dcms, pid, step = load_patient(
        "E:/CINE-Test1/Dr-Dataset/1st_cohort_SAX/Patient-BEV"
    )
"""

from __future__ import annotations

import os
import re
import glob
import numpy as np
from pathlib import Path
from typing import List, Tuple

try:
    import pydicom
except ImportError:
    raise ImportError("pydicom is required: pip install pydicom")


# ---------------------------------------------------------------------------
# Frequency-offset mapping
# ---------------------------------------------------------------------------

def _image_no_to_hz(image_no_1indexed: int, n_images: int) -> float:
    """
    Map 1-indexed image number to Hz.
    Scout sweep: -150 Hz to +150 Hz.
    step = 300 / (n_images - 1)
    """
    step = 300.0 / (n_images - 1)
    return -150.0 + (image_no_1indexed - 1) * step


def offsets_for_patient(n_images: int) -> List[float]:
    """Return list of Hz offsets for a patient with n_images images."""
    return [_image_no_to_hz(i + 1, n_images) for i in range(n_images)]


# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------

def _sort_key(path: str) -> int:
    """Sort by embedded image number in filename (imgXXXX-...)."""
    m = re.search(r"img(\d+)", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def load_patient(
    patient_dir: str | Path,
    normalise: bool = True,
) -> Tuple[List[np.ndarray], List[float], list, str, float]:
    """
    Load all DICOMs in patient_dir, sorted by image number.

    Parameters
    ----------
    patient_dir : str or Path
        Directory containing the patient's DICOM files.
    normalise : bool
        If True, normalise each image to [0, 1] using its own max.

    Returns
    -------
    images      : list[np.ndarray]  H×W float32, values in [0,1] if normalised
    offsets_hz  : list[float]       frequency offset in Hz per image
    dcm_list    : list[pydicom.Dataset]
    patient_id  : str               directory basename
    step_hz     : float             frequency step between images
    """
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name

    # Collect DICOM files — use set() to deduplicate .dcm/.DCM on Windows
    # (Windows filesystem is case-insensitive, so both globs return the same files)
    dcm_set = set(
        glob.glob(str(patient_dir / "*.dcm"))
        + glob.glob(str(patient_dir / "*.DCM"))
    )
    dcm_paths = sorted(dcm_set, key=_sort_key)

    if not dcm_paths:
        # Fallback: try files without extension
        all_files = [f for f in patient_dir.iterdir() if f.is_file()]
        dcm_paths = sorted([str(f) for f in all_files], key=lambda p: _sort_key(p))

    if not dcm_paths:
        raise FileNotFoundError(f"No DICOM files found in {patient_dir}")

    dcm_list = []
    images = []

    for path in dcm_paths:
        try:
            ds = pydicom.dcmread(path, force=True)
            dcm_list.append(ds)
            arr = ds.pixel_array.astype(np.float32)

            # Handle multi-frame: take first frame
            if arr.ndim == 3:
                arr = arr[0]

            if normalise:
                mx = arr.max()
                arr = arr / mx if mx > 0 else arr

            images.append(arr)
        except Exception as e:
            print(f"  [WARNING] Could not read {path}: {e}")

    n = len(images)
    if n == 0:
        raise RuntimeError(f"No valid images loaded from {patient_dir}")

    offsets_hz = offsets_for_patient(n)
    step_hz = offsets_hz[1] - offsets_hz[0] if n > 1 else 25.0

    return images, offsets_hz, dcm_list, patient_id, step_hz


def load_all_patients(
    data_root: str | Path,
    normalise: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Load all patients from data_root.

    Returns
    -------
    dict: {patient_id: {"images": [...], "offsets_hz": [...],
                         "dcm_list": [...], "step_hz": float}}
    """
    data_root = Path(data_root)
    patient_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    all_data = {}
    for d in patient_dirs:
        try:
            images, offsets_hz, dcm_list, pid, step_hz = load_patient(d, normalise)
            all_data[pid] = {
                "images": images,
                "offsets_hz": offsets_hz,
                "dcm_list": dcm_list,
                "step_hz": step_hz,
            }
            if verbose:
                print(f"  Loaded {pid}: {len(images)} images, step={step_hz:.0f} Hz")
        except Exception as e:
            print(f"  [ERROR] {d.name}: {e}")

    return all_data
