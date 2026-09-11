"""
seg_wrapper.py
--------------
Thin wrapper around the existing LVSA U-Net segmentation model.

Reuses freqscout_agent/tools/segment.py (SegmentationTool) and
freqscout_agent/tools/seg_advchain/ (the U-Net weights).

The wrapper:
1. Adds the project root to sys.path so imports resolve correctly.
2. Provides a simple `segment_patient()` function.
3. Saves/loads a per-session cache (seg_masks_cache.pkl) to avoid
   re-running the heavy model every time.

Labels:
    0 = Background
    1 = RV
    2 = Myocardium
    3 = LV Blood Pool

Usage
-----
    from utils.seg_wrapper import segment_patient, load_seg_cache

    # Segment one patient
    seg_result = segment_patient(images, patient_id, ckpt_path)

    # Or load all from cache
    cache = load_seg_cache("outputs/seg_masks_cache.pkl")
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Make sure freqscout_agent is importable
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]   # E:/CINE-Test1
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_CKPT_PATHS = [
    _REPO_ROOT / "Unet_LVSA_best.pkl",
    Path("E:/cardiotune/backend/ml/Cardiac_Multi_view_segmentation/checkpoints/Unet_LVSA_best.pkl"),
]

_CACHE_DEFAULT = Path(__file__).parent.parent / "outputs" / "seg_masks_cache.pkl"


# ---------------------------------------------------------------------------
# Import the real tool (lazy, to avoid import errors when tool not needed)
# ---------------------------------------------------------------------------

def _get_seg_tool(ckpt_path: Optional[str | Path] = None):
    """Import and instantiate SegmentationTool."""
    from freqscout_agent.tools.segment import SegmentationTool

    if ckpt_path is None:
        for p in _DEFAULT_CKPT_PATHS:
            if p.exists():
                ckpt_path = str(p)
                break
        if ckpt_path is None:
            raise FileNotFoundError(
                "Could not find Unet_LVSA_best.pkl. Pass ckpt_path explicitly."
            )

    return SegmentationTool(checkpoint=str(ckpt_path))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_patient(
    images: List[np.ndarray],
    patient_id: str,
    ckpt_path: Optional[str | Path] = None,
):
    """
    Run LVSA U-Net on all images for one patient.

    Parameters
    ----------
    images      : list of H×W float32 numpy arrays (0–1 normalised)
    patient_id  : str
    ckpt_path   : path to Unet_LVSA_best.pkl (auto-detected if None)

    Returns
    -------
    SegmentationResult (from freqscout_agent.tools.segment)
        .myocardium_mask   : H×W bool
        .lv_pool_mask      : H×W bool
        .rv_mask           : H×W bool
        .label_map         : H×W int8 (0=bg, 1=rv, 2=myo, 3=pool)
        .consensus_mask    : H×W int8 (majority-vote across all offsets)
    """
    tool = _get_seg_tool(ckpt_path)
    seg_result = tool.run(images=images, patient_id=patient_id)
    return seg_result


def segment_all_patients(
    all_data: Dict[str, dict],
    ckpt_path: Optional[str | Path] = None,
    cache_path: Optional[str | Path] = None,
    force_rerun: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Run segmentation for all patients, with cache support.

    Parameters
    ----------
    all_data    : output of dicom_loader.load_all_patients()
    ckpt_path   : U-Net checkpoint path
    cache_path  : where to save/load the cache (default: outputs/seg_masks_cache.pkl)
    force_rerun : if True, ignore existing cache
    verbose     : print progress

    Returns
    -------
    dict: {patient_id: SegmentationResult}
    """
    cache_path = Path(cache_path or _CACHE_DEFAULT)

    # Load cache if it exists and force_rerun is False
    if cache_path.exists() and not force_rerun:
        if verbose:
            print(f"[seg_wrapper] Loading segmentation cache from {cache_path}")
        cached_data = load_seg_cache(cache_path)
        if len(cached_data) > 0:
            return cached_data
        elif verbose:
            print(f"[seg_wrapper] Cache at {cache_path} is empty. Forcing rerun...")

    if verbose:
        print("[seg_wrapper] Running U-Net segmentation on all patients ...")

    tool = _get_seg_tool(ckpt_path)
    seg_cache = {}
    for pid, data in all_data.items():
        try:
            seg = tool.run(
                images=data["images"],
                offsets_hz=data["offsets_hz"],
                patient_id=pid,
                source_slice_idx=len(data["images"]) // 2
            )
            seg_cache[pid] = seg
            if verbose:
                myo_px = seg.myocardium_mask.sum()
                print(f"  {pid}: myo_pixels={myo_px}")
        except Exception as e:
            print(f"  [ERROR] {pid}: {e}")

    # Save cache only if we have results
    if len(seg_cache) > 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(seg_cache, f)
        if verbose:
            print(f"[seg_wrapper] Cache saved to {cache_path}")
    else:
        print("[seg_wrapper] WARNING: No patients were successfully segmented! Cache was NOT saved.")

    return seg_cache


def load_seg_cache(cache_path: str | Path) -> Dict[str, object]:
    """Load segmentation cache from disk."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Seg cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Mask extraction helpers
# ---------------------------------------------------------------------------

def get_masks(seg_result) -> Dict[str, np.ndarray]:
    """
    Extract boolean masks from a SegmentationResult.

    Returns
    -------
    dict with keys: "myo", "pool", "rv", "cardiac" (myo + pool + rv)
    """
    myo  = seg_result.myocardium_mask.astype(bool)
    pool = seg_result.lv_pool_mask.astype(bool)
    rv   = seg_result.rv_mask.astype(bool) if hasattr(seg_result, "rv_mask") else np.zeros_like(myo)
    cardiac = myo | pool | rv
    return {"myo": myo, "pool": pool, "rv": rv, "cardiac": cardiac}
