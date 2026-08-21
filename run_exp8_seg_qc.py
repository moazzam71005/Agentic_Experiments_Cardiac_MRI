"""
Exp 8 — Step 0+1: Segmentation QC
----------------------------------
Run the LVSA U-Net (advchain UNet_64) consensus segmenter on all 26 patients
and save colour-overlay PNGs for visual quality assurance.

Usage (from repo root):
    python run_exp8_seg_qc.py

Outputs written to:  exp8_seg_qc/<Patient-XX>/
  - slice_<offset>hz_overlay.png   (MYO=green, LV=blue, RV=yellow, band=semi-transparent)
  - mask_consensus.npy             (H×W uint8 consensus label map)
  - summary.json                   (myo_area_px, lv_area_px, method)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("exp8_seg_qc")

CHECKPOINT = "E:/cardiotune/backend/ml/Cardiac_Multi_view_segmentation/checkpoints/Unet_LVSA_best.pkl"
DATA_ROOT  = "Dr-Dataset"
EXCEL_PATH = "1stcohortSAX (1).xlsx"
OUT_DIR    = "exp8_seg_qc"
DEVICE     = "cpu"   # change to "cuda" if GPU is available


# ---------------------------------------------------------------------------
# Colour palette for overlay: BG=none, RV=yellow, MYO=green, LV=blue
# ---------------------------------------------------------------------------
LABEL_COLOURS = {
    0: None,            # BG — transparent
    1: (255, 200, 0),   # RV  — yellow
    2: (50,  220, 50),  # MYO — green
    3: (50,  100, 255), # LV  — blue
}


def colour_overlay(img_grey: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Return RGB uint8 with mask labels blended over the grey image."""
    if img_grey.max() > 1.0:
        img_grey = img_grey / (img_grey.max() + 1e-8)
    rgb = np.stack([img_grey, img_grey, img_grey], axis=-1)  # H×W×3 float
    overlay = rgb.copy()
    for label, colour in LABEL_COLOURS.items():
        if colour is None:
            continue
        region = mask == label
        if region.any():
            for c, val in enumerate(colour):
                overlay[region, c] = val / 255.0
    blended = (1 - alpha) * rgb + alpha * overlay
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def load_dicom_patient(folder: Path):
    """Load all DICOM images for a patient; return (images_list, offsets_hz_list)."""
    import pydicom, re

    dcm_files = sorted(folder.glob("*.dcm")) or sorted(folder.glob("*.DCM"))
    if not dcm_files:
        dcm_files = sorted(p for p in folder.rglob("*.dcm") if p.parent in (folder, folder.parent))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOMs in {folder}")

    loaded = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f))
            m = re.search(r"img(\d+)", f.name, re.IGNORECASE)
            sort_key = int(m.group(1)) if m else int(getattr(ds, "InstanceNumber", 0) or 0)
            loaded.append((sort_key, f.name, ds))
        except Exception:
            pass

    loaded.sort(key=lambda x: (x[0], x[1]))
    images, offsets = [], []
    step = 25.0
    for rank, (_, _, ds) in enumerate(loaded):
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim > 2:
            arr = arr.mean(axis=0)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        images.append(arr)
        # Hz offset inferred from rank if not stored
        offsets.append((rank - len(loaded) // 2) * step)

    return images, offsets


def discover_patients(data_root: str):
    """Return list of (patient_id, Path) for all Patient-* directories."""
    root = Path(data_root)
    return sorted(
        (p.name, p) for p in root.rglob("Patient-*") if p.is_dir()
    )


def main():
    from PIL import Image as PILImage

    ckpt = Path(CHECKPOINT)
    if not ckpt.exists():
        logger.error("Checkpoint not found: %s", ckpt)
        logger.error("Update CHECKPOINT variable at the top of this script.")
        sys.exit(1)

    logger.info("Loading LVSA U-Net from %s …", ckpt.name)
    from freqscout_agent.tools.seg_advchain.adapter import LVSASegAdapter
    adapter = LVSASegAdapter(str(ckpt), device=DEVICE)

    out_root = Path(OUT_DIR)
    out_root.mkdir(exist_ok=True)

    patients = discover_patients(DATA_ROOT)
    if not patients:
        logger.error("No Patient-* directories found under %s", DATA_ROOT)
        sys.exit(1)

    logger.info("Found %d patient folders.", len(patients))
    summary_all = []

    for pid, folder in patients:
        logger.info("--- %s ---", pid)
        try:
            images, offsets = load_dicom_patient(folder)
        except Exception as exc:
            logger.warning("  Skipping %s: %s", pid, exc)
            continue

        # Run consensus segmentation
        consensus = adapter.segment_consensus(images)
        myo_area  = int((consensus == 2).sum())
        lv_area   = int((consensus == 3).sum())
        rv_area   = int((consensus == 1).sum())

        logger.info("  %d images | MYO=%d px | LV=%d px | RV=%d px",
                    len(images), myo_area, lv_area, rv_area)

        pid_out = out_root / pid
        pid_out.mkdir(exist_ok=True)

        # Save consensus mask
        np.save(str(pid_out / "mask_consensus.npy"), consensus)

        # Save overlays for each offset image
        for i, (img, hz) in enumerate(zip(images, offsets)):
            rgb = colour_overlay(img, consensus)
            fname = pid_out / f"slice_{i:02d}_{int(hz):+d}hz_overlay.png"
            PILImage.fromarray(rgb).save(str(fname))

        rec = {
            "patient_id": pid,
            "n_images": len(images),
            "myo_area_px": myo_area,
            "lv_area_px":  lv_area,
            "rv_area_px":  rv_area,
            "method": "lvsa_unet_consensus",
        }
        (pid_out / "summary.json").write_text(json.dumps(rec, indent=2))
        summary_all.append(rec)

    # Write aggregate summary
    agg = out_root / "all_patients_summary.json"
    agg.write_text(json.dumps(summary_all, indent=2))

    # Print a table
    print("\n=== Segmentation QC Summary ===")
    print(f"{'Patient':20} {'N':>3} {'MYO px':>8} {'LV px':>7} {'RV px':>7} Status")
    for r in summary_all:
        ok = "OK" if r["myo_area_px"] > 200 and r["lv_area_px"] > 100 else "CHECK"
        print(f"{r['patient_id']:20} {r['n_images']:>3} {r['myo_area_px']:>8} {r['lv_area_px']:>7} {r['rv_area_px']:>7} {ok}")
    print(f"\nOverlays saved to: {out_root.resolve()}")
    print("Open a few to visually verify the myocardium ring is correctly identified.")


if __name__ == "__main__":
    main()
