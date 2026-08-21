"""
T1 ROI Segmenter
----------------
Segments LV blood pool and myocardium on the frequency-scout images and derives
AHA-style broad sectors (anterior / lateral / inferior / septal).

Segmentation strategy (in priority order):
  1. LVSA U-Net (advchain UNet_64) — trained on Singapore cardiac SAX dataset.
     Checkpoint: Unet_LVSA_best.pkl (see models.yaml → segmentation.checkpoint).
     Runs a *consensus mask* across all N frequency-offset images (majority vote),
     so banding artefacts on any single frame cannot corrupt the myocardium mask.
  2. Intensity-based fallback — when no checkpoint is configured or loading fails.

Label convention (0=BG, 1=RV, 2=MYO, 3=LV) is identical in both paths.
The single consensus mask is used for ALL downstream banding/quality metrics.
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    """Output of the segmentation tool."""
    patient_id: str
    source_slice_idx: int                   # which slice the mask was derived from
    source_offset_hz: float
    mask: np.ndarray                        # H×W uint8  0=bg 1=RV 2=myo 3=LV-pool
    lv_centroid: Tuple[float, float]        # (row, col) pixel centroid of LV pool
    sectors: Dict[str, np.ndarray]          # {"anterior","lateral","inferior","septal"} -> bool mask H×W
    provenance: Dict = field(default_factory=dict)

    @property
    def myocardium_mask(self) -> np.ndarray:
        return self.mask == 2

    @property
    def lv_pool_mask(self) -> np.ndarray:
        return self.mask == 3

    @property
    def cardiac_roi_mask(self) -> np.ndarray:
        """Union of myocardium and LV pool."""
        return (self.mask == 2) | (self.mask == 3)


# ---------------------------------------------------------------------------
# Main tool class
# ---------------------------------------------------------------------------

class SegmentationTool:
    """
    Parameters
    ----------
    checkpoint : str | None
        Path to a .pt / nnU-Net checkpoint. None = use intensity fallback.
    input_size : tuple
        (H, W) to resize images to before inference.
    device : str
        'cpu' or 'cuda'.
    """

    TOOL_NAME = "segment"

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        input_size: Tuple[int, int] = (256, 256),
        device: str = "cpu",
    ):
        self.checkpoint = checkpoint
        self.input_size = input_size
        self.device = device
        self._model = None        # LightweightUNet (legacy)
        self._lvsa_adapter = None # LVSASegAdapter  (primary)

        if checkpoint and Path(checkpoint).exists():
            self._load_model(checkpoint)
        else:
            if checkpoint:
                logger.warning(
                    "Checkpoint %s not found – falling back to intensity segmentation.",
                    checkpoint,
                )
            logger.info("SegmentationTool: using intensity-based fallback.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        images: List[np.ndarray],
        offsets_hz: List[float],
        patient_id: str,
        source_slice_idx: int,
    ) -> SegmentationResult:
        """
        Parameters
        ----------
        images : list of H×W float32 arrays (pixel values, already normalised 0-1)
        offsets_hz : list of float, same length as images
        patient_id : str
        source_slice_idx : int  index into `images` of the ranker's top candidate

        Returns
        -------
        SegmentationResult with the mask derived from images[source_slice_idx]
        and sectors ready to propagate to all slices.
        """
        src_img = images[source_slice_idx]

        if self._lvsa_adapter is not None:
            # Primary path: consensus mask across ALL frequency-offset images.
            # All 13 frames are the same anatomy — majority vote makes the mask
            # robust even when heavy banding corrupts individual frames.
            logger.info("Segmentation[%s]: running LVSA U-Net consensus across %d images.",
                        patient_id, len(images))
            mask = self._lvsa_adapter.segment_consensus(images)
        elif self._model is not None:
            mask = self._model_inference(src_img)
        else:
            mask = self._intensity_segment(src_img)

        lv_centroid = self._centroid(mask == 3)
        if lv_centroid is None:
            # If LV pool not found, fall back to image centre
            lv_centroid = (src_img.shape[0] / 2, src_img.shape[1] / 2)
            logger.warning("LV pool not detected for %s – using image centre.", patient_id)

        sectors = self._build_sectors(mask, lv_centroid)

        if self._lvsa_adapter is not None:
            method = "lvsa_unet_consensus"
        elif self._model is not None:
            method = "lightweight_unet"
        else:
            method = "intensity_fallback"

        result = SegmentationResult(
            patient_id=patient_id,
            source_slice_idx=source_slice_idx,
            source_offset_hz=offsets_hz[source_slice_idx],
            mask=mask,
            lv_centroid=lv_centroid,
            sectors=sectors,
            provenance={
                "tool": self.TOOL_NAME,
                "method": method,
                "source_slice_idx": source_slice_idx,
                "source_offset_hz": offsets_hz[source_slice_idx],
                "consensus_n": len(images) if method == "lvsa_unet_consensus" else 1,
            },
        )
        logger.info(
            "Segmentation done for %s (source slice %d @ %.0f Hz, method=%s).",
            patient_id, source_slice_idx, offsets_hz[source_slice_idx],
            result.provenance["method"],
        )
        return result

    # ------------------------------------------------------------------
    # Internal: model-based segmentation
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint: str):
        """
        Try to load the LVSA U-Net (advchain UNet_64).
        Falls back to a legacy LightweightUNet if the file extension is .pt/.pth,
        and then to intensity segmentation if that also fails.
        """
        ckpt = Path(checkpoint)
        # Primary: advchain LVSA U-Net (.pkl or .pth)
        if ckpt.suffix.lower() in (".pkl", ".pth"):
            try:
                from freqscout_agent.tools.seg_advchain.adapter import LVSASegAdapter
                self._lvsa_adapter = LVSASegAdapter(checkpoint, device=self.device)
                logger.info("LVSA U-Net (advchain) loaded from %s.", ckpt.name)
                return
            except Exception as exc:
                logger.warning(
                    "LVSA U-Net load failed (%s) — trying legacy checkpoint.", exc
                )
        # Legacy: freqscout_agent LightweightUNet (.pt)
        try:
            import torch
            from freqscout_agent.tools._unet import LightweightUNet
            model = LightweightUNet(in_channels=1, num_classes=4)
            state = torch.load(checkpoint, map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            model.to(self.device)
            self._model = model
            logger.info("Legacy LightweightUNet loaded from %s.", ckpt.name)
        except Exception as exc:
            logger.warning("Could not load any segmentation model: %s – using fallback.", exc)

    def _model_inference(self, img: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        from torchvision.transforms.functional import resize as tv_resize
        import torchvision.transforms.functional as tvF

        h, w = img.shape[:2]
        th, tw = self.input_size

        # Preprocess
        tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(th, tw), mode="bilinear", align_corners=False)

        with torch.no_grad():
            logits = self._model(tensor.to(self.device))  # 1×4×th×tw
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Resize back
        pred_pil = np.array(
            __import__("PIL").Image.fromarray(pred).resize((w, h), __import__("PIL").Image.NEAREST)
        )
        return pred_pil

    # ------------------------------------------------------------------
    # Internal: intensity-based fallback segmentation
    # ------------------------------------------------------------------

    def _intensity_segment(self, img: np.ndarray) -> np.ndarray:
        """
        Simple threshold + morphology approach for cardiac SAX:
          - Blood pool is the brightest connected region near the centre.
          - Myocardium is the ring surrounding it (dilated minus eroded blood pool).
          - RV is the second-brightest connected region.
        """
        from scipy import ndimage as ndi

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if img.ndim == 3:
            gray = img.mean(axis=2)
        else:
            gray = img.copy()

        # Normalise
        if gray.max() > gray.min():
            gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        H, W = gray.shape
        cy, cx = H // 2, W // 2

        # ---- LV blood pool: high-signal connected region near centre ----
        bright_thresh = np.percentile(gray, 80)
        bright = (gray > bright_thresh).astype(np.uint8)
        labeled, n = ndi.label(bright)
        if n == 0:
            return mask  # give up

        # Score regions by distance from centre (prefer closer ones)
        best_label, best_score = 0, -1
        for lb in range(1, n + 1):
            region = labeled == lb
            area = region.sum()
            if area < 50:
                continue
            rows, cols = np.where(region)
            dist = np.sqrt((rows.mean() - cy) ** 2 + (cols.mean() - cx) ** 2)
            score = area / (1.0 + dist)
            if score > best_score:
                best_score = score
                best_label = lb

        if best_label == 0:
            return mask

        lv_pool = labeled == best_label
        mask[lv_pool] = 3  # LV blood pool

        # ---- Myocardium: ring around LV pool ----
        struct = ndi.generate_binary_structure(2, 1)
        myo_thickness = max(3, int(min(H, W) * 0.06))
        dilated = ndi.binary_dilation(lv_pool, structure=struct, iterations=myo_thickness)
        myo = dilated & ~lv_pool
        # Exclude anything already too bright (fat/blood) or too dark (lung)
        myo_signal = gray[myo]
        if myo_signal.size > 0:
            med_myo = np.median(myo_signal)
            myo = myo & (gray > med_myo * 0.4) & (gray < med_myo * 1.8)
        mask[myo] = 2  # myocardium

        # ---- RV: second-brightest connected region (anterior-right of LV) ----
        bright_no_lv = bright.copy()
        bright_no_lv[dilated] = 0
        labeled2, n2 = ndi.label(bright_no_lv)
        if n2 > 0:
            best2, best2_score = 0, -1
            for lb in range(1, n2 + 1):
                region = labeled2 == lb
                area = region.sum()
                if area < 30:
                    continue
                rows, cols = np.where(region)
                dist = np.sqrt((rows.mean() - cy) ** 2 + (cols.mean() - cx) ** 2)
                score = area / (1.0 + dist * 0.5)
                if score > best2_score:
                    best2_score = score
                    best2 = lb
            if best2 > 0:
                mask[labeled2 == best2] = 1  # RV

        return mask

    # ------------------------------------------------------------------
    # Sector generation
    # ------------------------------------------------------------------

    def _build_sectors(
        self,
        mask: np.ndarray,
        lv_centroid: Tuple[float, float],
    ) -> Dict[str, np.ndarray]:
        """
        Divide the myocardium mask into 4 AHA-style sectors using angle from LV centroid.
        Standard cardiac short-axis orientation:
          anterior  = top    (270° – 360°/0° – 90° → angle near 270 or -90)
          lateral   = right  (0° – 90°)
          inferior  = bottom (90° – 180°)
          septal    = left   (180° – 270°)

        We use image-space angles (y increases downward):
          0° = right, 90° = down, 180° = left, 270° = up
        """
        H, W = mask.shape[:2]
        cy, cx = lv_centroid
        myo = mask == 2

        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        dy = rows - cy  # positive downward
        dx = cols - cx  # positive rightward
        angle = np.degrees(np.arctan2(dy, dx)) % 360  # 0–360

        sectors = {
            "anterior": myo & ((angle >= 270) | (angle < 45)),   # top (315±45)
            "lateral":  myo & (angle >= 315) | myo & (angle < 45) | myo & ((angle >= 45) & (angle < 135)),
            "inferior": myo & ((angle >= 90) & (angle < 180)),    # bottom
            "septal":   myo & ((angle >= 180) & (angle < 270)),   # left
        }
        # Simplify: clean up using quadrant-based partition (avoids overlap)
        sectors = {
            "anterior": myo & (((angle >= 315) | (angle < 45))),
            "lateral":  myo & ((angle >= 45) & (angle < 135)),
            "inferior": myo & ((angle >= 135) & (angle < 225)),
            "septal":   myo & ((angle >= 225) & (angle < 315)),
        }
        return sectors

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _centroid(binary_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        rows, cols = np.where(binary_mask)
        if rows.size == 0:
            return None
        return float(rows.mean()), float(cols.mean())
