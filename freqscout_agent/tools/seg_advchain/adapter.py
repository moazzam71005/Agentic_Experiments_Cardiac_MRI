"""
LVSA U-Net Adapter
------------------
Wraps the vendored UNet_64 checkpoint for direct 2D numpy inference —
no SimpleITK, no NIfTI, no subprocess required.

Preprocessing (matches advchain training):
  - Convert to float32, normalise to [0,1] (rescale to full range)
  - Z-score normalise (zero mean, unit std) per slice
  - Centre-crop / pad to 256×256
  - Run UNet_64 forward pass
  - Unpad back to original H×W

Consensus mask:
  - Run on all N frequency-offset images (same anatomy, different banding)
  - Take per-pixel majority-vote label — robust against banding artefacts
    corrupting any single offset.

Label convention (identical to freqscout_agent SegmentationResult):
  0 = Background
  1 = Right Ventricle  (RV)
  2 = Myocardium       (MYO)  ← primary ROI for bSSFP frequency selection
  3 = Left Ventricle   (LV blood pool)
"""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

CROP_SIZE = 256
NUM_CLASSES = 4


class LVSASegAdapter:
    """
    Parameters
    ----------
    checkpoint_path : str
        Absolute path to Unet_LVSA_best.pkl (or any LVSA .pkl / .pth checkpoint).
    device : str
        "cpu" or "cuda" / "cuda:0".
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self._model = self._load(checkpoint_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment_one(self, img: np.ndarray) -> np.ndarray:
        """
        Segment a single H×W float32 image.
        Returns H×W uint8 label map (0–3).
        """
        t, orig_h, orig_w, pad_h, pad_w = self._preprocess(img)
        with torch.no_grad():
            logits = self._model(t.to(self.device))       # 1×4×256×256
            pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return self._unpad(pred, orig_h, orig_w, pad_h, pad_w)

    def segment_consensus(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Run on all offset images (same anatomy, different artefacts) and return
        a consensus mask via per-pixel majority vote.  More robust than any single
        frame because banding darkens patches that could corrupt the segmentation.
        Returns H×W uint8 label map (0–3).
        """
        preds = []
        for img in images:
            try:
                preds.append(self.segment_one(img))
            except Exception as exc:
                logger.warning("Seg failed for one offset image: %s — skipping.", exc)

        if not preds:
            logger.error("All offset segmentations failed; returning zeros.")
            h, w = images[0].shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        # Stack → (N, H, W), then majority vote per pixel via bincount
        stack = np.stack(preds, axis=0)          # N×H×W
        N, H, W = stack.shape
        flat  = stack.reshape(N, H * W)           # N×(H*W)
        votes = np.zeros((NUM_CLASSES, H * W), dtype=np.int32)
        for cls in range(NUM_CLASSES):
            votes[cls] = (flat == cls).sum(axis=0)
        consensus = votes.argmax(axis=0).reshape(H, W).astype(np.uint8)
        return consensus

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self, checkpoint_path: str):
        from .unet import UNet

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model = UNet(input_channel=1, num_classes=NUM_CLASSES, feature_scale=1)

        suffix = ckpt_path.suffix.lower()
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        if suffix == ".pkl":
            # advchain format: {'model_state': ..., 'optimizer_state': ..., 'epoch': ...}
            state_dict = raw.get("model_state", raw)
        elif suffix == ".pth":
            # Some advchain checkpoints store the state dict directly
            state_dict = raw if isinstance(raw, dict) and "inc.conv.conv.0.weight" in raw else raw
        else:
            state_dict = raw

        # Strip DataParallel "module." prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys when loading LVSA U-Net: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected[:5])

        model.eval()
        model.to(self.device)
        logger.info("LVSA U-Net loaded from %s (device=%s).", ckpt_path.name, self.device)
        return model

    # ------------------------------------------------------------------
    # Preprocessing / postprocessing
    # ------------------------------------------------------------------

    def _preprocess(
        self, img: np.ndarray
    ) -> Tuple[torch.Tensor, int, int, int, int]:
        """
        Returns (tensor 1×1×256×256, orig_h, orig_w, pad_h, pad_w).
        pad_h / pad_w are the top/left padding added; used to unpad the prediction.
        """
        if img.ndim == 3:
            gray = img.mean(axis=2).astype(np.float32)
        else:
            gray = img.astype(np.float32)

        # --- [0,1] rescaling ---
        lo, hi = gray.min(), gray.max()
        if hi > lo:
            gray = (gray - lo) / (hi - lo)

        # --- Z-score normalise (matches advchain training default) ---
        mu, sigma = gray.mean(), gray.std()
        sigma = max(sigma, 1e-6)
        gray = (gray - mu) / sigma

        orig_h, orig_w = gray.shape

        # --- Centre-crop or pad to CROP_SIZE × CROP_SIZE ---
        # Pad axis if smaller; crop if larger
        th, tw = CROP_SIZE, CROP_SIZE

        # Height
        if orig_h < th:
            pad_top  = (th - orig_h) // 2
            pad_bot  = th - orig_h - pad_top
            gray = np.pad(gray, ((pad_top, pad_bot), (0, 0)), mode="reflect")
        else:
            pad_top = 0

        if orig_w < tw:
            pad_left = (tw - orig_w) // 2
            pad_right = tw - orig_w - pad_left
            gray = np.pad(gray, ((0, 0), (pad_left, pad_right)), mode="reflect")
        else:
            pad_left = 0

        # Crop centre if still larger
        h, w = gray.shape
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        gray = gray[start_h: start_h + th, start_w: start_w + tw]

        # The effective pad accounting for centre-crop offset
        net_pad_h = pad_top - start_h if orig_h < th else -start_h
        net_pad_w = pad_left - start_w if orig_w < tw else -start_w

        tensor = torch.from_numpy(gray[np.newaxis, np.newaxis]).float()  # 1×1×H×W
        return tensor, orig_h, orig_w, net_pad_h, net_pad_w

    def _unpad(
        self, pred: np.ndarray, orig_h: int, orig_w: int, pad_h: int, pad_w: int
    ) -> np.ndarray:
        """
        Crop/unpad prediction back to original H×W.

        ``pad_h`` / ``pad_w`` = amount of top/left padding that was added before
        the centre-crop (positive = we padded, negative = we centre-cropped).
        The original image content in the 256×256 prediction starts at
        row ``pad_h`` and column ``pad_w`` (when positive).

        Uses PIL NEAREST resize as a safe fallback when sizes still don't match.
        """
        h, w = pred.shape
        # When we added padding, the original content starts at offset pad_h / pad_w.
        # When we centre-cropped (pad_h < 0), original content starts at 0.
        r0 = max(pad_h, 0)
        c0 = max(pad_w, 0)
        r1 = min(r0 + orig_h, h)
        c1 = min(c0 + orig_w, w)
        cropped = pred[r0:r1, c0:c1]

        if cropped.shape == (orig_h, orig_w):
            return cropped

        # Fallback: resize via PIL NEAREST (label-safe)
        from PIL import Image as PILImage
        pil = PILImage.fromarray(cropped).resize((orig_w, orig_h), PILImage.NEAREST)
        return np.asarray(pil, dtype=np.uint8)
