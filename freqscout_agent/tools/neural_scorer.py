"""
T4 Neural Ranker Tool
---------------------
Wraps the DINOv2-L + Gradient-Aware Stack-Transformer from Exp5/Exp6 to produce
per-slice scores and softmax-normalised confidence values.

Priority order:
  1. Load cached DINOv2-L features from exp5_outputs/dinov2l_features.pkl
     and run the stack-transformer head (if a checkpoint is provided).
  2. If checkpoint missing but cache exists: score by L2-norm of the DINOv2
     feature vector (larger norm → higher visual complexity → proxy for quality).
  3. If nothing is available: run DINOv2-L from HuggingFace on the raw images
     and use the same feature-norm proxy.

All outputs are provenance-tagged.
"""

from __future__ import annotations

import logging
import pickle
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TOOL_NAME = "neural"


@dataclass
class NeuralResult:
    patient_id: str
    slice_scores: List[float]       # raw score per slice (unnormalised)
    softmax_confidence: List[float] # softmax over scores (sums to 1)
    ranking: List[int]              # slice indices sorted best→worst
    top1_idx: int
    top3_idx: List[int]
    method: str                     # "exp5_loocv" | "transformer" | "feature_norm" | "dinov2_live"
    provenance: Dict = field(default_factory=dict)


class NeuralRanker:

    TOOL_NAME = TOOL_NAME

    def __init__(
        self,
        dinov2_model: str = "facebook/dinov2-large",
        checkpoint: Optional[str] = None,
        feature_cache: str = "exp5_outputs/dinov2l_features.pkl",
        score_csv: str = "exp5_outputs/exp5_loocv_per_patient.csv",
        device: str = "cpu",
        use_cached_features: bool = True,
    ):
        self.dinov2_model_id = dinov2_model
        self.checkpoint = checkpoint
        self.feature_cache_path = Path(feature_cache)
        self.score_csv_path = Path(score_csv) if score_csv else None
        self.device = device
        self.use_cached_features = use_cached_features

        self._cache: Optional[Dict] = None
        self._oof_scores: Dict[str, Dict[int, float]] = {}
        self._transformer_model = None
        self._dinov2 = None

        self._load_oof_csv()
        if use_cached_features and self.feature_cache_path.exists():
            self._load_cache()

        if checkpoint and Path(checkpoint).exists():
            self._load_transformer(checkpoint)

    # ------------------------------------------------------------------

    def run(
        self,
        images: List[np.ndarray],
        offsets_hz: List[float],
        patient_id: str,
    ) -> NeuralResult:
        """Score all frequency-scout slices for one patient."""

        oof = self._scores_from_oof(patient_id, len(images))
        if oof is not None:
            scores, method = oof, "exp5_loocv"
        else:
            features = self._get_features(images, patient_id)
            scores, method = self._score(features, patient_id)

        softmax = self._softmax(scores)
        ranking = list(np.argsort(scores)[::-1])
        top1 = int(ranking[0])
        top3 = [int(r) for r in ranking[:3]]

        logger.info(
            "NeuralRanker[%s]: top1=%d (%.3f conf), method=%s",
            patient_id, top1, softmax[top1], method,
        )
        return NeuralResult(
            patient_id=patient_id,
            slice_scores=[float(s) for s in scores],
            softmax_confidence=[float(s) for s in softmax],
            ranking=ranking,
            top1_idx=top1,
            top3_idx=top3,
            method=method,
            provenance={
                "tool": TOOL_NAME,
                "method": method,
                "dinov2_model": self.dinov2_model_id,
                "checkpoint": self.checkpoint,
                "score_csv": str(self.score_csv_path) if self.score_csv_path else None,
            },
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _load_oof_csv(self):
        """Load Exp5 LOOCV per-slice scores (1-indexed image numbers in JSON)."""
        if self.score_csv_path is None or not self.score_csv_path.exists():
            return
        try:
            import ast
            import json
            import pandas as pd
            df = pd.read_csv(self.score_csv_path)
            if "patient_id" not in df.columns or "slice_scores" not in df.columns:
                logger.warning("Score CSV missing required columns: %s", self.score_csv_path)
                return
            n = 0
            for _, row in df.iterrows():
                pid = str(row["patient_id"]).strip()
                raw = row["slice_scores"]
                parsed = None
                if isinstance(raw, dict):
                    parsed = raw
                else:
                    try:
                        parsed = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        try:
                            parsed = ast.literal_eval(str(raw))
                        except (ValueError, SyntaxError):
                            parsed = None
                if not isinstance(parsed, dict):
                    continue
                self._oof_scores[pid] = {int(k): float(v) for k, v in parsed.items()}
                n += 1
            logger.info("Loaded Exp5 LOOCV scores for %d patients from %s.", n, self.score_csv_path)
        except Exception as exc:
            logger.warning("Could not load Exp5 score CSV: %s", exc)
            self._oof_scores = {}

    def _scores_from_oof(self, patient_id: str, n_slices: int) -> Optional[np.ndarray]:
        d = self._oof_scores.get(patient_id)
        if not d:
            return None
        keys = sorted(d)
        arr = np.array([d[k] for k in keys], dtype=float)
        if arr.shape[0] == n_slices:
            return arr
        logger.warning(
            "OOF scores for %s have %d entries but scout has %d frames.",
            patient_id, arr.shape[0], n_slices,
        )
        if arr.shape[0] > n_slices:
            return arr[:n_slices]
        pad = np.full(n_slices - arr.shape[0], float(arr.min()) - 1.0)
        return np.concatenate([arr, pad])

    def _load_cache(self):
        try:
            with open(self.feature_cache_path, "rb") as f:
                self._cache = pickle.load(f)
            logger.info("Loaded DINOv2-L feature cache from %s.", self.feature_cache_path)
        except Exception as exc:
            logger.warning("Could not load feature cache: %s", exc)
            self._cache = None

    def _get_features(
        self, images: List[np.ndarray], patient_id: str
    ) -> np.ndarray:
        """Return (N_slices, D) feature matrix."""

        # 1. Try cached features (numpy or torch)
        if self._cache is not None:
            key = patient_id
            if key in self._cache:
                feats = self._normalize_cached_feats(self._cache[key])
                if feats is not None:
                    # If cache length mismatches current scout length, fall back
                    if feats.shape[0] == len(images):
                        return feats
                    logger.warning(
                        "Cache for %s has %d rows but scout has %d frames; extracting live.",
                        key, feats.shape[0], len(images),
                    )
                else:
                    logger.warning("Cache entry for %s has unexpected shape; extracting live.", key)

        # 2. Extract live with DINOv2
        return self._extract_dinov2(images)

    @staticmethod
    def _normalize_cached_feats(feats) -> Optional[np.ndarray]:
        """Convert cached features to a 2D float32 numpy array, or None if unusable."""
        try:
            import torch
            if isinstance(feats, torch.Tensor):
                feats = feats.detach().cpu().numpy()
        except ImportError:
            pass
        feats = np.asarray(feats)
        if feats.ndim == 2 and feats.shape[0] > 0:
            return feats.astype(np.float32)
        return None

    def _extract_dinov2(self, images: List[np.ndarray]) -> np.ndarray:
        """Run DINOv2-L on the images and return (N, D) CLS token features."""
        import torch
        from transformers import AutoModel, AutoImageProcessor
        from PIL import Image

        if self._dinov2 is None:
            logger.info("Loading DINOv2 model %s…", self.dinov2_model_id)
            processor = AutoImageProcessor.from_pretrained(self.dinov2_model_id)
            model = AutoModel.from_pretrained(self.dinov2_model_id)
            model.eval()
            model.to(self.device)
            self._dinov2 = (model, processor)

        model, processor = self._dinov2
        feats = []
        with torch.no_grad():
            for img in images:
                if img.max() <= 1.0:
                    img_uint8 = (img * 255).astype(np.uint8)
                else:
                    img_uint8 = img.astype(np.uint8)
                if img_uint8.ndim == 2:
                    pil = Image.fromarray(img_uint8).convert("RGB")
                else:
                    pil = Image.fromarray(img_uint8)
                inputs = processor(images=pil, return_tensors="pt").to(self.device)
                outputs = model(**inputs)
                cls = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
                feats.append(cls)
        return np.stack(feats, axis=0)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _load_transformer(self, checkpoint: str):
        try:
            import torch
            # Import the gradient-aware transformer from exp6 if available
            import sys
            sys.path.insert(0, str(Path(checkpoint).parent))
            state = torch.load(checkpoint, map_location=self.device)
            logger.info("Loaded transformer checkpoint from %s.", checkpoint)
            self._transformer_model = state
        except Exception as exc:
            logger.warning("Could not load transformer checkpoint: %s", exc)

    def _score(self, features: np.ndarray, patient_id: str):
        """Return (scores array, method_name)."""

        if self._transformer_model is not None:
            try:
                scores = self._run_transformer(features)
                return scores, "transformer"
            except Exception as exc:
                logger.warning("Transformer scoring failed (%s); falling back.", exc)

        # Feature-norm proxy: higher L2 norm of CLS token → stronger visual signal
        # In DINOv2, clean / high-contrast images typically have larger norms.
        scores = np.linalg.norm(features, axis=1).astype(float)
        return scores, "feature_norm"

    def _run_transformer(self, features: np.ndarray) -> np.ndarray:
        import torch
        model = self._transformer_model
        if not callable(getattr(model, "forward", None)):
            raise RuntimeError("Stored state is not a callable model.")
        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
            logits = model(x)  # 1×N
            return logits.squeeze(0).cpu().numpy()

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        shifted = scores - scores.max()
        exp_s = np.exp(shifted)
        return exp_s / exp_s.sum()
