"""
T6 VLM Descriptor
-----------------
Prompts a medical vision-language model (MedGemma-4B-it or Qwen2.5-VL-7B) to
describe banding artifacts and image quality for each frequency-scout slice.

Two-crop strategy (from our chest X-ray prior work):
  - Full-slice image: overall context.
  - Cropped myocardial ROI (from segmentation): close-up for fine-grained banding
    severity versus the wall.

Output: structured text per slice, parsed into a typed dict for the evidence state.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TOOL_NAME = "vlm"

# ------------------------------------------------------------------
# System prompt + per-slice user prompt templates
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a specialist cardiac MRI physicist reviewing a balanced
steady-state free precession (bSSFP) frequency scout acquisition at 3.0 Tesla.
You are assessing off-resonance banding artifacts: dark stripe-like regions caused
by the bSSFP signal null at ±1/(2·TR) from the transmitter frequency.

Rules:
- Answer ONLY about what you see in the image.
- Use clinical frequency-scouting language.
- Be concise and factual.
- Your output MUST be valid JSON matching the schema provided.
- IMPORTANT: You are comparing multiple candidate frequency offsets.
  NOT all of them can be "optimal". Assign "optimal" to at most ONE.
  All others must be "acceptable", "suboptimal", or "reject".
"""

_USER_PROMPT_TEMPLATE = """Frequency-scout image assessment task.
  Patient: {patient_id}
  Frequency offset: {offset_hz:+.0f} Hz
  Candidate rank: {candidate_rank} of {n_candidates} candidates under review

Examine the FULL slice image and (if provided) the cropped myocardial ROI.
Focus on: (1) banding artifact location relative to the left ventricular myocardium,
(2) intra-myocardial signal uniformity, (3) blood-pool brightness.

Return ONLY this JSON (no other text):
{{
  "banding_present": true/false,
  "banding_location": "none" | "anterior" | "lateral" | "inferior" | "septal" | "multiple" | "peripheral/lung/fat",
  "banding_severity": "none" | "mild" | "moderate" | "severe",
  "banding_overlaps_myocardium": true/false,
  "banding_overlaps_lv_pool": true/false,
  "myocardial_uniformity": "excellent" | "good" | "fair" | "poor",
  "blood_pool_brightness": "bright" | "reduced" | "dark",
  "overall_verdict": "optimal" | "acceptable" | "suboptimal" | "reject",
  "brief_reasoning": "<one or two sentences describing what you see in the myocardium>",
  "confidence_in_assessment": "high" | "medium" | "low"
}}
"""

# ------------------------------------------------------------------
# Data class
# ------------------------------------------------------------------

@dataclass
class VLMSliceResult:
    patient_id: str
    slice_idx: int
    offset_hz: float

    banding_present: bool
    banding_location: str        # "none" | "anterior" | … | "peripheral/lung/fat"
    banding_severity: str        # "none" | "mild" | "moderate" | "severe"
    banding_overlaps_myocardium: bool
    banding_overlaps_lv_pool: bool
    myocardial_uniformity: str   # "excellent" | "good" | "fair" | "poor"
    blood_pool_brightness: str   # "bright" | "reduced" | "dark"
    overall_verdict: str         # "optimal" | "acceptable" | "suboptimal" | "reject"
    brief_reasoning: str
    confidence_in_assessment: str
    raw_response: str

    provenance: Dict = field(default_factory=dict)


# ------------------------------------------------------------------
# Main tool
# ------------------------------------------------------------------

class VLMDescriptor:

    TOOL_NAME = TOOL_NAME

    def __init__(
        self,
        provider: str = "hf",
        model_id: str = "google/medgemma-4b-it",
        load_in_4bit: bool = True,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ):
        self.provider = provider
        self.model_id = model_id
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------

    def run(
        self,
        images: List[np.ndarray],
        offsets_hz: List[float],
        patient_id: str,
        seg_result=None,             # SegmentationResult or None
        target_indices: Optional[List[int]] = None,  # None = run on all
    ) -> List[VLMSliceResult]:
        """
        Describe banding quality for selected slices.

        Parameters
        ----------
        target_indices : list of ints or None
            Which slices to describe. None = all. Typically we start with the
            top-3 neural candidates and add more if the gap-analysis requires it.
        """
        if target_indices is None:
            target_indices = list(range(len(images)))

        self._ensure_model_loaded()

        results = []
        n_candidates = len(target_indices)
        for rank, idx in enumerate(target_indices):
            img = images[idx]
            hz  = offsets_hz[idx]
            roi = self._crop_roi(img, seg_result) if seg_result is not None else None
            result = self._describe_slice(
                img, roi, idx, hz, patient_id, len(images),
                candidate_rank=rank + 1, n_candidates=n_candidates,
            )
            results.append(result)
            logger.info(
                "VLM[%s][%d @ %.0f Hz]: verdict=%s severity=%s",
                patient_id, idx, hz, result.overall_verdict, result.banding_severity,
            )

        # Post-process: enforce at-most-one "optimal".
        # If the VLM labelled multiple candidates "optimal", demote all but the
        # highest-confidence one to "acceptable" (prevents all-optimal collapse).
        optimal_idxs = [i for i, r in enumerate(results) if r.overall_verdict == "optimal"]
        if len(optimal_idxs) > 1:
            best = max(
                optimal_idxs,
                key=lambda i: {"high": 2, "medium": 1, "low": 0}.get(
                    results[i].confidence_in_assessment, 0
                ),
            )
            for i in optimal_idxs:
                if i != best:
                    results[i] = VLMSliceResult(
                        **{
                            **results[i].__dict__,
                            "overall_verdict": "acceptable",
                            "brief_reasoning": results[i].brief_reasoning
                            + " [demoted: at most one optimal allowed per scout]",
                        }
                    )
            logger.info(
                "VLM[%s]: demoted %d duplicate 'optimal' verdicts to 'acceptable'.",
                patient_id, len(optimal_idxs) - 1,
            )

        return results

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self):
        if self._model is not None:
            return
        mid = str(self.model_id or "").strip().lower()
        if (
            self.provider in ("mock", "none", "disabled")
            or mid in ("mock", "disabled-mock", "none", "")
            or "mock" in mid
        ):
            logger.info("VLM mock mode (no HuggingFace download).")
            self._model = "mock"
            return
        if self.provider == "ollama":
            logger.info("Using Ollama backend (model=%s).", self.model_id)
            return
        try:
            import torch
            logger.info("Loading VLM %s …", self.model_id)
            kwargs: Dict[str, Any] = {"device_map": self.device_map}
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = self._load_vlm_weights(kwargs)
            self._model.eval()
            logger.info("VLM loaded.")
        except Exception as exc:
            logger.error("Failed to load VLM: %s. Will use mock responses.", exc)
            self._model = "mock"

    def _load_vlm_weights(self, kwargs: Dict[str, Any]):
        """Qwen2.5-VL uses a dedicated class; fall back through HF auto classes."""
        mid = str(self.model_id).lower()
        if "qwen2.5-vl" in mid or "qwen2_5_vl" in mid:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id, trust_remote_code=True, **kwargs
                )
            except Exception as exc:
                logger.warning("Qwen2_5_VL class failed (%s); trying AutoModel.", exc)
        try:
            from transformers import AutoModelForImageTextToText
            return AutoModelForImageTextToText.from_pretrained(
                self.model_id, trust_remote_code=True, **kwargs
            )
        except Exception:
            from transformers import AutoModelForVision2Seq
            return AutoModelForVision2Seq.from_pretrained(
                self.model_id, trust_remote_code=True, **kwargs
            )

    def _describe_slice(
        self,
        img: np.ndarray,
        roi: Optional[np.ndarray],
        slice_idx: int,
        offset_hz: float,
        patient_id: str,
        n_slices: int,
        candidate_rank: int = 1,
        n_candidates: int = 1,
    ) -> VLMSliceResult:
        prompt = _USER_PROMPT_TEMPLATE.format(
            patient_id=patient_id,
            offset_hz=offset_hz,
            slice_idx=slice_idx,
            n_slices=n_slices,
            candidate_rank=candidate_rank,
            n_candidates=n_candidates,
        )

        if self._model == "mock" or self._model is None:
            raw = self._mock_response(slice_idx, offset_hz)
        elif self.provider == "ollama":
            raw = self._ollama_call(img, roi, prompt)
        else:
            try:
                raw = self._hf_call(img, roi, prompt)
            except Exception as exc:
                logger.warning(
                    "VLM inference failed for slice %d (%s). Using mock for this slice.",
                    slice_idx, exc,
                )
                raw = self._mock_response(slice_idx, offset_hz)

        parsed = self._parse_response(raw, slice_idx, offset_hz, patient_id)
        return parsed

    def _model_device(self):
        import torch
        try:
            return self._model.device
        except Exception:
            return next(p.device for p in self._model.parameters() if p.device.type != "meta")

    def _hf_call(self, img: np.ndarray, roi: Optional[np.ndarray], prompt: str) -> str:
        """Qwen2-VL / Qwen2.5-VL: chat template returns text; processor builds tensors."""
        import torch
        from PIL import Image

        def arr_to_pil(arr):
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                return Image.fromarray(arr).convert("RGB")
            return Image.fromarray(arr)

        # One image is more reliable on 8GB; ROI is optional extra
        pil_full = arr_to_pil(img)
        pil_images = [pil_full]
        if roi is not None:
            pil_images.append(arr_to_pil(roi))

        user_content = [{"type": "image", "image": p} for p in pil_images]
        user_content.append({"type": "text", "text": prompt})
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(text, str):
            # Some processors return token ids; decode back to string path
            text = str(text)

        proc_out = self._processor(
            text=[text],
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )

        device = self._model_device()
        if hasattr(proc_out, "to"):
            proc_out = proc_out.to(device)
            inputs = dict(proc_out)
        elif isinstance(proc_out, dict):
            inputs = {
                k: (v.to(device) if hasattr(v, "to") else v)
                for k, v in proc_out.items()
            }
        else:
            raise TypeError(f"Unexpected processor output type: {type(proc_out)}")

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = output[0, input_len:]
        decode = getattr(self._processor, "batch_decode", None)
        if decode is not None:
            response = self._processor.batch_decode(
                new_tokens.unsqueeze(0), skip_special_tokens=True
            )[0]
        else:
            tok = getattr(self._processor, "tokenizer", self._processor)
            response = tok.decode(new_tokens, skip_special_tokens=True)

        if prompt[:30] in response:
            response = response[response.find(prompt[:30]) + len(prompt):]
        return response.strip()

    def _ollama_call(self, img: np.ndarray, roi: Optional[np.ndarray], prompt: str) -> str:
        import requests

        def arr_to_b64(arr):
            from PIL import Image
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            if arr.ndim == 2:
                pil = Image.fromarray(arr).convert("RGB")
            else:
                pil = Image.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode()

        images_b64 = [arr_to_b64(img)]
        if roi is not None:
            images_b64.append(arr_to_b64(roi))

        payload = {
            "model": self.model_id,
            "system": _SYSTEM_PROMPT,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "")

    # ------------------------------------------------------------------
    # Parsing & mock
    # ------------------------------------------------------------------

    def _parse_response(
        self, raw: str, slice_idx: int, offset_hz: float, patient_id: str
    ) -> VLMSliceResult:
        defaults = {
            "banding_present": False,
            "banding_location": "none",
            "banding_severity": "none",
            "banding_overlaps_myocardium": False,
            "banding_overlaps_lv_pool": False,
            "myocardial_uniformity": "good",
            "blood_pool_brightness": "bright",
            "overall_verdict": "acceptable",
            "brief_reasoning": "",
            "confidence_in_assessment": "low",
        }
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                defaults.update({k: v for k, v in parsed.items() if k in defaults})
            except json.JSONDecodeError:
                logger.warning("Could not parse VLM JSON for slice %d.", slice_idx)

        return VLMSliceResult(
            patient_id=patient_id,
            slice_idx=int(slice_idx),
            offset_hz=float(offset_hz),
            banding_present=bool(defaults["banding_present"]),
            banding_location=str(defaults["banding_location"]),
            banding_severity=str(defaults["banding_severity"]),
            banding_overlaps_myocardium=bool(defaults["banding_overlaps_myocardium"]),
            banding_overlaps_lv_pool=bool(defaults["banding_overlaps_lv_pool"]),
            myocardial_uniformity=str(defaults["myocardial_uniformity"]),
            blood_pool_brightness=str(defaults["blood_pool_brightness"]),
            overall_verdict=str(defaults["overall_verdict"]),
            brief_reasoning=str(defaults["brief_reasoning"]),
            confidence_in_assessment=str(defaults["confidence_in_assessment"]),
            raw_response=raw,
            provenance={
                "tool": TOOL_NAME,
                "model_id": self.model_id,
                "slice_idx": int(slice_idx),
                "offset_hz": float(offset_hz),
            },
        )

    def _mock_response(self, slice_idx: int, offset_hz: float) -> str:
        """
        Return a physics-informed mock JSON that distinguishes offsets.
        Verdict grades: |Hz| ≤ 25 → acceptable, |Hz| ≤ 75 → suboptimal, >75 → reject.
        (Cannot assign 'optimal' in mock — only a real VLM looking at the image can.)
        """
        hz = float(offset_hz)
        abs_hz = abs(hz)
        has_band = abs_hz > 50
        severity = "severe" if abs_hz > 125 else ("moderate" if abs_hz > 75 else ("mild" if has_band else "none"))
        verdict = "reject" if abs_hz > 125 else ("suboptimal" if abs_hz > 75 else "acceptable")
        location = "inferior" if hz < -50 else ("anterior" if hz > 50 else "none")
        return json.dumps({
            "banding_present": bool(has_band),
            "banding_location": location,
            "banding_severity": severity,
            "banding_overlaps_myocardium": bool(abs_hz > 100),
            "banding_overlaps_lv_pool": False,
            "myocardial_uniformity": "good" if abs_hz <= 25 else ("fair" if abs_hz <= 75 else "poor"),
            "blood_pool_brightness": "bright",
            "overall_verdict": verdict,
            "brief_reasoning": f"Mock: offset {hz:+.0f} Hz, banding classified as {severity}.",
            "confidence_in_assessment": "low",
        })

    # ------------------------------------------------------------------
    # ROI crop
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_roi(img: np.ndarray, seg_result) -> Optional[np.ndarray]:
        """Crop a bounding box around the myocardium + LV pool."""
        roi_mask = (seg_result.mask == 2) | (seg_result.mask == 3)
        rows, cols = np.where(roi_mask)
        if rows.size == 0:
            return None
        pad = 15
        r0, r1 = max(0, rows.min() - pad), min(img.shape[0], rows.max() + pad)
        c0, c1 = max(0, cols.min() - pad), min(img.shape[1], cols.max() + pad)
        return img[r0:r1, c0:c1]
