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
"""

_USER_PROMPT_TEMPLATE = """Image description task for bSSFP frequency scout slice
  Patient: {patient_id}
  Frequency offset: {offset_hz:+.0f} Hz
  Slice index: {slice_idx} of {n_slices}

Examine the FULL slice image and (if provided) the cropped myocardial ROI.

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
  "brief_reasoning": "<one or two sentences>",
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
        for idx in target_indices:
            img = images[idx]
            hz  = offsets_hz[idx]
            roi = self._crop_roi(img, seg_result) if seg_result is not None else None
            result = self._describe_slice(
                img, roi, idx, hz, patient_id, len(images)
            )
            results.append(result)
            logger.info(
                "VLM[%s][%d @ %.0f Hz]: verdict=%s severity=%s",
                patient_id, idx, hz, result.overall_verdict, result.banding_severity,
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
    ) -> VLMSliceResult:
        prompt = _USER_PROMPT_TEMPLATE.format(
            patient_id=patient_id,
            offset_hz=offset_hz,
            slice_idx=slice_idx,
            n_slices=n_slices,
        )

        if self._model == "mock" or self._model is None:
            raw = self._mock_response(slice_idx, offset_hz)
        elif self.provider == "ollama":
            raw = self._ollama_call(img, roi, prompt)
        else:
            raw = self._hf_call(img, roi, prompt)

        parsed = self._parse_response(raw, slice_idx, offset_hz, patient_id)
        return parsed

    def _hf_call(self, img: np.ndarray, roi: Optional[np.ndarray], prompt: str) -> str:
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

        pil_images = [arr_to_pil(img)]
        if roi is not None:
            pil_images.append(arr_to_pil(roi))

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": pil} for pil in pil_images],
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        try:
            inputs = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        except Exception:
            # Fallback for processors that don't support apply_chat_template
            inputs = self._processor(
                text=_SYSTEM_PROMPT + "\n\n" + prompt,
                images=pil_images,
                return_tensors="pt",
            )

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)

        response = self._processor.decode(output[0], skip_special_tokens=True)
        # Strip any leading echo of the prompt
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
        """Return a plausible mock JSON for testing without a real VLM."""
        hz = float(offset_hz)
        has_band = bool(abs(hz) > 75)
        severity = "moderate" if abs(hz) > 100 else ("mild" if has_band else "none")
        verdict = "reject" if abs(hz) > 100 else ("acceptable" if has_band else "optimal")
        location = "inferior" if hz < -50 else ("anterior" if hz > 50 else "none")
        return json.dumps({
            "banding_present": bool(has_band),
            "banding_location": location,
            "banding_severity": severity,
            "banding_overlaps_myocardium": bool(has_band and abs(hz) > 100),
            "banding_overlaps_lv_pool": False,
            "myocardial_uniformity": "excellent" if not has_band else "fair",
            "blood_pool_brightness": "bright",
            "overall_verdict": verdict,
            "brief_reasoning": f"Mock response for offset {hz:+.0f} Hz.",
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
