"""
Multimodal VLM Client (Supporting OpenAI Vision / Qwen2-VL 4-bit / Template Fallback)
--------------------------------------------------------------------------------------
Allows local GPU execution of Vision-Language Models (e.g. Qwen2-VL-7B-Instruct)
or cloud execution via OpenAI/OpenRouter Vision endpoints.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger(__name__)


def encode_image_base64(image_path: Union[str, Path]) -> str:
    """Read an image file and return its base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class VLMClient:

    def __init__(
        self,
        backend: str = "template",               # "openai" | "hf" | "template"
        api_base: str = "https://api.openai.com/v1",
        api_key: str = "none",
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        load_in_4bit: bool = True,
        temperature: float = 0.1,
        max_new_tokens: int = 512,
    ):
        self.backend = backend
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self._hf_model = None
        self._hf_processor = None

        if self.backend == "hf":
            self._init_hf_vlm()

    def _init_hf_vlm(self):
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

            logger.info(f"Loading HuggingFace VLM {self.model_name} (4-bit: {self.load_in_4bit})...")

            bnb_config = None
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )

            self._hf_processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self._hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16 if not self.load_in_4bit else None,
                trust_remote_code=True,
            )
            logger.info("HuggingFace VLM loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace VLM: {e}. Falling back to template mode.")
            self.backend = "template"

    def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Multimodal inference passing prompt and an image.
        """
        if self.backend == "openai":
            return self._call_openai_vision(system_prompt, user_prompt, image_path)
        elif self.backend == "hf" and self._hf_model is not None:
            return self._call_hf_vlm(system_prompt, user_prompt, image_path)
        else:
            return ""

    def _call_openai_vision(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[Union[str, Path]] = None,
    ) -> str:
        url = f"{self.api_base.rstrip('/')}/chat/completions"

        user_content = [{"type": "text", "text": user_prompt}]
        if image_path and Path(image_path).exists():
            b64 = encode_image_base64(image_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            })

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
        }

        req_data = json.dumps(payload).encode("utf-8")

        for attempt in range(4):
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"].strip()
            except urllib.error.HTTPError as exc:
                if exc.code == 429 and attempt < 3:
                    wait_sec = (attempt + 1) * 15
                    logger.warning(f"Rate limit (429) hit. Waiting {wait_sec}s for rate limit window reset...")
                    time.sleep(wait_sec)
                    continue
                logger.warning(f"OpenAI Vision API call failed ({exc}). Returning empty.")
                return ""
            except Exception as exc:
                logger.warning(f"OpenAI Vision API call failed ({exc}). Returning empty.")
                return ""
        return ""

    def _call_hf_vlm(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: Optional[Union[str, Path]] = None,
    ) -> str:
        from PIL import Image
        import torch

        image = None
        if image_path and Path(image_path).exists():
            image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": ([{"type": "image", "image": image}] if image else []) + [{"type": "text", "text": user_prompt}],
            },
        ]

        prompt = self._hf_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = [image] if image else None
        inputs = self._hf_processor(
            text=[prompt],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self._hf_model.device)

        with torch.no_grad():
            outputs = self._hf_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
            )

        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        return self._hf_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
