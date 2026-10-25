"""
LLM Client (Supporting OpenAI API / Groq / HuggingFace 4-bit / Template Fallback)
----------------------------------------------------------------------------------
Designed to run either locally on GPU (RTX 3080 via HuggingFace bitsandbytes 4-bit)
or via Cloud APIs (Groq / OpenAI) with automatic retry and rate-limit backoff.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMClient:

    def __init__(
        self,
        backend: str = "template",               # "openai" | "hf" | "template"
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "none",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit: bool = True,
        temperature: float = 0.2,
        max_new_tokens: int = 1024,
    ):
        self.backend = backend
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        self._hf_model = None
        self._hf_tokenizer = None

        if self.backend == "hf":
            self._init_hf()

    def _init_hf(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            logger.info(f"Loading HuggingFace model {self.model_name} (4-bit: {self.load_in_4bit})...")

            bnb_config = None
            if self.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )

            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16 if not self.load_in_4bit else None,
                trust_remote_code=True,
            )
            logger.info("HuggingFace model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}. Falling back to template mode.")
            self.backend = "template"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.backend == "openai":
            return self._call_openai_api(system_prompt, user_prompt)
        elif self.backend == "hf" and self._hf_model is not None:
            return self._call_hf(system_prompt, user_prompt)
        else:
            return ""

    def _call_openai_api(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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
                logger.warning(f"OpenAI API call failed ({exc}). Returning empty.")
                return ""
            except Exception as exc:
                logger.warning(f"OpenAI API call failed ({exc}). Returning empty.")
                return ""
        return ""

    def _call_hf(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self._hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._hf_tokenizer([prompt], return_tensors="pt").to(self._hf_model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            outputs = self._hf_model.generate(
                **inputs,
                **gen_kwargs,
            )
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        return self._hf_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
