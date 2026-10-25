"""
FastAPI Server for Remote 10GB RTX 3080 PC (OpenAI-compatible Chat Completions)
--------------------------------------------------------------------------------
Run this script on the remote GPU machine over SSH:
    python server.py --model Qwen/Qwen2.5-7B-Instruct --port 8000 --load-in-4bit

Memory requirement: ~5.2 GB VRAM in 4-bit (fits comfortably in 10GB RTX 3080).
"""

import argparse
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("remote_llm_server")

app = FastAPI(title="FreqScout Remote LLM Server (10GB RTX 3080)")

model = None
tokenizer = None
current_model_name = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "Qwen/Qwen2.5-7B-Instruct"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 1024


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        import torch

        conv = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        gen_kwargs = {
            "max_new_tokens": req.max_tokens or 1024,
            "do_sample": (req.temperature or 0.0) > 0.0,
        }
        if (req.temperature or 0.0) > 0.0:
            gen_kwargs["temperature"] = req.temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        return {
            "id": "chatcmpl-freqscout-3080",
            "object": "chat.completion",
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
        }
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model": current_model_name, "gpu_available": True}


def main():
    parser = argparse.ArgumentParser(description="Run LLM server on 10GB RTX 3080")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="HF model name")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Load with 4-bit quantization")
    args = parser.parse_args()

    global model, tokenizer, current_model_name
    current_model_name = args.model

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading {args.model} on GPU (4-bit={args.load_in_4bit})...")

    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 if not args.load_in_4bit else None,
        trust_remote_code=True,
    )
    logger.info("Model loaded. Starting uvicorn server...")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
