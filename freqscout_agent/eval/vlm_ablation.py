"""
VLM Paradigm Ablation Track
----------------------------
Mirrors the ADRE Table-1 ablation: Zero-shot | Few-shot | RAG-only | Agentic-only | Full Agentic.
Provides harnesses to run the VLM in isolation (without the full agentic loop) for ablation.

Modes
-----
zero_shot  : single prompt with image(s), no examples, no retrieved knowledge.
few_shot   : k exemplar (image-set + reasoning) pairs prepended to prompt.
rag_only   : retrieved knowledge injected but no iterative tool loop.
agentic    : full loop but no VLM (CV + neural only).
full       : full agentic system (default pipeline).

LoRA fine-tuning scaffold is stubbed here (activated when doctor GT arrives).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    mode: str                           # "zero_shot"|"few_shot"|"rag_only"|"agentic"|"full"
    patient_id: str
    pred_slice: int
    pred_hz: float
    gt_slice: Optional[int]
    gt_hz: Optional[float]
    top1_correct: bool
    top3_correct: bool
    confidence: float
    raw_output: str = ""


class VLMAblation:

    # Exemplars for few-shot (bootstrapped from pipeline outputs, not doctor GT)
    EXEMPLARS: List[Dict] = []          # populated by load_exemplars()

    def __init__(
        self,
        vlm_tool=None,                   # VLMDescriptor instance
        retriever=None,                  # KnowledgeRetriever instance
        n_exemplars: int = 3,
        exemplar_path: Optional[str] = None,
    ):
        self.vlm = vlm_tool
        self.retriever = retriever
        self.n_exemplars = n_exemplars
        if exemplar_path:
            self._load_exemplars(exemplar_path)

    # ------------------------------------------------------------------

    def run_zero_shot(
        self,
        images: List,
        offsets_hz: List[float],
        patient_id: str,
        seg_result=None,
    ) -> AblationResult:
        """Run VLM with no examples, no retrieved knowledge."""
        if self.vlm is None:
            return self._dummy_result("zero_shot", patient_id, images, offsets_hz)

        results = self.vlm.run(images, offsets_hz, patient_id, seg_result)
        pred_slice, pred_hz, conf = self._best_from_vlm(results, offsets_hz)
        return AblationResult(
            mode="zero_shot", patient_id=patient_id,
            pred_slice=pred_slice, pred_hz=pred_hz, gt_slice=None, gt_hz=None,
            top1_correct=False, top3_correct=False,
            confidence=conf, raw_output=json.dumps([r.overall_verdict for r in results]),
        )

    def run_few_shot(
        self,
        images: List,
        offsets_hz: List[float],
        patient_id: str,
        seg_result=None,
    ) -> AblationResult:
        """Run VLM with k exemplars prepended."""
        if self.vlm is None or not self.EXEMPLARS:
            return self._dummy_result("few_shot", patient_id, images, offsets_hz)

        exemplars = self.EXEMPLARS[:self.n_exemplars]
        exemplar_text = self._format_exemplars(exemplars)
        # Temporarily override prompt template (not shown; relies on vlm_tool's flexible prompt)
        results = self.vlm.run(
            images, offsets_hz, patient_id, seg_result, extra_context=exemplar_text
        )
        pred_slice, pred_hz, conf = self._best_from_vlm(results, offsets_hz)
        return AblationResult(
            mode="few_shot", patient_id=patient_id,
            pred_slice=pred_slice, pred_hz=pred_hz, gt_slice=None, gt_hz=None,
            top1_correct=False, top3_correct=False, confidence=conf,
        )

    def run_rag_only(
        self,
        images: List,
        offsets_hz: List[float],
        patient_id: str,
        seg_result=None,
    ) -> AblationResult:
        """Run VLM with retrieved knowledge but no iterative loop."""
        if self.vlm is None:
            return self._dummy_result("rag_only", patient_id, images, offsets_hz)

        knowledge = ""
        if self.retriever:
            chunks = self.retriever.retrieve("bSSFP banding artifact optimal frequency offset cardiac MRI")
            knowledge = "\n".join(c.text for c in chunks[:3])

        results = self.vlm.run(
            images, offsets_hz, patient_id, seg_result, extra_context=knowledge
        )
        pred_slice, pred_hz, conf = self._best_from_vlm(results, offsets_hz)
        return AblationResult(
            mode="rag_only", patient_id=patient_id,
            pred_slice=pred_slice, pred_hz=pred_hz, gt_slice=None, gt_hz=None,
            top1_correct=False, top3_correct=False, confidence=conf,
        )

    # ------------------------------------------------------------------
    # LoRA fine-tuning scaffold
    # ------------------------------------------------------------------

    def prepare_lora_dataset(
        self,
        pipeline_outputs: List[Dict],
        output_dir: str = "lora_data",
    ):
        """
        Prepare training data for LoRA fine-tuning from pipeline outputs.
        Each sample: (image path(s), prompt, narrative response).

        When doctor GT reasoning narratives arrive:
          1. Replace pipeline outputs with doctor narratives.
          2. Run fine-tuning using the generated script.
          3. Re-evaluate on held-out patients.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        samples = []
        for p in pipeline_outputs:
            samples.append({
                "patient_id": p.get("patient_id"),
                "prompt":     p.get("stage_b_prompt", ""),
                "response":   p.get("narrative", ""),
                "images":     p.get("image_paths", []),
                "stage_a":    p.get("stage_a", {}),
            })

        with open(out / "lora_train.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # Write a Kaggle-ready LoRA training script scaffold
        self._write_lora_script(out)
        logger.info("LoRA dataset prepared: %d samples → %s", len(samples), out)

    @staticmethod
    def _write_lora_script(out_dir: Path):
        script = '''#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for Frequency-Scout VLM
================================================
Run on Kaggle GPU (T4/P100) when doctor reasoning GT is available.

Usage:
    python lora_finetune.py --model google/medgemma-4b-it \\
        --data lora_data/lora_train.jsonl \\
        --output lora_checkpoints/ \\
        --epochs 3 --lr 2e-4
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/medgemma-4b-it")
    parser.add_argument("--data", default="lora_data/lora_train.jsonl")
    parser.add_argument("--output", default="lora_checkpoints/")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import TrainingArguments, Trainer
        import json, torch
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install peft transformers trl")
        return

    print(f"Loading model {args.model}…")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    samples = []
    with open(args.data) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Dataset: {len(samples)} samples. Fine-tune loop stubbed — wire Trainer here.")
    # TODO: implement DataCollator + Trainer loop when GT arrives.
    print("LoRA scaffold ready. Implement DataCollator + Trainer and run on Kaggle GPU.")

if __name__ == "__main__":
    main()
'''
        (out_dir / "lora_finetune.py").write_text(script)
        logger.info("LoRA training script written to %s.", out_dir / "lora_finetune.py")

    # ------------------------------------------------------------------

    def _best_from_vlm(self, results, offsets_hz):
        """Pick the best slice from VLM verdicts."""
        verdict_rank = {"optimal": 3, "acceptable": 2, "suboptimal": 1, "reject": 0}
        best = max(results, key=lambda r: verdict_rank.get(r.overall_verdict, 1))
        return best.slice_idx, offsets_hz[best.slice_idx], 0.7

    def _dummy_result(self, mode, patient_id, images, offsets_hz):
        mid = len(images) // 2
        return AblationResult(
            mode=mode, patient_id=patient_id,
            pred_slice=mid, pred_hz=offsets_hz[mid],
            gt_slice=None, gt_hz=None,
            top1_correct=False, top3_correct=False,
            confidence=0.5,
        )

    def _load_exemplars(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        with open(p) as f:
            self.EXEMPLARS = [json.loads(l) for l in f if l.strip()]
        logger.info("Loaded %d few-shot exemplars.", len(self.EXEMPLARS))

    @staticmethod
    def _format_exemplars(exemplars: List[Dict]) -> str:
        lines = ["=== Few-Shot Examples ==="]
        for ex in exemplars:
            lines.append(f"\nPatient: {ex.get('patient_id','?')}")
            lines.append(f"Findings: {json.dumps(ex.get('stage_a',{}))[:400]}")
            lines.append(f"Narrative: {ex.get('narrative','')[:300]}")
        return "\n".join(lines)
