"""
Stage B / Stage-C: Clinical Narrative Generator
------------------------------------------------
Exp11 Stage-C produces a radiologist-style frequency-scout report grounded in:
  1. Stage-A structured findings (JSON) — physics reasons, graded sets
  2. Retrieved knowledge chunks (optional)
  3. Consistency report (source reconciliation)

Selection remains Exp10 physics fusion (W_VLM=0). This module explains the pick.

If no LLM is available, fills the Stage-C schema from Stage-A deterministically.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Any

from ..agent.evidence_state import EvidenceState
from ..agent.consistency import ConsistencyReport
from ..knowledge.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a specialist cardiac MRI physicist writing a clinical
frequency-scout report at 3.0T for a referring radiologist / radiographer.
Ground every claim in the provided Stage-A findings. Do NOT invent offsets,
metrics, or banding locations. Use Hz units. 3rd-person clinical style.
Follow the Stage-C section headings exactly. Max 400 words.
"""

_USER_TEMPLATE = """
=== PATIENT: {patient_id} ===

--- Stage-A Structured Findings (JSON) ---
{stage_a_json}

--- Retrieved Clinical Knowledge (excerpts) ---
{knowledge_text}

--- Evidence State Summary ---
{evidence_summary}

Task: Write a Stage-C clinical frequency-scout report with these sections:

1. CLINICAL QUESTION
2. REJECTED OFFSETS (hard veto)
3. MYOCARDIAL PASSBAND
4. PRIMARY RECOMMENDATION
5. ACCEPTABLE NEIGHBOURS (±25 Hz)
6. EVIDENCE RECONCILIATION
7. CONFIDENCE
"""


class StageBGenerator:
    """Stage-B LLM wrapper; Exp11 template implements the Stage-C report schema."""

    def __init__(
        self,
        model=None,
        processor=None,
        model_id: str = "google/medgemma-4b-it",
        temperature: float = 0.25,
        max_new_tokens: int = 512,
    ):
        self._model = model
        self._processor = processor
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        stage_a: Dict,
        state: EvidenceState,
        consistency: ConsistencyReport,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
    ) -> str:
        knowledge_text = self._format_knowledge(retrieved_chunks)
        evidence_summary = state.to_text_summary(max_chars_per_tool=600)

        if self._model is not None and self._model != "mock":
            narrative = self._llm_generate(
                stage_a, knowledge_text, evidence_summary, state.patient_id
            )
        else:
            narrative = self._template_narrative(stage_a, consistency)

        logger.info(
            "Stage-C narrative generated for %s (%d chars).",
            state.patient_id, len(narrative),
        )
        return narrative

    def _llm_generate(
        self,
        stage_a: Dict,
        knowledge_text: str,
        evidence_summary: str,
        patient_id: str,
    ) -> str:
        import torch

        prompt = _USER_TEMPLATE.format(
            patient_id=patient_id,
            stage_a_json=json.dumps(stage_a, indent=2)[:3000],
            knowledge_text=knowledge_text[:1500],
            evidence_summary=evidence_summary[:2000],
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            inputs = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            )
        except Exception:
            full_prompt = _SYSTEM_PROMPT + "\n\n" + prompt
            inputs = self._processor(text=full_prompt, return_tensors="pt")

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output = self._model.generate(**inputs, **gen_kwargs)
        text = self._processor.decode(output[0], skip_special_tokens=True)
        for marker in ["1. CLINICAL QUESTION", "CLINICAL QUESTION", "ASSESSMENT:"]:
            idx = text.find(marker)
            if idx != -1:
                text = text[idx:]
                break
        return text.strip()

    def _template_narrative(self, stage_a: Dict, consistency: ConsistencyReport) -> str:
        """Exp11 Stage-C radiologist report (deterministic template)."""
        rec = stage_a["recommended_offset"]
        slices = stage_a["per_slice"]
        neighbors = stage_a.get("acceptable_neighbors") or []
        rejected = stage_a.get("rejected_offsets") or [
            s for s in slices if s.get("verdict") == "reject"
        ]
        pb = stage_a.get("passband") or {}
        peak_hz = pb.get("peak_hz", rec.get("offset_hz"))
        peak_idx = pb.get("peak_idx", rec.get("slice_idx"))

        def hz_str(s):
            if isinstance(s, dict) and "offset_hz" in s:
                return f"{s['offset_hz']:+.0f} Hz"
            return f"{s:+.0f} Hz"

        def idx_str(s):
            return f"slice {s['slice_idx']} ({hz_str(s)})"

        # 1. Clinical question
        q1 = (
            "Frequency scout — select the RF offset that keeps the bSSFP null "
            "outside the LV myocardium and blood pool for subsequent CINE."
        )

        # 2. Rejected offsets
        if rejected:
            lines = []
            for s in rejected[:5]:
                reason = s.get("reason") or "banding enters the myocardium"
                lines.append(f"{idx_str(s)}: {reason}")
            q2 = "Hard veto — bSSFP null enters the cardiac ROI:\n  - " + "\n  - ".join(lines)
        else:
            q2 = "No offset showed a banding null overlapping myocardium or LV pool."

        # 3. Myocardial passband
        best = next((s for s in slices if s["slice_idx"] == rec["slice_idx"]), {})
        peak_slice = next((s for s in slices if s["slice_idx"] == peak_idx), best)
        q3 = (
            f"Passband peak at {peak_hz:+.0f} Hz (slice {peak_idx}). "
            f"At peak: banded fraction {peak_slice.get('banded_fraction_myo', 0):.1%}, "
            f"homogeneity {peak_slice.get('homogeneity', 0):.2f}, "
            f"CNR {peak_slice.get('cnr', 0):.1f}."
        )

        # 4. Primary recommendation
        rec_reason = rec.get("reason") or best.get("reason") or ""
        q4 = (
            f"Primary selection: {hz_str(rec)} (slice {rec['slice_idx']}). "
            f"{rec_reason} "
            f"Myocardial homogeneity {best.get('homogeneity', 0):.2f}, "
            f"CNR {best.get('cnr', 0):.1f}, "
            f"intra-myocardial banded fraction {best.get('banded_fraction_myo', 0):.1%}."
        )
        if rec["slice_idx"] != peak_idx:
            q4 += (
                f" Note: fused primary differs from passband peak ({peak_hz:+.0f} Hz); "
                f"peak remains listed as an acceptable neighbour when in-set."
            )

        # 5. Acceptable neighbours
        if neighbors:
            alt_lines = []
            for a in neighbors[:4]:
                rsn = a.get("reason") or "near-passband, no myocardial overlap"
                alt_lines.append(
                    f"{a['offset_hz']:+.0f} Hz ({a.get('verdict', 'acceptable')}): {rsn}"
                )
            q5 = (
                "Clinically acceptable neighbour(s) within the passband window "
                "(±1 scout step) — usable if primary is unavailable:\n  - "
                + "\n  - ".join(alt_lines)
            )
        else:
            q5 = "No other offsets fall in the passband acceptable window for this sweep."

        # 6. Evidence reconciliation
        neural = consistency.neural_top1
        cv = consistency.cv_top1
        vlm = consistency.vlm_top1
        offsets = [s["offset_hz"] for s in slices]
        def hz_i(i):
            if i is None or i < 0 or i >= len(offsets):
                return "?"
            return f"{offsets[i]:+.0f} Hz"

        parts = [
            f"Neural top-1 → {hz_i(neural)} (slice {neural})",
            f"CV (lowest banded_fraction_myo) → {hz_i(cv)} (slice {cv})",
        ]
        if vlm is not None:
            parts.append(f"VLM preferred → {hz_i(vlm)} (slice {vlm})")
        else:
            parts.append("VLM preferred → (not decisive / not run on enough slices)")
        parts.append(
            f"Selector (Exp10 fusion) → {hz_str(rec)}; "
            f"agreement={stage_a.get('evidence_agreement', 'unknown')}"
        )
        if consistency.disagreements:
            parts.append("Conflicts: " + "; ".join(consistency.disagreements))
        else:
            parts.append("No cross-source conflicts after CV metric aligned to banded_fraction_myo.")
        q6 = "\n  - ".join([""] + parts).lstrip("\n")

        # 7. Confidence
        basis = rec.get("confidence_basis") or (
            f"score {rec.get('confidence', consistency.confidence):.2f}"
        )
        q7 = (
            f"{rec.get('confidence_label', consistency.confidence_label).upper()} "
            f"(score {rec.get('confidence', consistency.confidence):.2f}). "
            f"Basis: {basis}. "
            f"Evidence sources: {', '.join(stage_a.get('evidence_sources', []))}."
        )

        return (
            f"1. CLINICAL QUESTION\n{q1}\n\n"
            f"2. REJECTED OFFSETS (hard veto)\n{q2}\n\n"
            f"3. MYOCARDIAL PASSBAND\n{q3}\n\n"
            f"4. PRIMARY RECOMMENDATION\n{q4}\n\n"
            f"5. ACCEPTABLE NEIGHBOURS (±25 Hz)\n{q5}\n\n"
            f"6. EVIDENCE RECONCILIATION\n{q6}\n\n"
            f"7. CONFIDENCE\n{q7}"
        )

    @staticmethod
    def _format_knowledge(chunks: Optional[List[RetrievedChunk]]) -> str:
        if not chunks:
            return "(No knowledge retrieved.)"
        parts = []
        for c in chunks[:5]:
            src = f"[{c.source} – {c.section}]" if c.section else f"[{c.source}]"
            parts.append(f"{src}\n{c.text}")
        return "\n\n".join(parts)
