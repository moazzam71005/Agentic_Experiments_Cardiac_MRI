"""
Stage B: Clinical Narrative Generator (T=0.25, grounded in Stage-A + retrieved knowledge)
------------------------------------------------------------------------------------------
Produces a clinical narrative in frequency-scouter language, grounded in:
  1. The Stage-A structured findings (JSON).
  2. Retrieved knowledge chunks (optional).
  3. The evidence state text summary.

The narrative follows the template from the clinical spec:
  - Describe the worst-offset(s) and where banding sits.
  - Walk through the optimal offset and explain why it's chosen.
  - Name the acceptable plateau.
  - Report confidence and evidence agreement.

If no LLM is available, produces a template-filled narrative from Stage-A data.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Any

from ..agent.evidence_state import EvidenceState
from ..agent.consistency import ConsistencyReport
from ..knowledge.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a specialist cardiac MRI physicist writing a clinical report
of a bSSFP frequency scout acquisition at 3.0T.
Write a concise but complete clinical narrative following the frequency-scouting vocabulary.
Ground every claim in the provided structured findings (Stage A) and knowledge base excerpts.
Do NOT introduce claims not supported by the evidence.
Write in 3rd-person clinical style. Use Hz units for offsets. Max 300 words.
"""

_USER_TEMPLATE = """
=== PATIENT: {patient_id} ===

--- Stage-A Structured Findings (JSON) ---
{stage_a_json}

--- Retrieved Clinical Knowledge (excerpts) ---
{knowledge_text}

--- Evidence State Summary ---
{evidence_summary}

Task: Write a clinical narrative frequency-scout report for this patient.
Format:
  ASSESSMENT:
  [2–4 sentences describing where banding is worst and why those offsets are rejected]

  OPTIMAL SELECTION:
  [2–3 sentences describing the recommended offset and the evidence supporting it]

  ALTERNATIVE OFFSETS:
  [1–2 sentences about acceptable alternatives, or "No acceptable alternatives identified."]

  CONFIDENCE:
  [One sentence with confidence level and basis]
"""


class StageBGenerator:

    def __init__(
        self,
        model=None,              # loaded HF model or None
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

    # ------------------------------------------------------------------

    def generate(
        self,
        stage_a: Dict,
        state: EvidenceState,
        consistency: ConsistencyReport,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
    ) -> str:
        """
        Generate the Stage-B narrative.

        Returns
        -------
        str : clinical narrative
        """
        knowledge_text = self._format_knowledge(retrieved_chunks)
        evidence_summary = state.to_text_summary(max_chars_per_tool=600)

        if self._model is not None and self._model != "mock":
            narrative = self._llm_generate(
                stage_a, knowledge_text, evidence_summary, state.patient_id
            )
        else:
            narrative = self._template_narrative(stage_a, consistency)

        logger.info("Stage-B narrative generated for %s (%d chars).", state.patient_id, len(narrative))
        return narrative

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

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
            stage_a_json=json.dumps(stage_a, indent=2)[:3000],  # cap to avoid context overflow
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
        # Strip echoed prompt
        for marker in ["ASSESSMENT:", "OPTIMAL SELECTION:"]:
            idx = text.find(marker)
            if idx != -1:
                text = text[idx:]
                break
        return text.strip()

    # ------------------------------------------------------------------
    # Template fallback (no LLM)
    # ------------------------------------------------------------------

    def _template_narrative(self, stage_a: Dict, consistency: ConsistencyReport) -> str:
        rec   = stage_a["recommended_offset"]
        slices = stage_a["per_slice"]
        neighbors = stage_a.get("acceptable_neighbors") or stage_a.get("alternative_offsets") or []
        rejected = stage_a.get("rejected_offsets") or [s for s in slices if s.get("verdict") == "reject"]
        pb = stage_a.get("passband") or {}

        def hz_str(s):
            if isinstance(s, dict) and "offset_hz" in s:
                return f"{s['offset_hz']:+.0f} Hz"
            return f"{s:+.0f} Hz"

        def idx_str(s):
            return f"slice {s['slice_idx']} ({hz_str(s)})"

        # Assessment: why rejected offsets are bad (physics reason)
        if rejected:
            lines = []
            for s in rejected[:4]:
                reason = s.get("reason") or "banding enters the myocardium"
                lines.append(f"{idx_str(s)}: {reason}")
            assessment = (
                "The following offsets are rejected because the bSSFP null enters "
                "the cardiac ROI:\n  - " + "\n  - ".join(lines)
            )
        else:
            assessment = "No offset showed a banding null overlapping myocardium or LV pool."

        best = next((s for s in slices if s["slice_idx"] == rec["slice_idx"]), {})
        rec_reason = rec.get("reason") or best.get("reason") or ""
        homog = best.get("homogeneity", 0)
        cnr   = best.get("cnr", 0)
        bfm   = best.get("banded_fraction_myo", 0)
        peak_hz = pb.get("peak_hz", rec.get("offset_hz"))

        optimal_block = (
            f"Primary selection: {hz_str(rec)} (slice {rec['slice_idx']}). "
            f"{rec_reason} "
            f"Myocardial homogeneity {homog:.2f}, CNR {cnr:.1f}, "
            f"intra-myocardial banded fraction {bfm:.1%}. "
            f"Myocardial passband peak is at {peak_hz:+.0f} Hz."
        )

        if neighbors:
            alt_lines = []
            for a in neighbors[:4]:
                rsn = a.get("reason") or "near-passband, no myocardial overlap"
                alt_lines.append(f"{a['offset_hz']:+.0f} Hz ({a.get('verdict', 'acceptable')}): {rsn}")
            alt_block = (
                "Clinically acceptable neighbour(s) — the referring doctor may use any of these "
                "if the primary offset is unavailable:\n  - " + "\n  - ".join(alt_lines)
            )
        else:
            alt_block = "No other acceptable neighbours were identified in this frequency sweep."

        conf_block = (
            f"Overall confidence: {rec.get('confidence_label', consistency.confidence_label)} "
            f"(score {rec.get('confidence', consistency.confidence):.2f}). "
            f"Evidence sources: {', '.join(stage_a.get('evidence_sources', []))}. "
            f"Agreement: {stage_a.get('evidence_agreement', 'unknown')}."
        )
        if consistency.disagreements:
            conf_block += f" Note: {'; '.join(consistency.disagreements)}"

        narrative = (
            f"ASSESSMENT:\n{assessment}\n\n"
            f"OPTIMAL SELECTION:\n{optimal_block}\n\n"
            f"ALTERNATIVE OFFSETS:\n{alt_block}\n\n"
            f"CONFIDENCE:\n{conf_block}"
        )
        return narrative

    @staticmethod
    def _format_knowledge(chunks: Optional[List[RetrievedChunk]]) -> str:
        if not chunks:
            return "(No knowledge retrieved.)"
        parts = []
        for c in chunks[:5]:
            src = f"[{c.source} – {c.section}]" if c.section else f"[{c.source}]"
            parts.append(f"{src}\n{c.text}")
        return "\n\n".join(parts)
