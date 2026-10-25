"""
LLM Arbitrator (Exp 13 — Active LLM Decision Loop)
---------------------------------------------------
Invoked ONLY when the ConsistencyChecker detects a 'divergent' or 'borderline_veto'
disagreement between tools. In those hard cases, this module:

  1. Serialises the full per-slice evidence table into a compact structured prompt.
  2. Queries the LLM to reason over the NUMBERS (not images — no VLM centre-bias).
  3. Parses the JSON response and runs a strict hallucination audit.
  4. Returns a validated ArbitrationResult or None (safe fallback to composite).

Design decisions:
  - temperature=0.0  → deterministic, reproducible across runs.
  - JSON-only output → strict schema enforced, easy to parse and audit.
  - Fallback on any parse/audit failure → never corrupts the pipeline.
  - Triggered only on divergent / borderline_veto → not called when tools agree
    (avoids slowing down confident cases and prevents over-reliance on LLM).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ArbitrationResult:
    """Validated output from the LLM Arbitrator."""
    recommended_image: int          # 1-indexed image number
    offset_hz: float                # e.g. +25.0
    confidence: str                 # "high" | "medium" | "low"
    rationale: str                  # one-sentence clinical justification
    audit_passed: bool              # whether the hallucination audit succeeded
    audit_notes: List[str]          # any audit warnings
    raw_response: str               # full LLM response text (for tracing)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

_ARBITRATOR_SYSTEM = """\
You are an expert cardiac MRI physicist and radiologist assistant. 
Your task is to resolve a disagreement between automated tools that are selecting 
the optimal radiofrequency offset for a 3.0T bSSFP cardiac frequency scout scan.

You will be given a table of quantitative metrics for all 13 frequency offset frames 
for one patient. You must reason carefully over the numbers and select the single best 
offset to use for the diagnostic cardiac CINE acquisition.

CRITICAL RULES:
1. The most important metric is banded_fraction_myo (BFM). A value > 0.08 means 
   the myocardium has >8% dark-band coverage — that frame should be REJECTED.
2. Lower BFM is always better. Homogeneity (1-CoV) should be as high as possible.
3. CNN score is a normalized deep-learning quality estimate (higher = better).
4. If multiple frames have similar low BFM, prefer the one with higher homogeneity.
5. Avoid extreme frequency offsets (±125 Hz, ±150 Hz) unless clearly superior.
6. You MUST respond with ONLY a single valid JSON object, nothing else.
"""

_ARBITRATOR_USER = """\
Patient: {patient_id}
Agreement Status: {agreement_status}

TOOL DISAGREEMENT SUMMARY:
- CNN favours: Image #{cnn_img} ({cnn_hz:+.0f} Hz)
- Physics/BFM favours: Image #{phys_img} ({phys_hz:+.0f} Hz)  
- Homogeneity favours: Image #{hom_img} ({hom_hz:+.0f} Hz)
- Disagreements: {disagreements}

FULL PER-SLICE EVIDENCE TABLE:
{evidence_table}

COMPOSITE SCORES (deterministic formula, for reference):
{composite_table}

TASK: Analyse the evidence and select the single best frequency offset.
Respond with ONLY this JSON object (no markdown, no explanation outside the JSON):
{{"recommended_image": <int 1-13>, "offset_hz": <float>, "confidence": "<high|medium|low>", "rationale": "<one concise sentence explaining your clinical reasoning>"}}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class LLMArbitrator:
    """
    Resolves tool disagreements by querying the LLM over numeric evidence.

    Parameters
    ----------
    llm_client : LLMClient
        A client from llm/client.py. Must have backend != "template" for
        real arbitration; falls back gracefully if template-only.
    min_bfm_reject_threshold : float
        Frames above this BFM are never recommended by the arbitrator (hard safety).
        Default 0.08 matches the existing Physics Veto threshold.
    """

    def __init__(self, llm_client, min_bfm_reject_threshold: float = 0.08):
        self.client = llm_client
        self.bfm_reject = min_bfm_reject_threshold

    # ──────────────────────────────────────────────────────────────────────────

    def arbitrate(
        self,
        store,                       # EvidenceStore
        consistency,                 # ConsistencyReport
        composite_scores: List[float],  # one per slice, from StageAGenerator
    ) -> Optional[ArbitrationResult]:
        """
        Main entry point. Returns ArbitrationResult if arbitration succeeds and
        audit passes, or None if LLM is unavailable / audit fails.
        """
        if self.client.backend == "template":
            logger.info("[Arbitrator] Template backend — skipping LLM arbitration.")
            return None

        evidence_table = self._build_evidence_table(store)
        composite_table = self._build_composite_table(store, composite_scores)

        disagreements_text = (
            "; ".join(consistency.disagreements)
            if consistency.disagreements
            else "Tools pick different offsets without clear consensus."
        )

        user_prompt = _ARBITRATOR_USER.format(
            patient_id=store.patient_id,
            agreement_status=consistency.agreement_status,
            cnn_img=consistency.cnn_top1_img,
            cnn_hz=consistency.cnn_top1_hz,
            phys_img=consistency.physics_top1_img,
            phys_hz=consistency.physics_top1_hz,
            hom_img=consistency.homog_top1_img,
            hom_hz=consistency.homog_top1_hz,
            disagreements=disagreements_text,
            evidence_table=evidence_table,
            composite_table=composite_table,
        )

        logger.info(
            "[Arbitrator] Invoking LLM for patient %s (status=%s)",
            store.patient_id, consistency.agreement_status,
        )

        raw = self.client.generate(
            system_prompt=_ARBITRATOR_SYSTEM,
            user_prompt=user_prompt,
        )

        if not raw or len(raw.strip()) < 10:
            logger.warning("[Arbitrator] Empty LLM response — falling back.")
            return None

        return self._parse_and_audit(raw, store, composite_scores)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_evidence_table(self, store) -> str:
        """Compact pipe-delimited table for LLM consumption."""
        bfm_map = store.get_metric_map("banded_fraction_myo")
        hom_map = store.get_metric_map("homogeneity")
        cnn_map = store.get_metric_map("cnn_score")
        cnr_map = store.get_metric_map("cnr")
        temp_map = store.get_metric_map("temporal_std_sweep")
        sec_map = store.get_metric_map("band_sectors")

        header = "Img | Offset(Hz) | BFM     | Homogeneity | CNN     | CNR   | TempStd | Sectors"
        sep    = "----|------------|---------|-------------|---------|-------|---------|--------"
        rows = [header, sep]

        for i in range(store.n_slices):
            img_no = i + 1
            hz = store.offsets_hz[i] if i < len(store.offsets_hz) else 0.0
            bfm = bfm_map.get(img_no, 0.0)
            hom = hom_map.get(img_no, 0.0)
            cnn = cnn_map.get(img_no, 0.0)
            cnr = cnr_map.get(img_no, 0.0)
            tmp = temp_map.get(img_no, 0.0)
            sec = str(sec_map.get(img_no, "none")).strip()
            if not sec or sec.lower() in ("nan", "none", "[]"):
                sec = "none"
            reject_flag = " ← REJECT" if bfm > self.bfm_reject else ""
            rows.append(
                f" #{img_no:<2d}| {hz:+9.1f} | {bfm:.5f} | {hom:.5f}    | "
                f"{cnn:.5f} | {cnr:5.2f} | {tmp:.5f} | {sec}{reject_flag}"
            )

        return "\n".join(rows)

    def _build_composite_table(self, store, composite_scores: List[float]) -> str:
        """Short ranked table of composite scores."""
        indexed = sorted(
            enumerate(composite_scores), key=lambda x: x[1], reverse=True
        )
        lines = ["Rank | Img | Offset(Hz) | Composite Score"]
        lines.append("-----|-----|------------|----------------")
        for rank, (idx, sc) in enumerate(indexed[:6], start=1):
            img_no = idx + 1
            hz = store.offsets_hz[idx] if idx < len(store.offsets_hz) else 0.0
            lines.append(f"  {rank}  | #{img_no:<2d}| {hz:+9.1f} | {sc:.6f}")
        return "\n".join(lines)

    def _parse_and_audit(
        self,
        raw: str,
        store,
        composite_scores: List[float],
    ) -> Optional[ArbitrationResult]:
        """Parse LLM JSON output and run hallucination audit."""
        audit_notes: List[str] = []

        # 1. Extract JSON from raw text (LLMs sometimes add markdown fences)
        json_match = re.search(r'\{[^{}]*"recommended_image"[^{}]*\}', raw, re.DOTALL)
        if not json_match:
            logger.warning("[Arbitrator] Could not extract JSON from LLM response.")
            return None

        try:
            parsed: Dict[str, Any] = json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            logger.warning("[Arbitrator] JSON parse error: %s", exc)
            return None

        # 2. Validate required fields
        required = {"recommended_image", "offset_hz", "confidence", "rationale"}
        missing = required - parsed.keys()
        if missing:
            logger.warning("[Arbitrator] Missing fields in LLM JSON: %s", missing)
            return None

        # 3. Type-check and range-check recommended_image
        try:
            rec_img = int(parsed["recommended_image"])
            rec_hz = float(parsed["offset_hz"])
            confidence = str(parsed["confidence"]).lower()
            rationale = str(parsed["rationale"])
        except (TypeError, ValueError) as exc:
            logger.warning("[Arbitrator] Type error in JSON fields: %s", exc)
            return None

        if not (1 <= rec_img <= store.n_slices):
            audit_notes.append(
                f"Hallucination: recommended_image={rec_img} out of range [1,{store.n_slices}]"
            )
            logger.warning("[Arbitrator] Audit FAIL — image number out of range.")
            return ArbitrationResult(
                recommended_image=rec_img, offset_hz=rec_hz,
                confidence=confidence, rationale=rationale,
                audit_passed=False, audit_notes=audit_notes, raw_response=raw,
            )

        # 4. Verify offset_hz matches known offset for that image
        expected_hz = store.offsets_hz[rec_img - 1] if rec_img - 1 < len(store.offsets_hz) else None
        if expected_hz is not None and abs(rec_hz - expected_hz) > 1.0:
            audit_notes.append(
                f"Hz mismatch: LLM said {rec_hz} Hz for image #{rec_img}, "
                f"but actual offset is {expected_hz} Hz. Correcting."
            )
            rec_hz = expected_hz  # silently correct — the image number is authoritative

        # 5. Hard safety check: never accept a frame with BFM > threshold
        bfm_map = store.get_metric_map("banded_fraction_myo")
        rec_bfm = bfm_map.get(rec_img, 0.0)
        if rec_bfm > self.bfm_reject:
            audit_notes.append(
                f"Safety veto: LLM recommended image #{rec_img} has BFM={rec_bfm:.3f} "
                f"> threshold {self.bfm_reject}. Arbitration rejected."
            )
            logger.warning(
                "[Arbitrator] Safety veto — LLM picked a heavily banded frame. Falling back."
            )
            return ArbitrationResult(
                recommended_image=rec_img, offset_hz=rec_hz,
                confidence=confidence, rationale=rationale,
                audit_passed=False, audit_notes=audit_notes, raw_response=raw,
            )

        # 6. Validate confidence label
        if confidence not in ("high", "medium", "low"):
            audit_notes.append(f"Non-standard confidence label '{confidence}' → normalised to 'medium'.")
            confidence = "medium"

        logger.info(
            "[Arbitrator] ✓ Audit passed — Image #%d (%+.0f Hz), BFM=%.4f, conf=%s",
            rec_img, rec_hz, rec_bfm, confidence,
        )

        return ArbitrationResult(
            recommended_image=rec_img,
            offset_hz=rec_hz,
            confidence=confidence,
            rationale=rationale,
            audit_passed=True,
            audit_notes=audit_notes,
            raw_response=raw,
        )
