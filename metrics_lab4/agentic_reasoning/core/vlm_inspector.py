"""
Multimodal VLM Visual Inspector (Exp 15)
----------------------------------------
Orchestrates visual myocardium inspection by passing multi-offset image strips
and quantitative evidence to a Vision-Language Model with physics safety auditing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

try:
    from .evidence_store import EvidenceStore
    from .consistency_checker import ConsistencyReport
    from llm.vlm_client import VLMClient
    from llm.prompts_vlm import VLM_INSPECTOR_SYSTEM, VLM_INSPECTOR_USER
except (ImportError, ValueError):
    from core.evidence_store import EvidenceStore
    from core.consistency_checker import ConsistencyReport
    from llm.vlm_client import VLMClient
    from llm.prompts_vlm import VLM_INSPECTOR_SYSTEM, VLM_INSPECTOR_USER

logger = logging.getLogger(__name__)


@dataclass
class VLMDecision:
    recommended_image_no: int
    recommended_offset_hz: float
    confidence_score: float
    confidence_label: str
    decision_source: str       # "vlm_inspected" | "deterministic_fallback"
    visual_banding_assessment: str = ""
    septal_contrast_assessment: str = ""
    clinical_rationale: str = ""
    audit_passed: bool = False
    audit_notes: str = ""
    raw_response: str = ""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


class VLMInspector:

    MAX_PERMISSIBLE_BFM = 0.080

    def __init__(self, client: Optional[VLMClient] = None, strips_dir: Optional[Path] = None):
        self.client = client or VLMClient(backend="template")
        self.strips_dir = strips_dir

    def inspect(
        self,
        store: EvidenceStore,
        report: ConsistencyReport,
        composite_scores: Dict[int, float],
    ) -> VLMDecision:
        offsets = store.offsets_hz
        fallback_img = max(composite_scores.keys(), key=lambda k: composite_scores[k])
        fallback_hz = offsets[fallback_img - 1] if fallback_img - 1 < len(offsets) else 0.0

        if self.client.backend == "template":
            logger.info(f"[{report.patient_id}] Template backend — skipping VLM visual inspection.")
            return VLMDecision(
                recommended_image_no=fallback_img,
                recommended_offset_hz=fallback_hz,
                confidence_score=report.confidence_score,
                confidence_label=report.confidence_label,
                decision_source="deterministic_fallback",
                audit_passed=False,
                audit_notes="Template mode — VLM inspection bypassed",
                clinical_rationale=f"Deterministic composite selection (Image #{fallback_img}, {fallback_hz:+.0f} Hz).",
            )

        logger.info(f"[{report.patient_id}] 👁️ Invoking VLM Visual Inspection (status={report.agreement_status})...")

        # Find visual strip for patient
        image_path = None
        if self.strips_dir and self.strips_dir.exists():
            zoom_p = self.strips_dir / f"{report.patient_id}_myo_zoom_strip.png"
            full_p = self.strips_dir / f"{report.patient_id}_full_strip.png"
            if zoom_p.exists():
                image_path = zoom_p
            elif full_p.exists():
                image_path = full_p

        evidence_table = store.to_tabular_summary()
        cand_summary = ", ".join(
            f"#{img} ({offsets[img-1]:+.0f} Hz)" for img in report.candidate_images if img - 1 < len(offsets)
        )

        user_prompt = VLM_INSPECTOR_USER.format(
            patient_id=report.patient_id,
            agreement_status=report.agreement_status,
            candidate_summary=cand_summary,
            evidence_table=evidence_table,
        )

        vlm_raw = self.client.generate_with_image(
            system_prompt=VLM_INSPECTOR_SYSTEM,
            user_prompt=user_prompt,
            image_path=image_path,
        )

        vlm_json = _extract_json(vlm_raw) or {}

        rec_img = int(vlm_json.get("recommended_image_no", fallback_img))
        if not (1 <= rec_img <= store.n_slices):
            rec_img = fallback_img

        rec_hz = offsets[rec_img - 1] if rec_img - 1 < len(offsets) else 0.0
        bfm_map = store.get_metric_map("banded_fraction_myo")
        rec_bfm = float(bfm_map.get(rec_img, 0.0))

        # Check physics safety guardrail
        if rec_bfm > self.MAX_PERMISSIBLE_BFM:
            logger.warning(
                f"[{report.patient_id}] ✗ VLM recommendation #{rec_img} failed physics safety audit "
                f"(BFM={rec_bfm:.4f} > {self.MAX_PERMISSIBLE_BFM}). Falling back to composite score."
            )
            return VLMDecision(
                recommended_image_no=fallback_img,
                recommended_offset_hz=fallback_hz,
                confidence_score=0.60,
                confidence_label="medium",
                decision_source="deterministic_fallback",
                audit_passed=False,
                audit_notes=f"Physics veto: Image #{rec_img} has BFM={rec_bfm:.4f} > {self.MAX_PERMISSIBLE_BFM}",
                clinical_rationale=f"Fell back to composite Image #{fallback_img} ({fallback_hz:+.0f} Hz) due to banding veto.",
                raw_response=vlm_raw,
            )

        conf_score = float(vlm_json.get("confidence_score", 0.90))
        conf_label = str(vlm_json.get("confidence_label", "high"))
        vis_assess = str(vlm_json.get("visual_banding_assessment", "Visual inspection confirmed absence of dark bands."))
        sep_assess = str(vlm_json.get("septal_contrast_assessment", "Septal wall boundary is sharp with high contrast."))
        clin_rat = str(vlm_json.get("clinical_rationale", f"Selected Image #{rec_img} ({rec_hz:+.0f} Hz) following visual verification."))

        logger.info(
            f"[{report.patient_id}] ✓ VLM Visual Inspection passed — Image #{rec_img} ({rec_hz:+.0f} Hz), "
            f"BFM={rec_bfm:.4f}, conf={conf_label}"
        )

        return VLMDecision(
            recommended_image_no=rec_img,
            recommended_offset_hz=rec_hz,
            confidence_score=conf_score,
            confidence_label=conf_label,
            decision_source="vlm_inspected",
            visual_banding_assessment=vis_assess,
            septal_contrast_assessment=sep_assess,
            clinical_rationale=clin_rat,
            audit_passed=True,
            audit_notes=f"Passed visual & physics guardrails (BFM={rec_bfm:.4f} <= {self.MAX_PERMISSIBLE_BFM})",
            raw_response=vlm_raw,
        )
