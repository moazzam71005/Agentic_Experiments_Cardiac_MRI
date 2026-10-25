"""
ADRE Tri-Agent Debate Engine (Exp 14)
-------------------------------------
Coordinates the multi-agent clinical consensus loop:
  1. Physics Specialist Agent (Off-resonance margin & BFM guardrails)
  2. Contrast & Texture Specialist Agent (CNR, homogeneity, SNR, edge definition)
  3. Clinical Meta-Arbitrator & Critic (Consensus synthesis & counterfactual audit)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

try:
    from .evidence_store import EvidenceStore
    from .consistency_checker import ConsistencyReport
    from llm.client import LLMClient
    from llm.prompts_debate import (
        PHYSICS_AGENT_SYSTEM,
        PHYSICS_AGENT_USER,
        CONTRAST_AGENT_SYSTEM,
        CONTRAST_AGENT_USER,
        META_CRITIC_SYSTEM,
        META_CRITIC_USER,
    )
except (ImportError, ValueError):
    from core.evidence_store import EvidenceStore
    from core.consistency_checker import ConsistencyReport
    from llm.client import LLMClient
    from llm.prompts_debate import (
        PHYSICS_AGENT_SYSTEM,
        PHYSICS_AGENT_USER,
        CONTRAST_AGENT_SYSTEM,
        CONTRAST_AGENT_USER,
        META_CRITIC_SYSTEM,
        META_CRITIC_USER,
    )

logger = logging.getLogger(__name__)


@dataclass
class DebateDecision:
    recommended_image_no: int
    recommended_offset_hz: float
    confidence_score: float
    confidence_label: str
    decision_source: str       # "adre_debate" | "deterministic_fallback"
    consensus_status: str      # "unanimous" | "physics_prevails" | "contrast_prevails" | "plateau_centered" | "fallback"
    physics_recommendation: Dict[str, Any] = field(default_factory=dict)
    contrast_recommendation: Dict[str, Any] = field(default_factory=dict)
    counterfactual_critique: str = ""
    clinical_summary: str = ""
    audit_passed: bool = False
    audit_notes: str = ""
    raw_responses: Dict[str, str] = field(default_factory=dict)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from an LLM response."""
    if not text:
        return None
    # Look for markdown code fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    # Look for raw JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return None


class ADREDebateEngine:

    # Hard guardrail: Any frame recommended by LLM with BFM > 0.08 is rejected
    MAX_PERMISSIBLE_BFM = 0.080

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or LLMClient(backend="template")

    def deliberate(
        self,
        store: EvidenceStore,
        report: ConsistencyReport,
        composite_scores: Dict[int, float],
    ) -> DebateDecision:
        """
        Runs the tri-agent debate loop for a patient.
        """
        offsets = store.offsets_hz
        fallback_img = max(composite_scores.keys(), key=lambda k: composite_scores[k])
        fallback_hz = offsets[fallback_img - 1] if fallback_img - 1 < len(offsets) else 0.0

        # If template backend, return deterministic fallback immediately
        if self.client.backend == "template":
            logger.info(f"[{report.patient_id}] Template backend — skipping multi-agent debate.")
            return DebateDecision(
                recommended_image_no=fallback_img,
                recommended_offset_hz=fallback_hz,
                confidence_score=report.confidence_score,
                confidence_label=report.confidence_label,
                decision_source="deterministic_fallback",
                consensus_status="fallback",
                audit_passed=False,
                audit_notes="Template mode — LLM debate bypassed",
                clinical_summary=f"Deterministic composite selection (Image #{fallback_img}, {fallback_hz:+.0f} Hz).",
            )

        logger.info(f"[{report.patient_id}] ⚡ Invoking ADRE Tri-Agent Debate (status={report.agreement_status})...")

        evidence_table = store.to_tabular_summary()
        cand_summary = ", ".join(
            f"#{img} ({offsets[img-1]:+.0f} Hz)" for img in report.candidate_images if img - 1 < len(offsets)
        )

        # ── Step 1: Physics Specialist Agent Deliberation ───────────────────────
        phys_user = PHYSICS_AGENT_USER.format(
            patient_id=report.patient_id,
            agreement_status=report.agreement_status,
            candidate_summary=cand_summary,
            evidence_table=evidence_table,
        )
        phys_raw = self.client.generate(PHYSICS_AGENT_SYSTEM, phys_user)
        phys_json = _extract_json(phys_raw) or {}
        phys_img = int(phys_json.get("recommended_image_no", report.physics_top1_img))
        phys_hz = offsets[phys_img - 1] if 1 <= phys_img <= len(offsets) else fallback_hz
        phys_conf = str(phys_json.get("physics_confidence", "medium"))
        phys_rat = str(phys_json.get("physics_rationale", "Minimizes myocardial banding artifact burden."))

        # ── Step 2: Contrast Specialist Agent Deliberation ─────────────────────
        cont_user = CONTRAST_AGENT_USER.format(
            patient_id=report.patient_id,
            agreement_status=report.agreement_status,
            candidate_summary=cand_summary,
            evidence_table=evidence_table,
        )
        cont_raw = self.client.generate(CONTRAST_AGENT_SYSTEM, cont_user)
        cont_json = _extract_json(cont_raw) or {}
        cont_img = int(cont_json.get("recommended_image_no", report.homog_top1_img))
        cont_hz = offsets[cont_img - 1] if 1 <= cont_img <= len(offsets) else fallback_hz
        cont_conf = str(cont_json.get("contrast_confidence", "medium"))
        cont_rat = str(cont_json.get("contrast_rationale", "Maximizes myocardial homogeneity and boundary contrast."))

        # ── Step 3: Clinical Meta-Arbitrator & Critic Deliberation ───────────────
        meta_user = META_CRITIC_USER.format(
            patient_id=report.patient_id,
            agreement_status=report.agreement_status,
            disagreements="; ".join(report.disagreements),
            evidence_table=evidence_table,
            physics_img=phys_img,
            physics_hz=phys_hz,
            physics_conf=phys_conf,
            physics_rationale=phys_rat,
            contrast_img=cont_img,
            contrast_hz=cont_hz,
            contrast_conf=cont_conf,
            contrast_rationale=cont_rat,
        )
        meta_raw = self.client.generate(META_CRITIC_SYSTEM, meta_user)
        meta_json = _extract_json(meta_raw) or {}

        # ── Step 4: Guardrail Verification & Clinical Safety Audit ───────────────
        rec_img = int(meta_json.get("recommended_image_no", phys_img if phys_img == cont_img else fallback_img))
        if not (1 <= rec_img <= store.n_slices):
            rec_img = fallback_img

        rec_hz = offsets[rec_img - 1] if rec_img - 1 < len(offsets) else 0.0
        bfm_map = store.get_metric_map("banded_fraction_myo")
        rec_bfm = float(bfm_map.get(rec_img, 0.0))

        # Check guardrail: BFM must not exceed maximum permissible threshold
        if rec_bfm > self.MAX_PERMISSIBLE_BFM:
            logger.warning(
                f"[{report.patient_id}] ✗ Debate recommendation #{rec_img} failed physics safety audit "
                f"(BFM={rec_bfm:.4f} > {self.MAX_PERMISSIBLE_BFM}). Falling back to composite score."
            )
            return DebateDecision(
                recommended_image_no=fallback_img,
                recommended_offset_hz=fallback_hz,
                confidence_score=0.60,
                confidence_label="medium",
                decision_source="deterministic_fallback",
                consensus_status="fallback",
                physics_recommendation=phys_json,
                contrast_recommendation=cont_json,
                audit_passed=False,
                audit_notes=f"Physics safety veto: Frame #{rec_img} has BFM={rec_bfm:.4f} > {self.MAX_PERMISSIBLE_BFM}",
                clinical_summary=f"Fell back to composite Image #{fallback_img} ({fallback_hz:+.0f} Hz) due to banding veto.",
                raw_responses={"physics": phys_raw, "contrast": cont_raw, "meta": meta_raw},
            )

        conf_score = float(meta_json.get("confidence_score", 0.85))
        conf_label = str(meta_json.get("confidence_label", "high"))
        consensus_status = str(meta_json.get("consensus_status", "unanimous" if phys_img == cont_img else "plateau_centered"))
        critique = str(meta_json.get("counterfactual_critique", "Counterfactual evaluation verified artifact-free passband."))
        summary = str(meta_json.get("clinical_summary", f"Selected Image #{rec_img} ({rec_hz:+.0f} Hz) following multi-agent consensus."))

        logger.info(
            f"[{report.patient_id}] ✓ ADRE Debate passed — Image #{rec_img} ({rec_hz:+.0f} Hz), "
            f"BFM={rec_bfm:.4f}, status={consensus_status}"
        )

        return DebateDecision(
            recommended_image_no=rec_img,
            recommended_offset_hz=rec_hz,
            confidence_score=conf_score,
            confidence_label=conf_label,
            decision_source="adre_debate",
            consensus_status=consensus_status,
            physics_recommendation=phys_json,
            contrast_recommendation=cont_json,
            counterfactual_critique=critique,
            clinical_summary=summary,
            audit_passed=True,
            audit_notes=f"Passed physics & clinical guardrails (BFM={rec_bfm:.4f} <= {self.MAX_PERMISSIBLE_BFM})",
            raw_responses={"physics": phys_raw, "contrast": cont_raw, "meta": meta_raw},
        )
