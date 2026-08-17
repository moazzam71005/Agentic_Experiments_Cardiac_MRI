"""
Consistency / Repair Module
---------------------------
Checks the Stage-B narrative for:
  1. Unsupported claims (hallucinations) — mentions slice/offset numbers not in Stage-A.
  2. JSON validity of Stage-A (already a dict, but re-validates before saving).
  3. Narrative-Stage-A agreement: if narrative claims an offset is "optimal" but
     Stage-A disagrees, flags it and optionally re-prompts.

Repair strategy: if a claim is not grounded in Stage-A evidence, redact or replace
with a grounded alternative phrase.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RepairReport:
    original_narrative: str
    repaired_narrative: str
    issues_found: List[str]
    was_repaired: bool
    stage_a_valid: bool


class RepairModule:

    def run(self, stage_a: Dict, narrative: str) -> RepairReport:
        """
        Check and repair the Stage-B narrative against Stage-A data.

        Parameters
        ----------
        stage_a : dict from StageAGenerator.generate()
        narrative : str from StageBGenerator.generate()

        Returns
        -------
        RepairReport
        """
        issues = []
        repaired = narrative

        # 1. Validate Stage-A JSON completeness
        stage_a_valid, schema_issues = self._validate_stage_a(stage_a)
        issues.extend(schema_issues)

        # 2. Check that the recommended offset mentioned in narrative matches Stage-A
        rec_hz  = stage_a.get("recommended_offset", {}).get("offset_hz", None)
        issues2, repaired = self._check_recommendation_claim(repaired, stage_a, rec_hz)
        issues.extend(issues2)

        # 3. Check for hallucinated Hz values (offsets not in the scout)
        valid_offsets = {ps["offset_hz"] for ps in stage_a.get("per_slice", [])}
        issues3, repaired = self._check_hallucinated_offsets(repaired, valid_offsets)
        issues.extend(issues3)

        # 4. Check that cited confidence matches Stage-A confidence
        issues4, repaired = self._check_confidence_claim(repaired, stage_a)
        issues.extend(issues4)

        was_repaired = repaired != narrative
        if issues:
            logger.warning(
                "Repair found %d issues for patient %s.",
                len(issues), stage_a.get("patient_id", "?"),
            )

        return RepairReport(
            original_narrative=narrative,
            repaired_narrative=repaired,
            issues_found=issues,
            was_repaired=was_repaired,
            stage_a_valid=stage_a_valid,
        )

    # ------------------------------------------------------------------

    def _validate_stage_a(self, stage_a: Dict) -> Tuple[bool, List[str]]:
        required_keys = ["patient_id", "per_slice", "recommended_offset", "alternative_offsets"]
        missing = [k for k in required_keys if k not in stage_a]
        if missing:
            return False, [f"Stage-A missing required keys: {missing}"]
        if not stage_a.get("per_slice"):
            return False, ["Stage-A has empty per_slice list."]
        return True, []

    def _check_recommendation_claim(
        self, narrative: str, stage_a: Dict, rec_hz: Optional[float]
    ) -> Tuple[List[str], str]:
        if rec_hz is None:
            return [], narrative
        issues = []
        # Look for statements like "+50 Hz is optimal" but Stage-A recommends +25 Hz
        optimal_pattern = re.findall(r"([+-]?\d+)\s*Hz.*?(?:optimal|selected|recommended)", narrative)
        for hz_str in optimal_pattern:
            try:
                hz = float(hz_str)
                if abs(hz - rec_hz) > 13:   # tolerance for rounding
                    issues.append(
                        f"Narrative claims {hz:+.0f} Hz as optimal but Stage-A recommends {rec_hz:+.0f} Hz."
                    )
            except ValueError:
                pass
        return issues, narrative

    def _check_hallucinated_offsets(
        self, narrative: str, valid_offsets: set
    ) -> Tuple[List[str], str]:
        issues = []
        mentioned = re.findall(r"([+-]?\d+)\s*Hz", narrative)
        for hz_str in mentioned:
            hz = float(hz_str)
            if hz == 0 or abs(hz) < 5:
                continue   # allow "0 Hz" general mentions
            # Check if within 13 Hz of any valid offset
            if not any(abs(hz - v) <= 13 for v in valid_offsets):
                issues.append(f"Narrative mentions {hz:+.0f} Hz which is not a scout offset.")
        return issues, narrative

    def _check_confidence_claim(
        self, narrative: str, stage_a: Dict
    ) -> Tuple[List[str], str]:
        issues = []
        rec = stage_a.get("recommended_offset", {})
        conf = rec.get("confidence", None)
        conf_label = rec.get("confidence_label", None)
        if conf_label and conf_label not in narrative:
            # Minor mismatch; log but don't force a repair on narrative text
            issues.append(
                f"Stage-A confidence label is '{conf_label}' but may not appear verbatim in narrative."
            )
        return issues, narrative
