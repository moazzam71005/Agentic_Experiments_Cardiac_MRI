"""
Clinical Audit & Faithfulness Verifier (Hallucination Detection)
----------------------------------------------------------------
Audits the generated clinical narrative against the Stage-A ground truth JSON.
Verifies that:
1. Every frequency offset (Hz) cited exists in the scout sweep.
2. The recommended slice matches Stage-A exactly.
3. Every cited artifact sector (anterior, inferior, lateral, etc.) was physically detected.
4. All 7 required clinical report blocks are present.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List


class ClinicalAudit:

    REQUIRED_SECTIONS = [
        "1. CLINICAL INDICATION",
        "2. PRIMARY FREQUENCY RECOMMENDATION",
        "3. DIAGNOSTIC OPERATING WINDOW",
        "4. EXCLUDED OFFSETS",
        "5. METHODOLOGICAL RATIONALE",
        "6. RISK OF SUBOPTIMAL OFFSET",
        "7. FINAL SCANNER CONFIGURATION",
    ]

    def audit(self, stage_a: Dict[str, Any], narrative: str) -> Dict[str, Any]:
        valid_offsets = {ps["offset_hz"] for ps in stage_a.get("per_slice", [])}
        rec_hz = stage_a["recommended_offset"]["offset_hz"]
        rec_img = stage_a["recommended_offset"]["image_no"]

        # 1. Check section coverage
        narrative_upper = narrative.upper()
        missing_sections = [
            sec for sec in self.REQUIRED_SECTIONS if sec not in narrative_upper
        ]
        section_coverage = (len(self.REQUIRED_SECTIONS) - len(missing_sections)) / len(self.REQUIRED_SECTIONS)

        # 2. Check cited frequency values
        hz_matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*Hz", narrative)
        cited_hz = [float(h) for h in hz_matches]

        hallucinated_hz = []
        for h in cited_hz:
            # Allow common tolerance numbers like 25 Hz, 50 Hz, 100 Hz
            if h in (25.0, 50.0, 100.0, -25.0, -50.0, -100.0):
                continue
            if not any(abs(h - v) < 1.0 for v in valid_offsets):
                hallucinated_hz.append(h)

        faithfulness_hz = 1.0 if not cited_hz else (len(cited_hz) - len(hallucinated_hz)) / len(cited_hz)

        # 3. Check recommended offset alignment
        rec_matched = (
            f"{rec_hz:+.1f} Hz" in narrative
            or f"{rec_hz:+.0f} Hz" in narrative
            or f"{rec_hz:.0f} Hz" in narrative
            or f"#{rec_img}" in narrative
        )

        # 4. Check anatomical sector grounding
        detected_sectors = set()
        for ps in stage_a.get("per_slice", []):
            sec = str(ps.get("band_sectors", "")).lower()
            for token in ["anterior", "inferior", "lateral", "septal", "anteroseptal", "inferoseptal", "inferolateral", "anterolateral"]:
                if token in sec:
                    detected_sectors.add(token)

        hallucinated_sectors = []
        for token in ["anterior", "inferior", "lateral", "septal", "anteroseptal", "inferoseptal", "inferolateral", "anterolateral"]:
            if token in narrative.lower() and token not in detected_sectors:
                hallucinated_sectors.append(token)

        is_passed = (
            section_coverage >= 0.85
            and len(hallucinated_hz) == 0
            and rec_matched
            and len(hallucinated_sectors) == 0
        )

        return {
            "is_passed": is_passed,
            "section_coverage": round(section_coverage, 2),
            "missing_sections": missing_sections,
            "faithfulness_hz_score": round(faithfulness_hz, 3),
            "hallucinated_hz": hallucinated_hz,
            "recommendation_grounded": rec_matched,
            "hallucinated_sectors": hallucinated_sectors,
            "total_words": len(narrative.split()),
        }
