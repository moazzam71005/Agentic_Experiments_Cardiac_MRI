"""
Reasoning Quality Metrics (GT-ready stubs)
------------------------------------------
These metrics require the doctor's reasoning ground truth (coming later).
Hooks are provided now; they gracefully degrade when GT is absent.

Implemented now (no GT needed):
  - Faithfulness check : all Hz values in narrative traceable to Stage-A.
  - Source citation    : does narrative reference at least one knowledge source?
  - Checklist coverage : fraction of checklist items addressed in narrative.

Stubbed for GT (when doctor notes arrive):
  - BERTScore          : semantic similarity vs physician narrative.
  - METEOR             : n-gram overlap metric.
  - Clinical Likert    : placeholder for manual rating 1-5.
  - Wilcoxon sig. test : statistical significance vs baseline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    patient_id: str
    faithfulness_score: float       # 0–1: fraction of Hz claims traceable to Stage-A
    has_source_citation: bool       # Narrative mentions a knowledge concept
    checklist_coverage: float       # 0–1: fraction of key items mentioned
    narrative_length: int           # word count

    # GT-ready stubs (None until doctor GT arrives)
    bertscore_f1: Optional[float] = None
    meteor: Optional[float] = None
    likert_clinical: Optional[float] = None


class ReasoningMetrics:

    _KEY_CONCEPTS = [
        "banding", "artifact", "myocardium", "offset", "optimal",
        "homogeneity", "cnr", "confidence", "acceptable",
    ]

    def evaluate(
        self,
        narrative: str,
        stage_a: Dict,
        doctor_gt: Optional[str] = None,
        patient_id: str = "unknown",
    ) -> ReasoningResult:

        faithfulness = self._faithfulness(narrative, stage_a)
        has_citation = self._has_source_citation(narrative)
        checklist_cov= self._checklist_coverage(narrative)
        word_count   = len(narrative.split())

        result = ReasoningResult(
            patient_id=patient_id,
            faithfulness_score=faithfulness,
            has_source_citation=has_citation,
            checklist_coverage=checklist_cov,
            narrative_length=word_count,
        )

        if doctor_gt is not None:
            result.bertscore_f1 = self._bertscore(narrative, doctor_gt)
            result.meteor        = self._meteor(narrative, doctor_gt)

        return result

    # ------------------------------------------------------------------

    def _faithfulness(self, narrative: str, stage_a: Dict) -> float:
        """Fraction of Hz values mentioned in narrative that appear in Stage-A."""
        valid_hz = {ps["offset_hz"] for ps in stage_a.get("per_slice", [])}
        mentions = re.findall(r"([+-]?\d+)\s*Hz", narrative)
        if not mentions:
            return 1.0  # no claims to check
        grounded = sum(
            1 for h in mentions
            if any(abs(float(h) - v) <= 13 for v in valid_hz) or abs(float(h)) < 5
        )
        return grounded / len(mentions)

    def _has_source_citation(self, narrative: str) -> bool:
        """Simple check: does the narrative mention any clinical concept?"""
        text_lower = narrative.lower()
        return any(c in text_lower for c in self._KEY_CONCEPTS)

    def _checklist_coverage(self, narrative: str) -> float:
        """Fraction of key reporting items present in narrative."""
        items = [
            "banding", "myocardium", "optimal", "confidence",
            "acceptable", "offset", "hz",
        ]
        text_lower = narrative.lower()
        covered = sum(1 for item in items if item in text_lower)
        return covered / len(items)

    # ------------------------------------------------------------------
    # GT stubs

    def _bertscore(self, pred: str, ref: str) -> float:
        try:
            from bert_score import score as bs_score
            P, R, F1 = bs_score([pred], [ref], lang="en", verbose=False)
            return float(F1.mean())
        except ImportError:
            logger.warning("bert_score not installed; BERTScore skipped.")
            return 0.0

    def _meteor(self, pred: str, ref: str) -> float:
        try:
            from nltk.translate.meteor_score import meteor_score
            from nltk.tokenize import word_tokenize
            return float(meteor_score([word_tokenize(ref)], word_tokenize(pred)))
        except ImportError:
            logger.warning("NLTK/meteor_score not installed; METEOR skipped.")
            return 0.0

    def print_summary(self, results: List[ReasoningResult]):
        import numpy as np
        if not results:
            print("No reasoning results to summarise.")
            return
        print("\n" + "=" * 50)
        print("  REASONING QUALITY METRICS")
        print("=" * 50)
        print(f"  Patients           : {len(results)}")
        print(f"  Faithfulness       : {np.mean([r.faithfulness_score for r in results]):.3f}")
        print(f"  Source citation    : {sum(r.has_source_citation for r in results)}/{len(results)}")
        print(f"  Checklist coverage : {np.mean([r.checklist_coverage for r in results]):.3f}")
        print(f"  Narrative length   : {np.mean([r.narrative_length for r in results]):.0f} words")

        bert_vals = [r.bertscore_f1 for r in results if r.bertscore_f1 is not None]
        if bert_vals:
            print(f"  BERTScore F1       : {np.mean(bert_vals):.3f}")
        else:
            print("  BERTScore F1       : (pending doctor GT)")
        print("=" * 50)
