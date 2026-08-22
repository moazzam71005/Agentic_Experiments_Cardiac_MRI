"""
Consistency Checker
-------------------
Cross-checks evidence from the neural ranker, CV tools (banding/quality), and VLM
to compute:
  - per-slice agreement flags
  - overall confidence = weighted concordance across evidence sources
  - disagreement flags that may trigger additional tool calls

Agreement logic:
  - Neural top-1 vs CV top-1 (lowest severity + highest homogeneity): agree/disagree
  - Neural top-1 vs VLM "optimal" verdict: agree/disagree
  - CV vs VLM: band detection agreement per slice
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .evidence_state import EvidenceState, EvidencePacket

logger = logging.getLogger(__name__)

TOOL_NAME = "consistency"


@dataclass
class ConsistencyReport:
    patient_id: str
    neural_top1: int
    cv_top1: int                        # Index of best slice by CV metrics
    vlm_top1: Optional[int]             # Index of slice rated "optimal" by VLM

    neural_cv_agree: bool
    neural_vlm_agree: bool
    cv_vlm_agree: bool

    # Per-slice agreement matrix
    per_slice_agreement: List[Dict]     # [{slice_idx, neural_ok, cv_ok, vlm_ok, agreed}]

    confidence: float                   # 0–1 overall trust score
    confidence_label: str               # "high" | "medium" | "low"
    disagreements: List[str]            # Human-readable conflict descriptions

    provenance: Dict = field(default_factory=dict)


class ConsistencyChecker:

    TOOL_NAME = TOOL_NAME

    # Weights for concordance: how much does each pair contribute?
    W_NEURAL_CV  = 0.4
    W_NEURAL_VLM = 0.35
    W_CV_VLM     = 0.25

    def run(self, state: EvidenceState) -> ConsistencyReport:
        neural_pkt  = self._get_neural(state)
        banding_pkt = self._get_banding(state)
        quality_pkt = self._get_quality(state)
        vlm_pkt     = self._get_vlm(state)

        neural_top1 = self._neural_top1(neural_pkt)
        cv_top1     = self._cv_top1(banding_pkt, quality_pkt, state.n_slices)
        vlm_top1    = self._vlm_top1(vlm_pkt)

        neural_cv_agree  = (neural_top1 == cv_top1)
        neural_vlm_agree = (vlm_top1 is None) or (neural_top1 == vlm_top1)
        cv_vlm_agree     = (vlm_top1 is None) or (cv_top1 == vlm_top1)

        # Per-slice agreement
        per_slice = self._per_slice_agreement(
            neural_pkt, banding_pkt, quality_pkt, vlm_pkt, state.n_slices
        )

        # --- Margin-based confidence (Exp 8 fix) ---
        # Use score margin between top-1 and top-2 from the neural ranker as the
        # primary signal (honest — not inflated by VLM rubber-stamping).
        # Cross-source agreement modulates but does not dominate.
        neural_margin = self._neural_score_margin(neural_pkt)
        # Concordance — without the "all-agree" boost that was causing VLM to
        # inflate confidence even when it labelled every slice "optimal".
        concordance = (
            self.W_NEURAL_CV  * float(neural_cv_agree)
            + self.W_NEURAL_VLM * float(neural_vlm_agree)
            + self.W_CV_VLM     * float(cv_vlm_agree)
        )
        # Confidence = 60% margin signal + 40% concordance signal.
        # VLM penalty only applies when VLM is active (has results) but fails to
        # discriminate. If VLM returned no results at all (disabled / W_VLM=0)
        # we skip the penalty — don't punish the system for not running VLM.
        vlm_discriminates = self._vlm_discriminates(vlm_pkt)
        vlm_penalty = 0.0
        if vlm_pkt:  # VLM was actually run
            vlm_penalty = 0.0 if vlm_discriminates else 0.08
        confidence = float(
            0.60 * min(neural_margin * 4.0, 1.0)   # scale: margin 0.05 → 0.20, 0.25 → 1.0
            + 0.40 * concordance
        ) - vlm_penalty
        confidence = max(0.10, min(confidence, 1.0))
        confidence_label = "high" if confidence >= 0.65 else ("medium" if confidence >= 0.35 else "low")

        disagreements = self._collect_disagreements(
            neural_top1, cv_top1, vlm_top1,
            neural_cv_agree, neural_vlm_agree, cv_vlm_agree, state,
        )

        # Register as evidence
        pkt = EvidencePacket(
            patient_id=state.patient_id,
            slice_idx=-1,
            offset_hz=0.0,
            tool=TOOL_NAME,
            iteration=0,
            data=None,
            summary=(
                f"Neural top-1={neural_top1} | CV top-1={cv_top1} | VLM top-1={vlm_top1} "
                f"| confidence={confidence:.2f} ({confidence_label})"
            ),
        )
        state.add(pkt)

        report = ConsistencyReport(
            patient_id=state.patient_id,
            neural_top1=neural_top1,
            cv_top1=cv_top1,
            vlm_top1=vlm_top1,
            neural_cv_agree=neural_cv_agree,
            neural_vlm_agree=neural_vlm_agree,
            cv_vlm_agree=cv_vlm_agree,
            per_slice_agreement=per_slice,
            confidence=confidence,
            confidence_label=confidence_label,
            disagreements=disagreements,
            provenance={"tool": TOOL_NAME},
        )
        logger.info(
            "Consistency[%s]: neural=%d cv=%d vlm=%s conf=%.2f (%s) | %d disagreements",
            state.patient_id, neural_top1, cv_top1, vlm_top1,
            confidence, confidence_label, len(disagreements),
        )
        return report

    # ------------------------------------------------------------------
    # Evidence extraction helpers
    # ------------------------------------------------------------------

    def _get_neural(self, state: EvidenceState):
        pkts = state.get_for_tool("neural")
        return pkts[0].data if pkts else None

    def _get_banding(self, state: EvidenceState) -> List:
        return [p.data for p in state.get_for_tool("banding")]

    def _get_quality(self, state: EvidenceState) -> List:
        return [p.data for p in state.get_for_tool("quality")]

    def _get_vlm(self, state: EvidenceState) -> List:
        return [p.data for p in state.get_for_tool("vlm")]

    # ------------------------------------------------------------------
    # Top-1 from each source
    # ------------------------------------------------------------------

    def _neural_score_margin(self, neural) -> float:
        """
        Softmax probability gap between top-1 and top-2 slice.
        Returns 0.0 if not available or only one slice.
        """
        if neural is None:
            return 0.0
        conf = list(getattr(neural, "softmax_confidence", []))
        if len(conf) < 2:
            return 0.0
        sorted_conf = sorted(conf, reverse=True)
        return float(sorted_conf[0] - sorted_conf[1])

    def _vlm_discriminates(self, vlm_results: List) -> bool:
        """
        Returns True if the VLM assigned at least 2 different verdicts across
        the slices it processed.  If all are "optimal" (or all the same), the
        VLM is not providing useful discrimination — treat as uninformative.
        """
        if not vlm_results:
            return False
        verdicts = {getattr(r, "overall_verdict", None) for r in vlm_results}
        return len(verdicts) >= 2

    def _neural_top1(self, neural) -> int:
        if neural is None:
            return 0
        return getattr(neural, "top1_idx", 0)

    def _cv_top1(self, banding_results: List, quality_results: List, n_slices: int) -> int:
        """
        Best CV slice: lowest banding severity + highest homogeneity.
        Score = (1 - severity) * 0.5 + homogeneity * 0.5
        """
        if not banding_results and not quality_results:
            return 0

        scores = [0.0] * n_slices
        for b in banding_results:
            idx = b.slice_idx
            if 0 <= idx < n_slices:
                scores[idx] += (1.0 - b.severity) * 0.5

        for q in quality_results:
            idx = q.slice_idx
            if 0 <= idx < n_slices:
                scores[idx] += q.homogeneity * 0.5

        import numpy as np
        return int(np.argmax(scores))

    def _vlm_top1(self, vlm_results: List) -> Optional[int]:
        """Return slice index rated 'optimal' by VLM; None if VLM not run."""
        if not vlm_results:
            return None
        optimal = [v for v in vlm_results if v.overall_verdict == "optimal"]
        if optimal:
            return optimal[0].slice_idx
        acceptable = [v for v in vlm_results if v.overall_verdict == "acceptable"]
        if acceptable:
            return acceptable[0].slice_idx
        return None

    # ------------------------------------------------------------------

    def _per_slice_agreement(self, neural, banding, quality, vlm, n_slices: int) -> List[Dict]:
        import numpy as np

        neural_scores = None
        if neural is not None and hasattr(neural, "slice_scores"):
            neural_scores = np.array(neural.slice_scores)

        band_severity = {b.slice_idx: b.severity for b in banding}
        qual_hom      = {q.slice_idx: q.homogeneity for q in quality}
        vlm_verdict   = {v.slice_idx: v.overall_verdict for v in vlm}

        result = []
        for i in range(n_slices):
            neural_ok = (
                neural_scores is not None and neural_scores[i] >= neural_scores.mean()
            )
            cv_ok = (
                band_severity.get(i, 0.5) < 0.3 and qual_hom.get(i, 0.5) > 0.7
            )
            vlm_ok = vlm_verdict.get(i, "acceptable") in ("optimal", "acceptable")
            agreed = (neural_ok == cv_ok) and (cv_ok == vlm_ok or i not in vlm_verdict)
            result.append({
                "slice_idx": i,
                "neural_ok": neural_ok,
                "cv_ok": cv_ok,
                "vlm_ok": vlm_ok if i in vlm_verdict else None,
                "agreed": agreed,
            })
        return result

    def _collect_disagreements(
        self, neural_top1, cv_top1, vlm_top1,
        neural_cv_agree, neural_vlm_agree, cv_vlm_agree,
        state: EvidenceState,
    ) -> List[str]:
        d = []
        offsets = state.offsets_hz
        def hz(idx): return f"{offsets[idx]:+.0f} Hz" if idx is not None and idx < len(offsets) else "?"

        if not neural_cv_agree:
            d.append(
                f"Neural ranker picks slice {neural_top1} ({hz(neural_top1)}) "
                f"but CV metrics favour slice {cv_top1} ({hz(cv_top1)})."
            )
        if not neural_vlm_agree and vlm_top1 is not None:
            d.append(
                f"Neural ranker picks slice {neural_top1} ({hz(neural_top1)}) "
                f"but VLM rates slice {vlm_top1} ({hz(vlm_top1)}) as optimal."
            )
        if not cv_vlm_agree and vlm_top1 is not None:
            d.append(
                f"CV metrics favour slice {cv_top1} ({hz(cv_top1)}) "
                f"but VLM prefers slice {vlm_top1} ({hz(vlm_top1)})."
            )
        return d
