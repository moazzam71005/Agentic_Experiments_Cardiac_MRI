"""
Agentic Reasoning Loop  (ADRE Algorithm 1 – adapted for bSSFP frequency scouting)
-----------------------------------------------------------------------------------
Orchestrates evidence acquisition, gap analysis, and consistency checking.

Loop body:
  1. Run gap analysis → identify missing checklist items.
  2. If complete (or t >= t_max), exit.
  3. Dispatch the recommended next tool.
  4. Update evidence state.
  5. Re-run gap analysis.
  6. After all tools are run, run consistency checker.
  7. If disagreements exist and budget allows, re-run VLM on disputed slices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence_state import EvidenceState, EvidencePacket
from .gap_analysis import GapAnalyser, GapReport
from .consistency import ConsistencyChecker, ConsistencyReport

logger = logging.getLogger(__name__)


@dataclass
class LoopTrace:
    """Full trace of the reasoning loop for audit / publication."""
    patient_id: str
    iterations: List[Dict] = field(default_factory=list)
    final_gap_report: Optional[GapReport] = None
    consistency_report: Optional[ConsistencyReport] = None
    terminated_at: int = 0
    termination_reason: str = ""


class AgentReasoningLoop:
    """
    Parameters
    ----------
    tools : dict mapping tool_name -> callable tool object
        Must include keys: "segment", "banding", "quality", "neural",
        "physics", "vlm", "retrieve".
    gap_analyser : GapAnalyser
    consistency_checker : ConsistencyChecker
    t_max : int
        Maximum loop iterations.
    """

    def __init__(
        self,
        tools: Dict[str, Any],
        gap_analyser: GapAnalyser,
        consistency_checker: ConsistencyChecker,
        t_max: int = 15,
    ):
        self.tools = tools
        self.gap_analyser = gap_analyser
        self.consistency_checker = consistency_checker
        self.t_max = t_max

    # ------------------------------------------------------------------

    def run(
        self,
        patient_id: str,
        images: List,
        offsets_hz: List[float],
        **extra_kwargs,
    ) -> tuple:
        """
        Execute the full agentic loop for one patient.

        Returns
        -------
        (EvidenceState, ConsistencyReport, LoopTrace)
        """
        state = EvidenceState(patient_id, len(images), offsets_hz)
        trace = LoopTrace(patient_id=patient_id)
        seg_result = None

        for t in range(self.t_max):
            gap = self.gap_analyser.analyse(state)
            logger.info(
                "[%s] Iter %d: %d unsatisfied items, next_tool=%s",
                patient_id, t, len(gap.unsatisfied), gap.next_tool,
            )

            trace.iterations.append({
                "iteration": t,
                "unsatisfied": [i.id for i in gap.unsatisfied],
                "next_tool": gap.next_tool,
                "reasoning": gap.reasoning,
            })

            if gap.is_complete:
                trace.termination_reason = "checklist_satisfied"
                trace.terminated_at = t
                break

            if gap.next_tool is None:
                trace.termination_reason = "no_further_tools"
                trace.terminated_at = t
                break

            # Dispatch the tool
            new_packets, seg_result = self._dispatch(
                gap.next_tool, t, state, seg_result,
                images, offsets_hz, patient_id, **extra_kwargs
            )
            state.add_many(new_packets)

        else:
            trace.termination_reason = "t_max_reached"
            trace.terminated_at = self.t_max

        # Consistency check
        consistency = self.consistency_checker.run(state)

        # If there are disagreements and budget allows, re-run VLM on disputed slices
        if consistency.disagreements and len(trace.iterations) < self.t_max - 2:
            self._handle_disagreements(
                consistency, state, images, offsets_hz, patient_id, len(trace.iterations)
            )
            consistency = self.consistency_checker.run(state)  # re-check

        trace.final_gap_report = self.gap_analyser.analyse(state)
        trace.consistency_report = consistency

        logger.info(
            "[%s] Loop done after %d iters (%s). Confidence=%.2f (%s).",
            patient_id, trace.terminated_at, trace.termination_reason,
            consistency.confidence, consistency.confidence_label,
        )
        return state, consistency, trace

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        tool_name: str,
        iteration: int,
        state: EvidenceState,
        seg_result,
        images: List,
        offsets_hz: List[float],
        patient_id: str,
        **extra_kwargs,
    ):
        """Run the named tool and return (List[EvidencePacket], updated_seg_result)."""
        packets = []

        if tool_name == "neural" and "neural" in self.tools:
            result = self.tools["neural"].run(images, offsets_hz, patient_id)
            pkt = EvidencePacket(
                patient_id=patient_id, slice_idx=-1, offset_hz=0.0,
                tool="neural", iteration=iteration, data=result,
                summary=(
                    f"Top-1: slice {result.top1_idx} ({offsets_hz[result.top1_idx]:+.0f} Hz), "
                    f"Top-3: {result.top3_idx}, conf={result.softmax_confidence[result.top1_idx]:.3f}"
                ),
            )
            packets.append(pkt)

        elif tool_name == "segment" and "segment" in self.tools:
            neural_pkts = state.get_for_tool("neural")
            top_idx = neural_pkts[0].data.top1_idx if neural_pkts else 0
            seg_result = self.tools["segment"].run(images, offsets_hz, patient_id, top_idx)
            pkt = EvidencePacket(
                patient_id=patient_id, slice_idx=top_idx,
                offset_hz=offsets_hz[top_idx],
                tool="segment", iteration=iteration, data=seg_result,
                summary=(
                    f"Segmentation from slice {top_idx} ({offsets_hz[top_idx]:+.0f} Hz). "
                    f"LV centroid: {tuple(round(c,1) for c in seg_result.lv_centroid)}."
                ),
            )
            packets.append(pkt)

        elif tool_name == "banding" and "banding" in self.tools and seg_result is not None:
            results = self.tools["banding"].run(images, offsets_hz, seg_result)
            for r in results:
                packets.append(EvidencePacket(
                    patient_id=patient_id, slice_idx=r.slice_idx,
                    offset_hz=r.offset_hz,
                    tool="banding", iteration=iteration, data=r,
                    summary=(
                        f"Band={'YES' if r.band_present else 'no'}, "
                        f"myo_overlap={r.myo_overlap_fraction:.2f}, "
                        f"severity={r.severity:.2f}, sectors={r.band_sectors}"
                    ),
                ))

        elif tool_name == "quality" and "quality" in self.tools and seg_result is not None:
            results = self.tools["quality"].run(images, offsets_hz, seg_result)
            for r in results:
                packets.append(EvidencePacket(
                    patient_id=patient_id, slice_idx=r.slice_idx,
                    offset_hz=r.offset_hz,
                    tool="quality", iteration=iteration, data=r,
                    summary=(
                        f"CoV={r.cov:.3f}, homogeneity={r.homogeneity:.3f}, "
                        f"CNR={r.cnr:.2f}, SNR={r.snr:.2f}"
                    ),
                ))

        elif tool_name == "physics" and "physics" in self.tools:
            result = self.tools["physics"].run(patient_id, offsets_hz)
            # Prefer Excel-derived offsets for the rest of the loop
            if result.offsets_hz and len(result.offsets_hz) == len(offsets_hz):
                offsets_hz[:] = list(result.offsets_hz)
                state.offsets_hz = list(result.offsets_hz)
            pkt = EvidencePacket(
                patient_id=patient_id, slice_idx=-1, offset_hz=0.0,
                tool="physics", iteration=iteration, data=result,
                summary=(
                    f"Offsets: {[f'{h:+.0f}' for h in result.offsets_hz]}, "
                    f"GT_slice={result.gt_best_slice}, GT_Hz={result.gt_offset_hz} "
                    f"(eval-only, not used in reasoning)"
                ),
            )
            packets.append(pkt)

        elif tool_name == "vlm" and "vlm" in self.tools:
            # Exp11: visual QA on physics peak + neural candidates + myo-overlap rejects
            targets = self._vlm_target_indices(state, images)
            results = self.tools["vlm"].run(
                images, offsets_hz, patient_id, seg_result, target_indices=targets
            )
            for r in results:
                packets.append(EvidencePacket(
                    patient_id=patient_id, slice_idx=r.slice_idx,
                    offset_hz=r.offset_hz,
                    tool="vlm", iteration=iteration, data=r,
                    summary=(
                        f"Verdict={r.overall_verdict}, band={r.banding_severity}, "
                        f"location={r.banding_location}, uniformity={r.myocardial_uniformity}"
                    ),
                ))

        elif tool_name == "retrieve" and "retrieve" in self.tools:
            # Build contextual queries from current evidence
            queries = self._build_retrieval_queries(state)
            retrieved = []
            for q in queries[:2]:   # at most 2 queries per iteration
                chunks = self.tools["retrieve"].retrieve(q)
                retrieved.extend(chunks)
            if retrieved:
                summary = f"Retrieved {len(retrieved)} chunks for {len(queries)} queries."
                pkt = EvidencePacket(
                    patient_id=patient_id, slice_idx=-1, offset_hz=0.0,
                    tool="retrieve", iteration=iteration, data=retrieved,
                    summary=summary,
                )
                packets.append(pkt)
        else:
            logger.warning("Tool '%s' not available or prerequisite missing.", tool_name)

        return packets, seg_result

    # ------------------------------------------------------------------

    def _build_retrieval_queries(self, state: EvidenceState) -> List[str]:
        """Construct retrieval queries from the current evidence state."""
        queries = ["bSSFP banding artifact frequency offset selection cardiac MRI 3T"]

        band_pkts = state.get_for_tool("banding")
        if band_pkts:
            affected = [p.data for p in band_pkts if getattr(p.data, "bands_in_myocardium", False)]
            if affected:
                sectors = set(
                    s for b in affected for s in (getattr(b, "band_sectors", None) or [])
                )
                if sectors:
                    queries.append(
                        f"banding artifact myocardium overlap {' '.join(sorted(sectors))} optimal offset selection"
                    )

        qual_pkts = state.get_for_tool("quality")
        if qual_pkts:
            queries.append("myocardial homogeneity CoV CNR SNR bSSFP image quality metrics")

        return queries

    # ------------------------------------------------------------------

    def _handle_disagreements(
        self, consistency, state, images, offsets_hz, patient_id, current_iter
    ):
        """Re-run VLM on disputed slices to resolve disagreements."""
        disputed_slices = set()
        for d in consistency.disagreements:
            for i in range(len(images)):
                if str(i) in d:
                    disputed_slices.add(i)

        if not disputed_slices and consistency.vlm_top1 is None:
            # Expand to any Exp11 targets not yet covered by VLM
            covered = set(state.covered_slices("vlm"))
            disputed_slices = set(self._vlm_target_indices(state, images)) - covered

        if disputed_slices and "vlm" in self.tools:
            # Prefer Exp11 target set intersection with disputed, else disputed alone
            covered = set(state.covered_slices("vlm"))
            targets = [i for i in self._vlm_target_indices(state, images) if i in disputed_slices]
            if not targets:
                targets = sorted(disputed_slices - covered) or sorted(disputed_slices)
            seg_pkts = state.get_for_tool("segment")
            seg_result = seg_pkts[0].data if seg_pkts else None
            results = self.tools["vlm"].run(
                images, offsets_hz, patient_id, seg_result,
                target_indices=targets,
            )
            for r in results:
                pkt = EvidencePacket(
                    patient_id=patient_id, slice_idx=r.slice_idx,
                    offset_hz=r.offset_hz,
                    tool="vlm", iteration=current_iter + 1, data=r,
                    summary=(
                        f"[re-check] Verdict={r.overall_verdict}, band={r.banding_severity}, "
                        f"location={r.banding_location}"
                    ),
                )
                state.add(pkt)
            logger.info(
                "Disagreement re-check: ran VLM on %d disputed slices.", len(targets)
            )

    def _vlm_target_indices(self, state: EvidenceState, images: List) -> List[int]:
        """
        Exp11 VLM targets (capped ~7): neural top-3 + lowest-bfm (+runner-up)
        + up to 3 myocardium-overlap rejects. Explains, does not score (W_VLM=0).
        """
        n = len(images)
        targets: set = set()
        ranked = []
        overlaps: List[int] = []

        neural_pkts = state.get_for_tool("neural")
        if neural_pkts:
            for i in getattr(neural_pkts[0].data, "top3_idx", []) or []:
                if 0 <= int(i) < n:
                    targets.add(int(i))
        elif n:
            targets.add(n // 2)

        qual = [p.data for p in state.get_for_tool("quality")]
        if qual:
            ranked = sorted(
                qual,
                key=lambda q: (
                    float(getattr(q, "banded_fraction_myo", 1.0)),
                    -float(getattr(q, "homogeneity", 0.0)),
                ),
            )
            for q in ranked[:2]:
                idx = int(q.slice_idx)
                if 0 <= idx < n:
                    targets.add(idx)

        band = [p.data for p in state.get_for_tool("banding")]
        overlaps = [
            int(b.slice_idx)
            for b in band
            if getattr(b, "bands_in_myocardium", False) and 0 <= int(b.slice_idx) < n
        ]
        for idx in overlaps[:3]:
            targets.add(idx)

        if not targets and n:
            targets.add(min(n // 2, n - 1))

        ordered = sorted(targets)
        if len(ordered) > 7:
            keep: set = set()
            if ranked:
                keep.add(int(ranked[0].slice_idx))
            if neural_pkts:
                keep.add(int(neural_pkts[0].data.top1_idx))
            keep.update(overlaps[:2])
            for i in ordered:
                if len(keep) >= 7:
                    break
                keep.add(i)
            ordered = sorted(i for i in keep if 0 <= i < n)[:7]
        return ordered
