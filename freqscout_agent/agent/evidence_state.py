"""
Evidence State  (append-only provenance-tagged store)
------------------------------------------------------
Stores all evidence collected during the agentic loop.
Each packet is tagged with: patient_id, slice_idx, offset_hz, tool, iteration.
Values are never overwritten — new observations are appended, giving a full audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidencePacket:
    """A single evidence item produced by one tool for one slice."""
    patient_id: str
    slice_idx: int
    offset_hz: float
    tool: str
    iteration: int
    data: Any                   # The actual result object (BandingResult, QualityResult, etc.)
    summary: str = ""           # Human-readable one-liner for the reasoning engine


class EvidenceState:
    """
    Per-patient, append-only evidence store.

    Usage
    -----
        state = EvidenceState(patient_id="P01")
        state.add(packet)
        state.get_for_tool("banding")   -> List[EvidencePacket]
        state.to_text_summary()         -> str  (for LLM context)
    """

    def __init__(self, patient_id: str, n_slices: int, offsets_hz: List[float]):
        self.patient_id = patient_id
        self.n_slices   = n_slices
        self.offsets_hz = offsets_hz
        self._packets: List[EvidencePacket] = []

    # ------------------------------------------------------------------

    def add(self, packet: EvidencePacket):
        self._packets.append(packet)

    def add_many(self, packets: List[EvidencePacket]):
        self._packets.extend(packets)

    # ------------------------------------------------------------------

    def get_for_tool(self, tool: str) -> List[EvidencePacket]:
        return [p for p in self._packets if p.tool == tool]

    def get_for_slice(self, slice_idx: int) -> List[EvidencePacket]:
        return [p for p in self._packets if p.slice_idx == slice_idx]

    def tools_present(self) -> List[str]:
        return list({p.tool for p in self._packets})

    def has_tool(self, tool: str) -> bool:
        return any(p.tool == tool for p in self._packets)

    def covered_slices(self, tool: str) -> List[int]:
        return [p.slice_idx for p in self._packets if p.tool == tool]

    # ------------------------------------------------------------------

    def to_text_summary(self, max_chars_per_tool: int = 2000) -> str:
        """
        Produce a compact text summary for injection into the LLM reasoning prompt.
        Groups evidence by slice and tool.
        """
        lines = [f"=== Evidence State for Patient {self.patient_id} ==="]
        lines.append(f"Slices: {self.n_slices}  |  Offsets: {[f'{h:+.0f}' for h in self.offsets_hz]}")

        tools_seen = self.tools_present()
        for tool in tools_seen:
            packets = self.get_for_tool(tool)
            lines.append(f"\n[Tool: {tool.upper()}]")
            char_budget = max_chars_per_tool
            for pkt in packets:
                entry = f"  Slice {pkt.slice_idx:2d} ({pkt.offset_hz:+.0f} Hz): {pkt.summary}"
                if char_budget - len(entry) < 0:
                    lines.append("  … (truncated)")
                    break
                lines.append(entry)
                char_budget -= len(entry)

        return "\n".join(lines)

    def slice_evidence_dict(self) -> Dict[int, Dict[str, Any]]:
        """
        Return {slice_idx: {tool: summary_str, ...}} for quick lookup.
        """
        result: Dict[int, Dict[str, Any]] = {i: {} for i in range(self.n_slices)}
        for pkt in self._packets:
            if pkt.slice_idx not in result:
                result[pkt.slice_idx] = {}
            result[pkt.slice_idx][pkt.tool] = pkt.summary
        return result

    def __len__(self):
        return len(self._packets)
