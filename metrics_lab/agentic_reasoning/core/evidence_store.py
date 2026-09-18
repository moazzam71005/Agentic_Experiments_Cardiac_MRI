"""
Evidence Store (ADRE-style append-only provenance-tagged repository)
---------------------------------------------------------------------
Tracks all measurements, tool outputs, and observations for a frequency-scout patient.
Provides serialisation and compact text formatting for LLM prompt ingestion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidencePacket:
    """A single atomic piece of evidence produced by a measurement tool."""
    patient_id: str
    image_no: int              # 1-indexed (1..13)
    offset_hz: float           # e.g. -150.0 to +150.0
    tool: str                  # e.g. "cnn", "banding", "homogeneity", "contrast", "temporal", "physics"
    metric_name: str           # e.g. "banded_fraction_myo", "homogeneity", "cnn_score"
    value: Any                 # numeric value, string, or structured dict
    provenance: str            # source identifier (e.g. "EfficientNet-B0_LOOCV", "UNet_LVSA_BandingDetector")
    summary: str = ""          # human-readable clinical summary line


class EvidenceStore:
    """
    Per-patient append-only evidence store.
    """

    def __init__(self, patient_id: str, n_slices: int = 13, offsets_hz: Optional[List[float]] = None):
        self.patient_id = patient_id
        self.n_slices = n_slices
        self.offsets_hz = offsets_hz or [-150.0 + i * 25.0 for i in range(n_slices)]
        self._packets: List[EvidencePacket] = []

    def add(self, packet: EvidencePacket) -> None:
        self._packets.append(packet)

    def add_many(self, packets: List[EvidencePacket]) -> None:
        self._packets.extend(packets)

    def get_for_tool(self, tool: str) -> List[EvidencePacket]:
        return [p for p in self._packets if p.tool == tool]

    def get_for_image(self, image_no: int) -> List[EvidencePacket]:
        return [p for p in self._packets if p.image_no == image_no]

    def get_metric_map(self, metric_name: str) -> Dict[int, Any]:
        """Return {image_no: value} for a given metric."""
        return {p.image_no: p.value for p in self._packets if p.metric_name == metric_name}

    def tools_present(self) -> List[str]:
        return sorted(list({p.tool for p in self._packets}))

    def has_tool(self, tool: str) -> bool:
        return any(p.tool == tool for p in self._packets)

    def to_tabular_summary(self) -> str:
        """
        Formats per-slice evidence into a clean Markdown table for LLM prompts.
        """
        lines = [
            f"### Per-Slice Quantitative Evidence Matrix (Patient: {self.patient_id})",
            "| Image # | Offset (Hz) | CNN Score | Banded Frac (Myo) | Homogeneity (1-CoV) | CNR | Temporal Std | Banded Sectors |",
            "|---|---|---|---|---|---|---|---|"
        ]

        by_image: Dict[int, Dict[str, Any]] = {i + 1: {} for i in range(self.n_slices)}
        for p in self._packets:
            if p.image_no in by_image:
                by_image[p.image_no][p.metric_name] = p.value

        for img_no in sorted(by_image.keys()):
            d = by_image[img_no]
            hz = self.offsets_hz[img_no - 1] if img_no - 1 < len(self.offsets_hz) else 0.0
            cnn = f"{float(d.get('cnn_score', 0.0)):.4f}" if "cnn_score" in d else "N/A"
            bfm = f"{float(d.get('banded_fraction_myo', 0.0)):.4f}" if "banded_fraction_myo" in d else "N/A"
            hom = f"{float(d.get('homogeneity', 0.0)):.4f}" if "homogeneity" in d else "N/A"
            cnr = f"{float(d.get('cnr', 0.0)):.2f}" if "cnr" in d else "N/A"
            temp = f"{float(d.get('temporal_std_sweep', 0.0)):.4f}" if "temporal_std_sweep" in d else "N/A"
            sec = str(d.get("band_sectors", "")).strip()
            if not sec or sec.lower() in ("nan", "none", "[]"):
                sec = "none"

            lines.append(f"| #{img_no:<2d} | {hz:+6.1f} Hz | {cnn} | {bfm} | {hom} | {cnr} | {temp} | {sec} |")

        return "\n".join(lines)

    def to_narrative_context(self) -> str:
        """Compact text summary of overall findings."""
        lines = [
            f"Patient {self.patient_id} Evidence Summary:",
            f"- Total slices acquired: {self.n_slices}",
            f"- Tools applied: {', '.join(self.tools_present())}"
        ]
        return "\n".join(lines)
