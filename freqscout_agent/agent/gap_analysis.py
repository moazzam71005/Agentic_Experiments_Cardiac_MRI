"""
Gap Analyser
------------
Compares the current Evidence State against the requirements Checklist and returns
a list of unsatisfied items plus a recommended next tool to run.

This drives the agentic loop: the agent calls gap_analysis after each tool invocation,
decides if the checklist is complete, and if not, selects the next tool to run.
"""

from __future__ import annotations

import logging
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .evidence_state import EvidenceState

logger = logging.getLogger(__name__)


@dataclass
class ChecklistItem:
    id: str
    description: str
    required_tools: List[str]
    required: bool
    satisfied: bool = False


@dataclass
class GapReport:
    unsatisfied: List[ChecklistItem]
    satisfied: List[ChecklistItem]
    is_complete: bool                   # All required items satisfied?
    next_tool: Optional[str]            # Recommended next tool name
    reasoning: str                      # Why this tool is recommended


class GapAnalyser:

    def __init__(self, checklist_path: str = "freqscout_agent/configs/checklist.yaml"):
        self.checklist_path = Path(checklist_path)
        self._items: List[ChecklistItem] = []
        self._load_checklist()

    # ------------------------------------------------------------------

    def analyse(self, state: EvidenceState) -> GapReport:
        """Evaluate which checklist items are satisfied given the current evidence."""
        self._evaluate(state)

        unsatisfied = [i for i in self._items if not i.satisfied and i.required]
        satisfied   = [i for i in self._items if i.satisfied]
        is_complete = len(unsatisfied) == 0

        next_tool, reasoning = None, "All checklist items satisfied."
        if unsatisfied:
            next_tool, reasoning = self._pick_next_tool(unsatisfied, state)

        return GapReport(
            unsatisfied=unsatisfied,
            satisfied=satisfied,
            is_complete=is_complete,
            next_tool=next_tool,
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------

    def _evaluate(self, state: EvidenceState):
        tools_present = set(state.tools_present())
        for item in self._items:
            if item.id == "cross_consistency":
                item.satisfied = state.has_tool("consistency")
            elif item.id == "alternatives_identified":
                item.satisfied = state.has_tool("recommendation")
            elif item.id == "recommendation_justified":
                item.satisfied = (
                    state.has_tool("recommendation")
                    and len([t for t in ["banding", "quality", "neural", "vlm"] if t in tools_present]) >= 2
                )
            elif item.id == "knowledge_retrieved":
                item.satisfied = state.has_tool("retrieve")
            else:
                # Satisfied if all required_tools have produced at least one packet
                item.satisfied = all(t in tools_present for t in item.required_tools)

    def _pick_next_tool(
        self, unsatisfied: List[ChecklistItem], state: EvidenceState
    ) -> tuple:
        """Simple priority ordering for tool selection."""
        # Priority: neural > segment > banding > quality > physics > vlm > retrieve > consistency
        priority_order = ["neural", "segment", "banding", "quality", "physics", "vlm", "retrieve"]

        for tool in priority_order:
            for item in unsatisfied:
                if tool in item.required_tools:
                    return tool, f"Running {tool} to satisfy: {item.description}"

        # Fallback: pick first unsatisfied item's first required tool
        if unsatisfied[0].required_tools:
            t = unsatisfied[0].required_tools[0]
            return t, f"Running {t} to satisfy: {unsatisfied[0].description}"

        return None, "No further tools needed; checklist gap is a synthesis step."

    # ------------------------------------------------------------------

    def _load_checklist(self):
        if not self.checklist_path.exists():
            logger.warning("Checklist not found at %s; using defaults.", self.checklist_path)
            self._items = self._default_items()
            return
        with open(self.checklist_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._items = [
            ChecklistItem(
                id=d["id"],
                description=d["description"],
                required_tools=d.get("required_tools", []),
                required=d.get("required", True),
            )
            for d in data.get("checklist", [])
        ]
        logger.info("Loaded %d checklist items.", len(self._items))

    @staticmethod
    def _default_items() -> List[ChecklistItem]:
        return [
            ChecklistItem("neural_scored",    "Neural ranker scores obtained.",    ["neural"],   True),
            ChecklistItem("seg_done",         "Segmentation mask obtained.",        ["segment"],  True),
            ChecklistItem("band_all_slices",  "Banding detected on all slices.",    ["banding"],  True),
            ChecklistItem("quality_all_slices","Quality metrics computed.",          ["quality"],  True),
            ChecklistItem("physics_attached", "Frequency offsets attached.",        ["physics"],  True),
            ChecklistItem("vlm_described",    "VLM description obtained.",          ["vlm"],      True),
            ChecklistItem("knowledge_retrieved","Knowledge retrieved.",              ["retrieve"], True),
            ChecklistItem("cross_consistency","Consistency check run.",             [],           True),
            ChecklistItem("alternatives_identified","Alternative offsets listed.",  [],           True),
            ChecklistItem("recommendation_justified","Recommendation justified.",   [],           True),
        ]
