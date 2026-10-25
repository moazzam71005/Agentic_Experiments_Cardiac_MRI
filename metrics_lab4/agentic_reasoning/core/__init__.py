"""
Core modules for Exp 15 (VLM Visual Grounding & Crop Inspection)
"""
from .evidence_store import EvidenceStore, EvidencePacket
from .gap_analyzer import GapAnalyzer, ChecklistItem
from .consistency_checker import ConsistencyChecker, ConsistencyReport
from .vlm_inspector import VLMInspector, VLMDecision
from .stage_a_generator import StageAGenerator
