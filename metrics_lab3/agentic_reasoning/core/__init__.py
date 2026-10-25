"""
Core modules for Exp 14 (ADRE-Debate)
"""
from .evidence_store import EvidenceStore, EvidencePacket
from .gap_analyzer import GapAnalyzer, ChecklistItem
from .consistency_checker import ConsistencyChecker, ConsistencyReport
from .adre_debate import ADREDebateEngine, DebateDecision
from .stage_a_generator import StageAGenerator
