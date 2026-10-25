"""
LLM subpackage for Exp 14 (ADRE-Debate)
"""
from .client import LLMClient
from .narrative_generator import NarrativeGenerator
from .prompts_debate import (
    PHYSICS_AGENT_SYSTEM,
    PHYSICS_AGENT_USER,
    CONTRAST_AGENT_SYSTEM,
    CONTRAST_AGENT_USER,
    META_CRITIC_SYSTEM,
    META_CRITIC_USER,
)
