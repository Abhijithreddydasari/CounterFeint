"""
LLM-backed role policies for CounterFeint.

All classes here wrap an OpenAI-compatible chat endpoint (HuggingFace Router,
Ollama, vLLM, etc.) behind the same ``act(observation) -> Action`` contract the
scripted policies expose. Each LLM policy falls back deterministically to a
scripted counterpart on any failure (API error, timeout, bad JSON, invalid
action schema) so downstream code — including RL training rollouts — never
observes a policy crash.

Modules:
    base             :class:`LLMPolicyBase` — shared retry / timeout / validate
                       / fallback machinery.
    prompts          Role-specific system prompts.
    llm_fraudster    :class:`LLMFraudster`  (falls back to ReactiveFraudster).
    llm_investigator :class:`LLMInvestigator` (falls back to ScriptedInvestigator).
"""

from .base import LLMPolicyBase
from .llm_fraudster import LLMFraudster
from .llm_investigator import LLMInvestigator
from .prompts import FRAUDSTER_SYSTEM_PROMPT, INVESTIGATOR_SYSTEM_PROMPT

__all__ = [
    "LLMPolicyBase",
    "LLMFraudster",
    "LLMInvestigator",
    "FRAUDSTER_SYSTEM_PROMPT",
    "INVESTIGATOR_SYSTEM_PROMPT",
]
