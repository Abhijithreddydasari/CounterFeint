"""
LLM-backed role policies for CounterFeint.

All classes here wrap an OpenAI-compatible chat endpoint (HuggingFace Router,
Ollama, vLLM, etc.) — or a local ``transformers`` model in the case of
:class:`HFInvestigator` — behind the same ``act(observation) -> Action``
contract the scripted policies expose. Each LLM policy falls back
deterministically to a scripted counterpart on any failure (API error,
timeout, bad JSON, invalid action schema) so downstream code — including
RL training rollouts — never observes a policy crash.

Modules:
    base             :class:`LLMPolicyBase` — shared parse / validate /
                       fallback machinery and a pluggable chat backend.
    prompts          Role-specific system prompts.
    llm_fraudster    :class:`LLMFraudster`  (HTTP backend, falls back to
                       :class:`counterfeint.scripted.ReactiveFraudster`).
    llm_investigator :class:`LLMInvestigator` (HTTP backend, falls back to
                       :class:`counterfeint.scripted.ScriptedInvestigator`).
    hf_investigator  :class:`HFInvestigator` (local ``model.generate``
                       backend; what TRL trains).
"""

from .base import LLMPolicyBase
from .llm_fraudster import LLMFraudster
from .llm_investigator import LLMInvestigator
from .prompts import FRAUDSTER_SYSTEM_PROMPT, INVESTIGATOR_SYSTEM_PROMPT

# `HFInvestigator` requires `transformers` + `torch` at import time. We
# import it lazily so e.g. `pytest counterfeint/tests/test_eval_suite.py`
# in a slim CI image still works without the training stack installed.
try:
    from .hf_investigator import HFInvestigator
except ImportError:  # pragma: no cover - exercised only when transformers missing
    HFInvestigator = None  # type: ignore[assignment]

__all__ = [
    "LLMPolicyBase",
    "LLMFraudster",
    "LLMInvestigator",
    "HFInvestigator",
    "FRAUDSTER_SYSTEM_PROMPT",
    "INVESTIGATOR_SYSTEM_PROMPT",
]
