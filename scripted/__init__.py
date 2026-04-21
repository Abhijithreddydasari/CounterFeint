"""
Scripted baseline policies for the CounterFeint FraudArena.

Each policy takes a role observation (dict from the WebSocket payload) and
returns a role-specific Pydantic action. They are deterministic, dependency-
free, and safe to use in CI or as opponents during self-play warmup.

Policies:

    fraudster.ScriptedFraudster     fixed 2-propose / end_turn / commit sequence
    fraudster.ReactiveFraudster     adapts category + slot based on verdicts seen
    fraudster.GibberishFraudster    low-plausibility adversary (negative control)

    investigator.ScriptedInvestigator   one-investigate-then-verdict heuristic

    auditor.HeuristicAuditor        rule-based flags + derived scores (Phase 1)
                                    Replaced by graders/auditor_track_{a,b}.py
                                    in Phase 2.
"""

from .auditor import HeuristicAuditor
from .fraudster import (
    GibberishFraudster,
    ReactiveFraudster,
    ScriptedFraudster,
)
from .investigator import ScriptedInvestigator

__all__ = [
    "GibberishFraudster",
    "HeuristicAuditor",
    "ReactiveFraudster",
    "ScriptedFraudster",
    "ScriptedInvestigator",
]
