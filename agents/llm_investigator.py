"""
LLM-backed Investigator policy.

Wraps :class:`.base.LLMPolicyBase` with the Investigator system prompt and
observation-to-user-prompt assembly logic. Falls back to
:class:`counterfeint.scripted.ScriptedInvestigator` on any failure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import AdReviewAction
from ..scripted._base import PolicyBase
from ..scripted.investigator import ScriptedInvestigator
from .base import LLMPolicyBase
from .prompts import INVESTIGATOR_SYSTEM_PROMPT, INVESTIGATOR_USER_TEMPLATE


def _truncate(s: str, n: int) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[: n - 3] + "..."


class LLMInvestigator(LLMPolicyBase):
    """LLM Investigator with a :class:`ScriptedInvestigator` fallback."""

    system_prompt = INVESTIGATOR_SYSTEM_PROMPT
    action_model = AdReviewAction
    _log_name = "investigator"

    def __init__(
        self,
        *,
        fallback_policy: Optional[PolicyBase] = None,
        **kwargs: Any,
    ) -> None:
        if fallback_policy is None:
            fallback_policy = ScriptedInvestigator()
        super().__init__(fallback_policy=fallback_policy, **kwargs)

    # ------------------------------------------------------------------
    def _build_user_prompt(self, observation: Dict[str, Any]) -> str:
        pending: List[str] = observation.get("available_ads", []) or []
        queue_status = observation.get("queue_status") or {}

        findings = _truncate(
            observation.get("investigation_findings") or "(nothing pulled yet)",
            n=1800,
        )

        verdict_history = _truncate(
            observation.get("verdict_history_summary") or "(no verdicts yet)",
            n=600,
        )

        current_ad_info = observation.get("current_ad_info") or ""
        if not current_ad_info:
            current_ad_info = "No ad in focus yet — focus one by starting with an investigate action on any pending ad."

        return INVESTIGATOR_USER_TEMPLATE.format(
            task_id=queue_status.get("task_id", "?"),
            steps_remaining=queue_status.get("steps_remaining", "?"),
            investigation_budget=queue_status.get("investigation_budget", "?"),
            reviewed_count=queue_status.get("reviewed", 0),
            queue_may_grow=observation.get("queue_may_grow", False),
            pending_len=len(pending),
            pending_preview=", ".join(pending[:12])
            + (f" +{len(pending) - 12}" if len(pending) > 12 else ""),
            current_ad_info=current_ad_info,
            findings_preview=findings,
            verdict_history=verdict_history,
            feedback=(observation.get("feedback") or "").strip() or "(none)",
        )


__all__ = ["LLMInvestigator"]
