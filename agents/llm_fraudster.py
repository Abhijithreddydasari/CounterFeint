"""
LLM-backed Fraudster policy.

Wraps :class:`.base.LLMPolicyBase` with the Fraudster system prompt and
observation-to-user-prompt assembly logic. Falls back to
:class:`counterfeint.scripted.ReactiveFraudster` on any failure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import FraudsterAction
from ..scripted._base import PolicyBase
from ..scripted.fraudster import ReactiveFraudster
from .base import LLMPolicyBase
from .prompts import FRAUDSTER_SYSTEM_PROMPT, FRAUDSTER_USER_TEMPLATE


def _compact_queue(queue: List[Dict[str, Any]], max_items: int = 6) -> str:
    """Condense the current_queue field to a single-line preview."""
    if not queue:
        return "[empty]"
    parts: List[str] = []
    for entry in queue[:max_items]:
        ad_id = entry.get("ad_id", "?")
        cat = entry.get("category", "?")
        status = entry.get("status", "pending")
        mine = " (mine)" if entry.get("is_my_proposal") else ""
        parts.append(f"{ad_id}/{cat}/{status}{mine}")
    trailer = "" if len(queue) <= max_items else f" +{len(queue) - max_items} more"
    return "; ".join(parts) + trailer


def _compact_prior_verdicts(
    verdicts: List[Dict[str, Any]], max_items: int = 5
) -> str:
    if not verdicts:
        return "[none yet]"
    recent = verdicts[-max_items:]
    out: List[str] = []
    for v in recent:
        ad_id = v.get("ad_id", "?")
        decision = v.get("verdict", "?")
        conf = v.get("confidence")
        mine = " [mine]" if v.get("was_my_proposal") else ""
        rationale = (v.get("rationale") or "").strip()
        if len(rationale) > 80:
            rationale = rationale[:77] + "..."
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
        out.append(f"{ad_id}={decision}@{conf_str}{mine}: {rationale}")
    return " | ".join(out)


def _compact_inv_targets(
    targets: Dict[str, List[str]], max_items: int = 6
) -> str:
    if not targets:
        return "[none]"
    parts = []
    for ad_id in list(targets.keys())[:max_items]:
        t_list = targets[ad_id]
        parts.append(f"{ad_id}:{','.join(t_list)}")
    trailer = "" if len(targets) <= max_items else f" +{len(targets) - max_items}"
    return " | ".join(parts) + trailer


class LLMFraudster(LLMPolicyBase):
    """LLM Fraudster with a :class:`ReactiveFraudster` deterministic fallback."""

    system_prompt = FRAUDSTER_SYSTEM_PROMPT
    action_model = FraudsterAction
    _log_name = "fraudster"

    def __init__(
        self,
        *,
        fallback_policy: Optional[PolicyBase] = None,
        fallback_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        if fallback_policy is None:
            fallback_policy = ReactiveFraudster(seed=fallback_seed)
        super().__init__(fallback_policy=fallback_policy, **kwargs)

    # ------------------------------------------------------------------
    def _build_user_prompt(self, observation: Dict[str, Any]) -> str:
        queue = observation.get("current_queue", []) or []
        verdicts = observation.get("prior_verdicts", []) or []
        inv_targets = observation.get("investigation_targets_used", {}) or {}

        return FRAUDSTER_USER_TEMPLATE.format(
            round_number=observation.get("round_number", 0),
            rounds_remaining=observation.get("rounds_remaining", 0),
            proposals_used=observation.get("proposals_used", 0),
            proposals_remaining=observation.get("proposals_remaining", 0),
            actions_left_this_turn=observation.get("actions_left_this_turn", 0),
            allowed_categories=", ".join(
                observation.get("allowed_categories", []) or ["(none)"]
            ),
            queue_len=len(queue),
            current_queue_preview=_compact_queue(queue),
            prior_verdicts_preview=_compact_prior_verdicts(verdicts),
            investigation_targets_preview=_compact_inv_targets(inv_targets),
            feedback=(observation.get("feedback") or "").strip() or "(none)",
        )


__all__ = ["LLMFraudster"]
