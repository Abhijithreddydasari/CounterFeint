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


def _truncate_keep_tail(s: str, n: int) -> str:
    """Hard-cap a string at ``n`` chars but keep the TAIL (most recent content)."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= n:
        return s
    return "...(older trimmed)\n" + s[-(n - len("...(older trimmed)\n")) :]


# Columns rendered per decided ad. Mixes discriminative ring-detection
# columns (payment_id, registrar, domain, targeting_fingerprint) with
# non-discriminative decoys (category, country, account_age_days) so
# the model must learn which collisions matter for link_accounts.
_DECIDED_AD_COLUMNS: List[str] = [
    "category",
    "country",
    "account_age_days",
    "domain",
    "registrar",
    "payment_id",
    "payment_type",
    "targeting_fingerprint",
]


def _render_decided_ads(
    decided_ads: List[Dict[str, Any]],
    *,
    max_ads: int = 15,
) -> str:
    """Render already-verdicted ads with their verdict + key signals.

    Each decided ad shows: verdict, confidence, and a curated mix of
    discriminative + decoy parameters extracted from the evidence ledger.
    This gives the model memory of past ads so it can detect cross-ad
    signal collisions for link_accounts.
    """
    if not decided_ads:
        return "(none yet)"

    kept = decided_ads[-max_ads:]
    trailer = (
        f"  (+{len(decided_ads) - max_ads} older decided ads not shown)\n"
        if len(decided_ads) > max_ads else ""
    )
    lines: List[str] = []
    for entry in kept:
        ad_id = entry.get("ad_id", "?")
        verdict = entry.get("verdict", "?")
        confidence = entry.get("confidence", "?")
        if isinstance(confidence, float):
            confidence = f"{confidence:.2f}"
        cells: List[str] = []
        for col in _DECIDED_AD_COLUMNS:
            val = entry.get(col)
            if val not in (None, ""):
                cells.append(f"{col}={val}")
        params = " | ".join(cells) if cells else "(no signals)"
        lines.append(f"  {ad_id}: verdict={verdict} confidence={confidence} | {params}")
    return (trailer + "\n".join(lines)).rstrip()


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

        findings = _truncate_keep_tail(
            observation.get("investigation_findings") or "(nothing pulled yet)",
            n=1800,
        )

        current_ad_info = observation.get("current_ad_info") or ""
        if not current_ad_info:
            current_ad_info = "No ad in focus yet — investigate any pending ad."

        decided_ads_history = _render_decided_ads(
            observation.get("decided_ads") or []
        )

        return INVESTIGATOR_USER_TEMPLATE.format(
            steps_remaining=queue_status.get("steps_remaining", "?"),
            investigation_budget=queue_status.get("investigation_budget", "?"),
            reviewed_count=queue_status.get("reviewed", 0),
            queue_may_grow=observation.get("queue_may_grow", False),
            pending_len=len(pending),
            pending_preview=", ".join(pending[:12])
            + (f" +{len(pending) - 12}" if len(pending) > 12 else ""),
            current_ad_info=current_ad_info,
            findings_preview=findings,
            decided_ads_history=decided_ads_history,
            feedback=(observation.get("feedback") or "").strip() or "(none)",
        )


__all__ = ["LLMInvestigator"]
