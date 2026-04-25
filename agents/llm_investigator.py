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
    """Hard-cap a string at ``n`` chars from the FRONT (older content first)."""
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= n else s[: n - 3] + "..."


def _truncate_keep_tail(s: str, n: int) -> str:
    """Hard-cap a string at ``n`` chars but keep the TAIL (most recent content).

    Used for findings/verdict logs that grow append-only across an
    episode: the freshest pulls / verdicts are the ones the model needs
    most for its next action, so dropping the oldest entries is safer
    than dropping the newest.
    """
    if not s:
        return ""
    s = s.strip()
    if len(s) <= n:
        return s
    return "...(older trimmed)\n" + s[-(n - len("...(older trimmed)\n")) :]


# Order matters: the per-ad row is emitted in this order, so the LLM sees
# high-signal ring columns close together and low-signal columns at the
# tail. (We do NOT tell the model which is which — it has to learn.)
_LEDGER_COLUMNS: List[str] = [
    "category",
    "country",
    "account_age_days",
    "advertiser_id",
    "verified_business",
    "domain",
    "registrar",
    "domain_age_days",
    "payment_id",
    "payment_type",
    "targeting_fingerprint",
]
_LEDGER_MAX_ADS: int = 12


def _render_evidence_ledger(
    ledger: Dict[str, Dict[str, Any]],
    *,
    max_ads: int = _LEDGER_MAX_ADS,
) -> str:
    """Render the per-ad evidence dict as a compact text table.

    One ad per line. Only fields that are *populated* on at least one
    listed ad are emitted on a row, and unpopulated fields on a given
    row are skipped (so freshly-touched ads don't spam ``=?`` columns).

    Cap at ``max_ads`` ads so the prompt stays within budget on large
    queues; we keep the most recent ``max_ads`` entries since those are
    the most likely ring members the Investigator just touched.
    """
    if not ledger:
        return "(no evidence collected yet — investigate at least one ad)"

    ad_ids = list(ledger.keys())
    if len(ad_ids) > max_ads:
        kept = ad_ids[-max_ads:]
        trailer = f"  (+{len(ad_ids) - max_ads} older ads not shown)"
    else:
        kept = ad_ids
        trailer = ""

    lines: List[str] = []
    for ad_id in kept:
        entry = ledger.get(ad_id, {}) or {}
        cells: List[str] = []
        for col in _LEDGER_COLUMNS:
            if col in entry and entry[col] not in (None, ""):
                cells.append(f"{col}={entry[col]}")
        if cells:
            lines.append(f"  {ad_id}: " + " | ".join(cells))
    return ("\n".join(lines) + trailer).rstrip()


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

        verdict_history = _truncate_keep_tail(
            observation.get("verdict_history_summary") or "(no verdicts yet)",
            n=600,
        )

        current_ad_info = observation.get("current_ad_info") or ""
        if not current_ad_info:
            current_ad_info = "No ad in focus yet — focus one by starting with an investigate action on any pending ad."

        evidence_ledger = _render_evidence_ledger(
            observation.get("evidence_ledger") or {}
        )

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
            evidence_ledger=evidence_ledger,
            feedback=(observation.get("feedback") or "").strip() or "(none)",
        )


__all__ = ["LLMInvestigator"]
