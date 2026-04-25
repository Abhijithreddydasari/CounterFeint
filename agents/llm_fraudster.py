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


def _compact_queue(queue: List[Dict[str, Any]], max_items: int = 8) -> str:
    """Condense the current_queue field to a single-line preview.

    Items the Fraudster owns (``is_my_proposal == True``) are always
    listed first so they never get dropped by the truncation cap — the
    Fraudster's own slate is the part it actually needs to reason about
    when picking ``modify_pending_ad`` vs ``propose_ad``.
    """
    if not queue:
        return "[empty]"

    own = [e for e in queue if e.get("is_my_proposal")]
    others = [e for e in queue if not e.get("is_my_proposal")]
    ordered = own + others

    parts: List[str] = []
    for entry in ordered[:max_items]:
        ad_id = entry.get("ad_id", "?")
        cat = entry.get("category", "?")
        status = entry.get("status", "pending")
        slot = entry.get("slot_index")
        mine = (
            f" (mine, slot={slot})"
            if entry.get("is_my_proposal") and slot is not None
            else (" (mine)" if entry.get("is_my_proposal") else "")
        )
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


# Order matches the Investigator's evidence-ledger column order, so the
# Fraudster can mentally cross-reference its slate against the columns
# the Investigator reasons over. We do NOT label which columns are the
# discriminative ones — that's something the Fraudster has to learn
# from rejection patterns.
_PROPOSAL_SIGNAL_COLUMNS: List[str] = [
    "slot_index",
    "investigator_verdict",
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


def _render_my_proposal_signals(
    signals: Dict[str, Dict[str, Any]],
    *,
    max_ads: int = 8,
) -> str:
    """Render the Fraudster's per-proposal signal table as compact text."""
    if not signals:
        return "(no proposals submitted yet)"
    ad_ids = list(signals.keys())
    if len(ad_ids) > max_ads:
        kept = ad_ids[-max_ads:]
        trailer = f"  (+{len(ad_ids) - max_ads} older proposals not shown)"
    else:
        kept = ad_ids
        trailer = ""
    lines: List[str] = []
    for ad_id in kept:
        entry = signals.get(ad_id, {}) or {}
        cells: List[str] = []
        for col in _PROPOSAL_SIGNAL_COLUMNS:
            if col in entry and entry[col] not in (None, ""):
                cells.append(f"{col}={entry[col]}")
        if cells:
            lines.append(f"  {ad_id}: " + " | ".join(cells))
    return ("\n".join(lines) + trailer).rstrip()


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
            my_proposal_signals_preview=_render_my_proposal_signals(
                observation.get("my_proposal_signals") or {}
            ),
            feedback=(observation.get("feedback") or "").strip() or "(none)",
        )


__all__ = ["LLMFraudster"]
