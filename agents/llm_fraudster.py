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
    """Condense the current_queue field to a single-line preview."""
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
        return "(none yet)"
    recent = verdicts[-max_items:]
    out: List[str] = []
    for v in recent:
        ad_id = v.get("ad_id", "?")
        decision = v.get("verdict", "?")
        conf = v.get("confidence")
        mine = " [mine]" if v.get("was_my_proposal") else ""
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
        out.append(f"{ad_id}={decision}@{conf_str}{mine}")
    return " | ".join(out)


# Columns rendered per proposal. Matches the Investigator's decided-ads
# column set so both sides reason over the same signal vocabulary.
_PROPOSAL_SIGNAL_COLUMNS: List[str] = [
    "slot_index",
    "investigator_verdict",
    "category",
    "country",
    "account_age_days",
    "domain",
    "registrar",
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


# Per-task-tier guidance appended to the user prompt. Easy tasks get a
# louder fraud posture (so the Investigator can succeed at all in the
# pre-training baseline); hard tasks get a stealthier posture (so the
# trained Investigator's lift is meaningful). We deliberately surface
# this in the USER prompt instead of mutating ``system_prompt`` so the
# policy still sees a stable system role across all tasks (the system
# prompt is the schema contract; per-task hints live in the user
# message where the rest of the per-turn observation lives).
_TASK_TIER_HINTS: Dict[str, str] = {
    "task_1": (
        "Tier hint (task_1, easy): use blatant fraud cues — obvious "
        "scam ad copy, unrelated targeting, cheap TLDs. The Investigator "
        "needs a clean baseline to learn against."
    ),
    "task_2": (
        "Tier hint (task_2, medium): mix one or two subtle ads with the "
        "obvious ones; vary registrars / payment types across your slate."
    ),
    "task_3": (
        "Tier hint (task_3, hard): aim for stealth. Construct fraud "
        "rings via shared payment_id / domain / targeting_fingerprint "
        "across 2-3 of your proposals so the only path to detection is "
        "cross-ad ring inference, not single-ad surface red flags."
    ),
    "task_3_unseen": (
        "Tier hint (task_3_unseen, hard generalisation): same posture "
        "as task_3 but you are being evaluated on a held-out seed — do "
        "NOT collapse to a single template; vary your slate."
    ),
}


def _task_tier_hint(task_id: str) -> str:
    return _TASK_TIER_HINTS.get(task_id or "", "")


# Field whitelist per action_type. Keys not in the whitelist for the
# action's type are dropped before Pydantic validation so a small
# Llama-class model that mixes (e.g. ``slot_index`` on a ``propose_ad``,
# or ``ad_copy`` on a ``modify_pending_ad``) still produces a valid
# action instead of falling back. ``rationale`` and ``action_type`` are
# always allowed.
_FRAUDSTER_FIELDS_BY_TYPE: Dict[str, set[str]] = {
    "propose_ad": {
        "action_type", "ad_copy", "category",
        "landing_page_blurb", "targeting_summary", "rationale",
    },
    "modify_pending_ad": {
        "action_type", "slot_index",
        "new_ad_copy", "new_landing_page_blurb", "rationale",
    },
    "end_turn": {"action_type", "rationale"},
    "commit_final": {"action_type", "rationale"},
}


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
        tier_hint = _task_tier_hint(observation.get("task_id", ""))

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
            my_proposal_signals_preview=_render_my_proposal_signals(
                observation.get("my_proposal_signals") or {}
            ),
            tier_hint=tier_hint or "(no tier hint for this task)",
            feedback=(observation.get("feedback") or "").strip() or "(none)",
        )

    # ------------------------------------------------------------------
    # Schema-coercion shim. Llama 3.1 in particular tends to:
    #   1. emit ``"slot_index": -1`` on ``propose_ad`` (it interprets
    #      the slot field as "no slot yet"), which violates the
    #      ``ge=0`` constraint and trips the Pydantic validator → the
    #      whole step then falls back to the deterministic
    #      ReactiveFraudster, polluting fallback metrics.
    #   2. include modify-only fields (``slot_index``, ``new_ad_copy``)
    #      on a ``propose_ad`` action and vice-versa. These are
    #      Optional in the schema so they pass validation, but they
    #      poison the audit log because ``_serialize_fraudster_action``
    #      copies them through verbatim.
    #
    # We normalise both classes of issue here so the fallback only
    # fires on hard JSON / unknown-action errors.
    # ------------------------------------------------------------------
    def _coerce_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        action_type = data.get("action_type")
        if action_type not in _FRAUDSTER_FIELDS_BY_TYPE:
            return data
        allowed = _FRAUDSTER_FIELDS_BY_TYPE[action_type]
        out: Dict[str, Any] = {k: v for k, v in data.items() if k in allowed}

        # Llama 3.1 frequently emits ``targeting_summary`` (and other
        # free-text fields) as a structured dict like
        #   {"age_range": [13, 65], "genders": ["male", "female"], ...}
        # or a list, rather than the schema-required string. Flatten
        # these into a deterministic string representation here so we
        # don't burn through the fallback budget on every other turn.
        for text_field in (
            "ad_copy",
            "landing_page_blurb",
            "targeting_summary",
            "new_ad_copy",
            "new_landing_page_blurb",
            "new_targeting_summary",
        ):
            if text_field in out and not isinstance(out[text_field], str):
                out[text_field] = _stringify_text_field(out[text_field])

        if action_type == "modify_pending_ad":
            slot = out.get("slot_index")
            if isinstance(slot, str):
                try:
                    out["slot_index"] = int(slot)
                except (TypeError, ValueError):
                    out.pop("slot_index", None)
            slot = out.get("slot_index")
            if isinstance(slot, int) and slot < 0:
                # -1 is meaningless for a modify; surface as missing
                # so the env returns its "modify_pending_ad requires
                # slot_index" error rather than silently rewriting
                # slot 0.
                out.pop("slot_index", None)

        return out


def _stringify_text_field(value: Any) -> str:
    """Flatten a dict/list LLM emission into a comma-joined string.

    For dict inputs, joins ``key=value`` pairs (sub-lists comma-joined,
    sub-dicts JSON-encoded as a fallback). For list inputs, joins the
    items by ``", "``. Anything else is rendered through ``str()``.
    The result is intended to preserve the LLM's intent as readable
    text without crashing the schema validator.
    """
    if isinstance(value, dict):
        parts: List[str] = []
        for k, v in value.items():
            if isinstance(v, list):
                rendered = ",".join(str(item) for item in v)
            elif isinstance(v, dict):
                rendered = ";".join(f"{ki}={vi}" for ki, vi in v.items())
            else:
                rendered = str(v)
            parts.append(f"{k}={rendered}")
        return "; ".join(parts)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


__all__ = ["LLMFraudster"]
