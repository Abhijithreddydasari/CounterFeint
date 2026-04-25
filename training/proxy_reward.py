"""
Per-completion proxy reward for online GRPO.

Why a proxy reward?
-------------------
TRL's :class:`trl.GRPOTrainer` (verified against the installed
TRL 1.2.0 source — see ``trl/trainer/grpo_trainer.py``)
calls ``unwrapped_model.generate(...)`` itself and then forwards
those *fresh* completions to ``reward_func(prompts, completions, ...)``.
Any reward function that tries to look up the completion in a
pre-collected ``{(prompt, completion) -> reward}`` table will *miss
on every single step* — the lookup key (the freshly generated
completion) is never the same string as the recorded one.

That's the bug the original notebook had: ``make_reward_fn`` returned
a constant ``-0.01`` for almost every step → zero advantage signal →
no learning.

This module replaces it with a **verifiable, per-completion** reward
that scores any (prompt, completion) pair WITHOUT touching the
FraudArena server, by:

  1. **Format / schema validity** — does the completion parse as JSON
     and validate as :class:`AdReviewAction`? Currently the dominant
     source of fallback at training time, so worth ~50% of the budget.
  2. **Coherence** — when the action references an ``ad_id`` /
     ``linked_ad_id``, that ad must actually appear in the prompt's
     pending list / current-focus block.
  3. **Action-class matches recorded gold** — when the dataset row
     carries a recorded gold-action class (verdict / investigate /
     link_accounts), generations of the right class get a small bonus.
  4. **Decision matches recorded gold** — when the row's
     ``terminal_grader_score`` says the recorded episode succeeded,
     verdicts / investigation_targets / linked_ad_ids that match the
     recorded ones get a larger bonus (it was a high-quality demo).
     When the recorded episode FAILED, copying the recorded action is
     mildly *penalised* (don't imitate failure modes).

This is consistent with how TRL's verifiable-reward GRPO recipes
look (e.g. ``open-r1`` / ``oat-math``): a fast, deterministic scorer
that captures schema correctness + a small task-specific signal.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from counterfeint.models import AdReviewAction


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _extract_json_text(raw: str) -> str:
    text = (raw or "").strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text


def _parse_completion(completion: str) -> Optional[AdReviewAction]:
    text = _extract_json_text(completion)
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return AdReviewAction.model_validate(data)
    except ValidationError:
        return None


def _action_class(action_type: str) -> str:
    return "verdict" if action_type in {"verdict", "link_accounts"} else "investigate"


# Lightweight {key: value} extraction from the recorded action_repr
# string the rollout collector stores in metadata. We only need a
# handful of fields and avoid eval()/AST on untrusted strings.
_REPR_FIELD_RE = re.compile(r"(\w+)=(?:'([^']*)'|([^,)\s]+))")


def _gold_fields_from_metadata(meta: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Best-effort extraction of (action_type, ad_id, verdict, target, linked)
    from the dataset row's recorded metadata."""
    out: Dict[str, Optional[str]] = {
        "action_type": None,
        "ad_id": None,
        "verdict": None,
        "investigation_target": None,
        "linked_ad_id": None,
    }
    repr_str = meta.get("action_repr")
    if not isinstance(repr_str, str):
        return out
    for match in _REPR_FIELD_RE.finditer(repr_str):
        k = match.group(1)
        v = match.group(2) if match.group(2) is not None else match.group(3)
        if k in out:
            out[k] = v
    return out


def _coherent_with_prompt(text: str, prompt: str) -> bool:
    """Soft check: the referenced ad_id appears verbatim in the prompt."""
    return bool(text) and text in prompt


def proxy_reward_one(
    prompt: str,
    completion: str,
    *,
    gold: Dict[str, Optional[str]],
    gold_episode_score: float,
) -> float:
    """Score a single (prompt, completion) pair on the [-0.5, 2.0] range.

    See module docstring for the rationale; this is the function GRPO
    calls per generation.
    """
    action = _parse_completion(completion)
    if action is None:
        # Hard schema failure — small negative so GRPO learns to avoid
        # the surface form, but capped so a long run of failures doesn't
        # destabilise advantages.
        return -0.5

    reward = 0.0

    # 1. Schema validity.
    reward += 0.6

    # 2. Coherence — the action references real IDs the prompt mentions.
    if action.ad_id and _coherent_with_prompt(action.ad_id, prompt):
        reward += 0.1
    if action.linked_ad_id and _coherent_with_prompt(action.linked_ad_id, prompt):
        reward += 0.1

    # 3. Action-class matches the recorded gold class. Small bonus —
    # we don't want to lock the model into mimicking the recorded
    # action, just nudge it toward the right *kind* of decision.
    gold_at = gold.get("action_type")
    if gold_at and _action_class(action.action_type) == _action_class(gold_at):
        reward += 0.2

    # 4. Decision matches recorded gold, scaled by recorded episode
    # quality. High-quality recorded episodes act as soft anchors;
    # low-quality ones don't (and the verdict/target/link fields don't
    # match, no penalty either way — we just don't add a bonus).
    quality = max(0.0, min(1.0, gold_episode_score))
    if quality > 0.0:
        if action.action_type == "verdict" and gold.get("verdict") == action.verdict:
            reward += 0.6 * quality
        if (
            action.action_type == "investigate"
            and gold.get("investigation_target") == action.investigation_target
        ):
            reward += 0.5 * quality
        if (
            action.action_type == "link_accounts"
            and gold.get("linked_ad_id") == action.linked_ad_id
        ):
            reward += 0.6 * quality

    return reward


def make_proxy_reward_fn(
    *,
    gold_lookup: Dict[str, Dict[str, Any]],
):
    """Build a TRL-compatible reward function.

    ``gold_lookup`` maps each ``prompt`` string in the dataset to its
    gold metadata + ``terminal_grader_score`` (constructed once at
    dataset-build time; see :func:`build_gold_lookup`).
    """

    def reward_fn(prompts: List[str], completions: List[str], **_: Any) -> List[float]:
        out: List[float] = []
        for prompt, completion in zip(prompts, completions):
            gold = gold_lookup.get(prompt)
            if gold is None:
                # Prompt the trainer batched but we never recorded —
                # only score schema validity + coherence.
                out.append(
                    proxy_reward_one(
                        prompt, completion,
                        gold={"action_type": None, "ad_id": None,
                              "verdict": None, "investigation_target": None,
                              "linked_ad_id": None},
                        gold_episode_score=0.0,
                    )
                )
                continue
            out.append(
                proxy_reward_one(
                    prompt, completion,
                    gold=gold["fields"],
                    gold_episode_score=float(gold["episode_score"]),
                )
            )
        return out

    return reward_fn


def build_gold_lookup(samples: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Build the prompt → gold map from a list of
    :class:`counterfeint.training.rollout.InvestigatorTrainingSample`.

    Most-recent recording wins on duplicate prompts (rare; happens only
    if the same observation is reached twice in different episodes).
    """
    out: Dict[str, Dict[str, Any]] = {}
    for s in samples:
        out[s.prompt] = {
            "fields": _gold_fields_from_metadata(s.metadata),
            "episode_score": float(s.terminal_grader_score),
        }
    return out


__all__ = [
    "build_gold_lookup",
    "make_proxy_reward_fn",
    "proxy_reward_one",
]
