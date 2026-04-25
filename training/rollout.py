"""
Episode-collection and per-step instrumentation for Investigator training.

This module bridges the **episode-level** rewards CounterFeint computes
(see :mod:`counterfeint.graders.multi_agent_rewards`) and the
**per-(prompt, completion)** rows TRL's ``GRPOTrainer`` consumes:

  1. :class:`RecordingHFInvestigator` decorates an
     :class:`~counterfeint.agents.hf_investigator.HFInvestigator` and
     snapshots every ``act()`` call's prompt / completion / action.
  2. :func:`collect_episode` runs one full FraudArena three-agent episode
     with that recorder in the Investigator slot, then asks
     :func:`records_to_samples` to spread the episode-end Investigator
     reward across the recorded turns (with a verdict-vs-investigate
     shaping split — see ``ROUND_2_Q5_REALISM_REWARDS_TRAINING.md`` §3.2).
  3. :func:`collect_dataset` repeats step 2 over a
     ``{task_id: [seed, ...]}`` map and returns a flat list of
     :class:`InvestigatorTrainingSample`.
  4. :func:`samples_to_hf_dataset` converts that list to a
     ``datasets.Dataset`` ready for ``GRPOTrainer``.

The :class:`TracingPolicy` wrapper prints a one-line summary of every
agent action during a rollout — handy when running in a notebook to
sanity-check that the LLMs are actually doing something useful.

Why a separate module?  The training notebook used to inline ~280 lines
of this; pulling it out keeps the notebook to thin orchestration
(``§5`` is now ``from counterfeint.training import collect_dataset``)
and lets us unit-test the reward-distribution logic without spinning
up the FraudArena server.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from counterfeint.scripted import HeuristicAuditor, ReactiveFraudster

# `HFInvestigator` is a fwd reference here so this module can be imported
# even when transformers/torch aren't installed (e.g. running unit tests
# in a slim CI image).
try:
    from counterfeint.agents.hf_investigator import HFInvestigator  # noqa: F401
except ImportError:  # pragma: no cover - optional heavy dep
    HFInvestigator = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass: one row of the Investigator training dataset.
# ---------------------------------------------------------------------------


@dataclass
class InvestigatorTrainingSample:
    """One ``(prompt, completion, reward)`` row for TRL ``GRPOTrainer``.

    The ``reward`` is the Investigator's per-turn slice of the episode's
    composite Investigator reward (see
    :func:`records_to_samples` for the verdict-vs-investigate shaping
    split). Side columns (``task_id`` / ``seed`` / ``step_idx`` /
    ``terminal_grader_score`` / ``end_reason`` / ``metadata``) are kept
    for offline analysis and to let any *future* online reward function
    look up per-step ground-truth labels.
    """

    prompt: str
    completion: str
    reward: float
    task_id: str
    seed: int
    step_idx: int
    terminal_grader_score: float = 0.0
    end_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "reward": float(self.reward),
            "task_id": self.task_id,
            "seed": int(self.seed),
            "step_idx": int(self.step_idx),
            "terminal_grader_score": float(self.terminal_grader_score),
            "end_reason": self.end_reason,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Per-act() recorder.
# ---------------------------------------------------------------------------


class RecordingHFInvestigator:
    """Decorator around ``HFInvestigator`` that records each ``act()`` call.

    Every step we capture the LLM's last ``user_prompt`` and its raw
    ``completion`` (both populated by
    :class:`~counterfeint.agents.base.LLMPolicyBase.act`). On
    fallback steps both are ``None`` — :func:`records_to_samples` skips
    those rows since GRPO has no completion to score.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.step_records: List[Dict[str, Any]] = []
        self._last_step_idx: int = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def fallback_count(self) -> int:
        return getattr(self._inner, "fallback_count", 0)

    def reset(self) -> None:
        self.step_records.clear()
        self._last_step_idx = 0
        self._inner.reset()

    def act(self, observation: Dict[str, Any]) -> Any:
        result = self._inner.act(observation)
        self._last_step_idx += 1
        prompt = getattr(self._inner, "last_prompt", None)
        completion = getattr(self._inner, "last_completion", None)
        self.step_records.append(
            {
                "step_idx": self._last_step_idx,
                "prompt": prompt,
                "completion": completion,
                "fallback_used": prompt is None or completion is None,
                "action_repr": repr(result),
            }
        )
        return result


# ---------------------------------------------------------------------------
# Live one-line trace of every agent's action (notebook UX).
# ---------------------------------------------------------------------------


def summarise_action(
    role: str,
    action: Any,
    *,
    max_rationale_chars: int = 80,
) -> str:
    """Compact one-line summary of any role's action for the live trace."""

    def _g(name: str, default: Any = None) -> Any:
        if hasattr(action, name):
            return getattr(action, name)
        if isinstance(action, dict):
            return action.get(name, default)
        return default

    at = _g("action_type", "?")
    ad = _g("ad_id", "")
    parts: List[str] = [str(at)]
    if ad:
        parts.append(str(ad))

    if at == "investigate":
        tgt = _g("investigation_target")
        if tgt:
            parts.append(f"target={tgt}")
    elif at == "verdict":
        v = _g("verdict")
        c = _g("confidence")
        rationale = (_g("rationale") or "").strip()
        if v:
            parts.append(str(v))
        if c is not None:
            try:
                parts.append(f"@{float(c):.2f}")
            except (TypeError, ValueError):
                pass
        if rationale:
            if len(rationale) > max_rationale_chars:
                rationale = rationale[: max_rationale_chars - 3] + "..."
            parts.append(f'"{rationale}"')
    elif at == "link_accounts":
        linked = _g("linked_ad_id")
        reason = (_g("link_reason") or "").strip()
        if linked:
            parts.append(f"<-> {linked}")
        if reason:
            if len(reason) > max_rationale_chars:
                reason = reason[: max_rationale_chars - 3] + "..."
            parts.append(f'"{reason}"')
    elif at in {"propose_ad", "modify_pending_ad"}:
        cat = _g("category")
        if cat:
            parts.append(f"cat={cat}")
        copy = (_g("ad_copy") or _g("new_ad_copy") or "").strip()
        if copy:
            if len(copy) > max_rationale_chars:
                copy = copy[: max_rationale_chars - 3] + "..."
            parts.append(f'"{copy}"')
    return " ".join(parts)


class TracingPolicy:
    """Thin wrapper that prints one trace line per ``.act()`` and forwards.

    Set ``enabled=False`` to make it a no-op decorator (zero overhead).
    """

    _ROLE_TAG = {
        "fraudster": "FRAUD ",
        "investigator": "INVEST",
        "auditor": "AUDIT ",
    }

    def __init__(
        self,
        inner: Any,
        role: str,
        *,
        enabled: bool = True,
        max_rationale_chars: int = 80,
    ) -> None:
        self._inner = inner
        self._role = role
        self._enabled = bool(enabled)
        self._max_rationale_chars = int(max_rationale_chars)
        self._n = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def reset(self) -> None:
        self._n = 0
        if hasattr(self._inner, "reset"):
            self._inner.reset()

    def act(self, observation: Dict[str, Any]) -> Any:
        result = self._inner.act(observation)
        self._n += 1
        if not self._enabled:
            return result

        tag = self._ROLE_TAG.get(self._role, self._role.upper()[:6])
        inner_name = type(self._inner).__name__
        fallback = ""
        if isinstance(self._inner, RecordingHFInvestigator):
            rec = self._inner.step_records[-1] if self._inner.step_records else {}
            if rec.get("fallback_used"):
                fallback = " [FB]"
                inner_name = "HFInvestigator"
        elif (
            getattr(self._inner, "last_error", None)
            and getattr(self._inner, "fallback_count", 0) > 0
        ):
            fallback = " [FB]"

        print(
            f"      {tag} #{self._n:02d} ({inner_name:<22}){fallback}  "
            f"{summarise_action(self._role, result, max_rationale_chars=self._max_rationale_chars)}",
            flush=True,
        )
        return result


# ---------------------------------------------------------------------------
# Per-step reward shaping.
# ---------------------------------------------------------------------------

# Verdict / link_accounts are consequential decisions; investigate calls are
# preparatory. Splitting 80/20 (with matched counts → each verdict carries
# 4× the credit of each investigate) gives the Investigator a stronger
# gradient on the action that actually moves the grader without dropping
# the credit on tool use. See ROUND_2_Q5_REALISM_REWARDS_TRAINING.md §3.2.
_VERDICT_REWARD_SHARE = 0.80
_VERDICT_ACTION_TYPES = ("verdict", "link_accounts")


def classify_action(action_repr: Optional[str]) -> str:
    """Return ``"verdict"`` for consequential actions, ``"investigate"`` otherwise."""
    if not action_repr:
        return "investigate"
    text = action_repr.lower()
    return (
        "verdict"
        if any(f"action_type='{t}'" in text for t in _VERDICT_ACTION_TYPES)
        else "investigate"
    )


def records_to_samples(
    records: List[Dict[str, Any]],
    *,
    episode_result: Dict[str, Any],
    task_id: str,
    seed: int,
) -> List[InvestigatorTrainingSample]:
    """Distribute the episode-end Investigator reward across recorded turns.

    Verdicts / link_accounts get an :data:`_VERDICT_REWARD_SHARE` share
    of the episode reward; investigate calls share the rest. If the
    episode contains only one action class we fall back to a uniform
    split so we don't divide by zero.
    """
    grader_score = float(episode_result.get("grader_score", 0.0))
    end_reason = episode_result.get("end_reason")
    rewards_by_role = episode_result.get("rewards_by_role") or {}
    investigator_total = float(rewards_by_role.get("investigator", 0.0))

    investigator_records = [
        r for r in records
        if r.get("prompt") is not None and r.get("completion") is not None
    ]
    if not investigator_records:
        logger.warning(
            "No usable Investigator turns in episode %s/seed=%s — every step "
            "fell back to the scripted policy.",
            task_id, seed,
        )
        return []

    classes = [classify_action(r.get("action_repr")) for r in investigator_records]
    n_verdict = sum(1 for c in classes if c == "verdict")
    n_invest = len(investigator_records) - n_verdict

    if n_verdict == 0 or n_invest == 0:
        per_turn = investigator_total / len(investigator_records)
        per_turn_rewards = [per_turn] * len(investigator_records)
    else:
        verdict_share = _VERDICT_REWARD_SHARE * investigator_total / n_verdict
        invest_share = (1.0 - _VERDICT_REWARD_SHARE) * investigator_total / n_invest
        per_turn_rewards = [
            verdict_share if c == "verdict" else invest_share for c in classes
        ]

    return [
        InvestigatorTrainingSample(
            prompt=r["prompt"],
            completion=r["completion"],
            reward=per_turn_rewards[i],
            task_id=task_id,
            seed=seed,
            step_idx=int(r["step_idx"]),
            terminal_grader_score=grader_score,
            end_reason=end_reason,
            metadata={
                "action_repr": r.get("action_repr"),
                "action_class": classes[i],
            },
        )
        for i, r in enumerate(investigator_records)
    ]


# ---------------------------------------------------------------------------
# Top-level driver: collect one episode / a whole dataset.
# ---------------------------------------------------------------------------


PolicyFactory = Callable[[], Any]


def collect_episode(
    *,
    hf_investigator: Any,
    task_id: str,
    seed: int,
    fraudster_factory: Optional[PolicyFactory] = None,
    auditor_factory: Optional[PolicyFactory] = None,
    env_base_url: Optional[str] = None,
    log: bool = False,
    show_trace: bool = False,
    max_rationale_chars: int = 80,
) -> List[InvestigatorTrainingSample]:
    """Run one three-agent episode and return its Investigator training rows.

    Lazily imports the FraudArena driver so callers running unit tests
    on this module don't need ``websockets`` / a live server.
    """
    from counterfeint.inference import ENV_URL, run_three_agent_episode

    fraudster_factory = fraudster_factory or (lambda: ReactiveFraudster(seed=seed))
    auditor_factory = auditor_factory or (lambda: HeuristicAuditor())

    recorder = RecordingHFInvestigator(hf_investigator)
    recorder.reset()

    fraudster = fraudster_factory()
    investigator: Any = recorder
    auditor = auditor_factory()
    if show_trace:
        fraudster = TracingPolicy(
            fraudster, "fraudster", enabled=True,
            max_rationale_chars=max_rationale_chars,
        )
        investigator = TracingPolicy(
            recorder, "investigator", enabled=True,
            max_rationale_chars=max_rationale_chars,
        )
        auditor = TracingPolicy(
            auditor, "auditor", enabled=True,
            max_rationale_chars=max_rationale_chars,
        )

    result = run_three_agent_episode(
        task_id,
        fraudster_policy=fraudster,
        investigator_policy=investigator,
        auditor_policy=auditor,
        env_base_url=env_base_url or ENV_URL,
        seed=seed,
        log=log,
    )

    return records_to_samples(
        recorder.step_records,
        episode_result=result,
        task_id=task_id,
        seed=seed,
    )


def collect_dataset(
    *,
    hf_investigator: Any,
    seeds_by_task: Dict[str, List[int]],
    fraudster_factory: Optional[PolicyFactory] = None,
    auditor_factory: Optional[PolicyFactory] = None,
    env_base_url: Optional[str] = None,
    show_trace: bool = False,
    max_rationale_chars: int = 80,
) -> List[InvestigatorTrainingSample]:
    """Run :func:`collect_episode` over every (task, seed) and concat results."""
    out: List[InvestigatorTrainingSample] = []
    n_eps = sum(len(v) for v in seeds_by_task.values())
    done = 0
    skipped = 0
    for task_id, seeds in seeds_by_task.items():
        for seed in seeds:
            done += 1
            print(f"  [{done}/{n_eps}] {task_id} seed={seed} ...", flush=True)
            try:
                samples = collect_episode(
                    hf_investigator=hf_investigator,
                    task_id=task_id,
                    seed=seed,
                    fraudster_factory=fraudster_factory,
                    auditor_factory=auditor_factory,
                    env_base_url=env_base_url,
                    show_trace=show_trace,
                    max_rationale_chars=max_rationale_chars,
                )
                out.extend(samples)
                if show_trace:
                    print(
                        f"      -> {len(samples)} usable Investigator turn(s) "
                        f"| fallback {hf_investigator.fallback_count}/"
                        f"{hf_investigator.call_count}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001 — log + continue
                skipped += 1
                print(
                    f"      SKIPPED ({type(exc).__name__}: {exc}). "
                    f"Continuing with next seed.",
                    flush=True,
                )
    if skipped:
        print(
            f"\n  Note: {skipped}/{n_eps} episodes were skipped due to "
            f"transport errors (commonly Ollama timeouts under low-VRAM "
            f"conditions). Set USE_OLLAMA_FRAUDSTER=False or "
            f"LLM_FRAUDSTER_RATIO=0.0 in §1 to avoid them.",
            flush=True,
        )
    return out


def samples_to_hf_dataset(samples: List[InvestigatorTrainingSample]) -> Any:
    """Convert :class:`InvestigatorTrainingSample` rows to ``datasets.Dataset``."""
    from datasets import Dataset
    return Dataset.from_list([s.to_dict() for s in samples])


__all__ = [
    "InvestigatorTrainingSample",
    "RecordingHFInvestigator",
    "TracingPolicy",
    "classify_action",
    "collect_dataset",
    "collect_episode",
    "records_to_samples",
    "samples_to_hf_dataset",
    "summarise_action",
]
