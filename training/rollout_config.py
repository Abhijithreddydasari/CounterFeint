"""
CounterFeint training rollout configuration.

Encapsulates the **opponent-mixing curriculum** used during Investigator
training. The Investigator is the only policy whose weights move; every
opponent (Fraudster, Auditor) stays frozen so we never run a
two-LLM-co-training loop. Within each batch, Fraudster rollouts are
drawn from a fixed split:

    +-------------+--------------------------------+--------+
    | Share        | Opponent                       | Why    |
    +=============+================================+========+
    | 70 %         | ReactiveFraudster (programmatic) | Stable |
    | 30 %         | LLMFraudster (frozen Llama)      | Open   |
    +-------------+--------------------------------+--------+

This keeps the gradient signal stable while still exposing the trained
Investigator to genuine open-vocabulary adversarial pressure (so that
the Track A reasoning audit and ``n_fraud_leaks`` headline metric
both improve, not just the easy procedural-fraud cases).

The GRPO hyperparameters here are **placeholder values**; final
learning rate / KL coefficient / group size land at onsite compute
day after a sweep.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional

try:
    from ..scripted import ReactiveFraudster
except ImportError:  # pragma: no cover - script-level fallback
    from counterfeint.scripted import ReactiveFraudster  # type: ignore[no-redef]


FraudsterKind = Literal["reactive", "llm"]


# ---------------------------------------------------------------------------
# Rollout split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RolloutSplit:
    """Probability distribution over Fraudster opponent kinds.

    Probabilities must sum to 1.0 (validated in :meth:`__post_init__`).
    Use the helper :func:`sample_fraudster_kind` to draw a kind for a
    single rollout from a deterministic ``random.Random``.
    """

    reactive_share: float = 0.70
    llm_share: float = 0.30

    def __post_init__(self) -> None:
        total = self.reactive_share + self.llm_share
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"RolloutSplit shares must sum to 1.0 (got {total:.3f})"
            )
        for share in (self.reactive_share, self.llm_share):
            if not (0.0 <= share <= 1.0):
                raise ValueError(
                    f"RolloutSplit share out of range [0,1]: {share}"
                )


DEFAULT_ROLLOUT_SPLIT = RolloutSplit(reactive_share=0.70, llm_share=0.30)


def sample_fraudster_kind(
    rng: random.Random,
    split: RolloutSplit = DEFAULT_ROLLOUT_SPLIT,
) -> FraudsterKind:
    """Draw a Fraudster opponent kind for a single rollout."""
    return "llm" if rng.random() < split.llm_share else "reactive"


def build_fraudster_factory_for_rollout(
    rng: random.Random,
    split: RolloutSplit = DEFAULT_ROLLOUT_SPLIT,
    *,
    llm_factory: Optional[Callable[[], Any]] = None,
) -> Callable[[], Any]:
    """Return a zero-arg callable that builds the Fraudster for one rollout.

    The returned factory is consumed once by
    :func:`counterfeint.inference.run_three_agent_episode`. Sampling is
    done **eagerly** here (using ``rng``) so the caller can log the
    chosen kind alongside other batch metadata; the factory closure
    itself is deterministic.
    """
    kind = sample_fraudster_kind(rng, split)
    if kind == "llm":
        # Lazy-import to keep training-time openai dep optional in CI.
        if llm_factory is None:
            try:
                from ..agents import LLMFraudster  # type: ignore[import-not-found]
            except ImportError:
                from counterfeint.agents import LLMFraudster  # type: ignore[no-redef]

            def llm_factory() -> Any:
                return LLMFraudster()

        return llm_factory
    return lambda: ReactiveFraudster(seed=rng.randrange(0, 2**31))


# ---------------------------------------------------------------------------
# GRPO config (placeholder hyperparameters)
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    """Placeholder GRPO hyperparameters. Tune onsite."""

    learning_rate: float = 1e-6
    group_size: int = 4
    kl_coefficient: float = 0.01
    max_response_tokens: int = 512
    rollouts_per_step: int = 32
    grad_accum_steps: int = 4
    eval_every_n_steps: int = 25
    seed: int = 7
    rollout_split: RolloutSplit = field(default_factory=lambda: DEFAULT_ROLLOUT_SPLIT)
    notes: str = (
        "Placeholder values — final hyperparameters set onsite after a "
        "small sweep. Keep KL low enough that the Investigator can drift "
        "off the Llama-3 prior toward the action-budget-aware behaviour "
        "the grader rewards, but high enough that Track A audits don't "
        "collapse into terse single-token rationales."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "group_size": self.group_size,
            "kl_coefficient": self.kl_coefficient,
            "max_response_tokens": self.max_response_tokens,
            "rollouts_per_step": self.rollouts_per_step,
            "grad_accum_steps": self.grad_accum_steps,
            "eval_every_n_steps": self.eval_every_n_steps,
            "seed": self.seed,
            "rollout_split": {
                "reactive_share": self.rollout_split.reactive_share,
                "llm_share": self.rollout_split.llm_share,
            },
            "notes": self.notes,
        }


DEFAULT_GRPO_CONFIG = GRPOConfig()
