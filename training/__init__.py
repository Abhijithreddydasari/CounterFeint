"""Training-time helpers for CounterFeint (rollout splits, GRPO config).

Kept in a separate top-level package so the runtime Docker image (which
does not need any training code) can ignore this directory entirely.
"""

from .rollout_config import (
    DEFAULT_GRPO_CONFIG,
    DEFAULT_ROLLOUT_SPLIT,
    GRPOConfig,
    RolloutSplit,
    build_fraudster_factory_for_rollout,
    sample_fraudster_kind,
)

__all__ = [
    "DEFAULT_GRPO_CONFIG",
    "DEFAULT_ROLLOUT_SPLIT",
    "GRPOConfig",
    "RolloutSplit",
    "build_fraudster_factory_for_rollout",
    "sample_fraudster_kind",
]
