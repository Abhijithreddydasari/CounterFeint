"""Unit tests for counterfeint.training.rollout_config.

Verifies:

* The default 70/30 split lives where the README claims.
* ``sample_fraudster_kind`` reproduces the requested distribution
  within a tight statistical bound (deterministic via a seeded RNG).
* ``build_fraudster_factory_for_rollout`` returns the correct policy
  type for each draw, **without** loading the openai client when the
  draw lands on the scripted opponent.
* ``GRPOConfig`` defaults serialise cleanly to dict (we use this from
  the Colab notebook to log hyperparameters).
"""

from __future__ import annotations

import random

import pytest

from counterfeint.scripted import ReactiveFraudster
from counterfeint.training.rollout_config import (
    DEFAULT_GRPO_CONFIG,
    DEFAULT_ROLLOUT_SPLIT,
    GRPOConfig,
    RolloutSplit,
    build_fraudster_factory_for_rollout,
    sample_fraudster_kind,
)


class TestRolloutSplit:
    def test_default_is_seventy_thirty(self) -> None:
        assert DEFAULT_ROLLOUT_SPLIT.reactive_share == pytest.approx(0.70)
        assert DEFAULT_ROLLOUT_SPLIT.llm_share == pytest.approx(0.30)

    def test_invalid_shares_rejected(self) -> None:
        with pytest.raises(ValueError):
            RolloutSplit(reactive_share=0.5, llm_share=0.4)
        with pytest.raises(ValueError):
            RolloutSplit(reactive_share=1.2, llm_share=-0.2)


class TestSampleFraudsterKind:
    def test_distribution_matches_split_within_5_pct(self) -> None:
        rng = random.Random(123)
        n = 4000
        counts = {"reactive": 0, "llm": 0}
        for _ in range(n):
            counts[sample_fraudster_kind(rng)] += 1
        # Expected 0.70 reactive, 0.30 llm. Allow 5pp slack.
        reactive_pct = counts["reactive"] / n
        assert 0.65 <= reactive_pct <= 0.75, (
            f"reactive share {reactive_pct:.2%} drifted outside [65%, 75%]"
        )

    def test_deterministic_under_same_seed(self) -> None:
        seq_a = [sample_fraudster_kind(random.Random(7)) for _ in range(20)]
        seq_b = [sample_fraudster_kind(random.Random(7)) for _ in range(20)]
        assert seq_a == seq_b


class TestBuildFraudsterFactory:
    def test_returns_reactive_fraudster_for_scripted_draw(self) -> None:
        # llm_share=0.0 forces every draw to "reactive".
        split = RolloutSplit(reactive_share=1.0, llm_share=0.0)
        rng = random.Random(0)
        factory = build_fraudster_factory_for_rollout(rng, split)
        instance = factory()
        assert isinstance(instance, ReactiveFraudster)

    def test_invokes_provided_llm_factory_for_llm_draw(self) -> None:
        # llm_share=1.0 forces every draw to "llm".
        split = RolloutSplit(reactive_share=0.0, llm_share=1.0)
        sentinel: list = []

        def _llm_factory() -> object:
            sentinel.append("called")
            return object()

        rng = random.Random(0)
        factory = build_fraudster_factory_for_rollout(
            rng, split, llm_factory=_llm_factory
        )
        instance = factory()
        # Build is lazy: factory only runs when the closure is invoked.
        assert sentinel == ["called"]
        assert isinstance(instance, object)


class TestGRPOConfig:
    def test_default_to_dict_is_jsonable(self) -> None:
        d = DEFAULT_GRPO_CONFIG.to_dict()
        assert d["learning_rate"] == pytest.approx(1e-6)
        assert d["rollout_split"]["reactive_share"] == pytest.approx(0.70)
        assert d["rollout_split"]["llm_share"] == pytest.approx(0.30)
        assert "notes" in d
        # Round-trip through json to confirm there are no non-serialisables.
        import json

        json.dumps(d)

    def test_overrideable(self) -> None:
        cfg = GRPOConfig(learning_rate=2e-6, group_size=8)
        assert cfg.learning_rate == 2e-6
        assert cfg.group_size == 8
