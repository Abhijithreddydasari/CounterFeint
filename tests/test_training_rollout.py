"""
Unit tests for :mod:`counterfeint.training.rollout`.

These exercise the per-step recorder, the action-class shaping math
inside :func:`records_to_samples`, and the side-column wiring without
spinning up an HF model or the FraudArena server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from counterfeint.models import AdReviewAction
from counterfeint.training.rollout import (
    RecordingHFInvestigator,
    TracingPolicy,
    classify_action,
    records_to_samples,
    summarise_action,
)


# ---------------------------------------------------------------------------
# Stand-in for HFInvestigator that exposes the same recording slots.
# ---------------------------------------------------------------------------


class _FakeInvestigator:
    """Minimal stand-in matching the HFInvestigator recording contract."""

    def __init__(self, plan: List[Dict[str, Any]]) -> None:
        self._plan = list(plan)
        self.fallback_count = 0
        self.call_count = 0
        self.last_prompt: Optional[str] = None
        self.last_completion: Optional[str] = None
        self.last_error = None

    def reset(self) -> None:
        self.fallback_count = 0
        self.call_count = 0
        self.last_prompt = None
        self.last_completion = None
        self.last_error = None

    def act(self, _observation: Dict[str, Any]) -> AdReviewAction:
        self.call_count += 1
        spec = self._plan.pop(0)
        # Match LLMPolicyBase.act() semantics: a fallback step leaves
        # last_prompt / last_completion as None (which is what the
        # recorder uses to flag the row).
        self.last_prompt = None
        self.last_completion = None
        if spec.get("fallback"):
            self.fallback_count += 1
        else:
            self.last_prompt = spec["prompt"]
            self.last_completion = spec["completion"]
        return spec["action"]


# ---------------------------------------------------------------------------
# RecordingHFInvestigator
# ---------------------------------------------------------------------------


class TestRecordingHFInvestigator:
    def test_records_one_entry_per_act(self) -> None:
        inner = _FakeInvestigator(
            plan=[
                {
                    "prompt": "p1", "completion": "c1",
                    "action": AdReviewAction(
                        action_type="investigate",
                        ad_id="ad_001",
                        investigation_target="payment_method",
                        rationale="x",
                    ),
                },
                {
                    "prompt": "p2", "completion": "c2",
                    "action": AdReviewAction(
                        action_type="verdict",
                        ad_id="ad_001",
                        verdict="reject",
                        confidence=0.9,
                        rationale="bad payment trail",
                    ),
                },
            ],
        )
        rec = RecordingHFInvestigator(inner)
        rec.reset()

        rec.act({})
        rec.act({})

        assert len(rec.step_records) == 2
        assert rec.step_records[0]["prompt"] == "p1"
        assert rec.step_records[0]["completion"] == "c1"
        assert rec.step_records[0]["fallback_used"] is False
        assert rec.step_records[1]["completion"] == "c2"
        assert rec.fallback_count == 0

    def test_fallback_step_marks_record_and_skips_text(self) -> None:
        inner = _FakeInvestigator(
            plan=[
                {
                    "fallback": True,
                    "action": AdReviewAction(
                        action_type="verdict",
                        ad_id="ad_001",
                        verdict="approve",
                        confidence=0.4,
                        rationale="fallback",
                    ),
                }
            ],
        )
        rec = RecordingHFInvestigator(inner)
        rec.reset()

        rec.act({})

        assert len(rec.step_records) == 1
        # _FakeInvestigator clears its slots on fallback to mimic the
        # base policy's behaviour ⇒ recorder marks fallback_used.
        assert rec.step_records[0]["fallback_used"] is True
        assert rec.fallback_count == 1


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------


class TestRecordsToSamples:
    @staticmethod
    def _record(prompt: str, completion: str, action_repr: str, step_idx: int) -> Dict[str, Any]:
        return {
            "step_idx": step_idx,
            "prompt": prompt,
            "completion": completion,
            "fallback_used": False,
            "action_repr": action_repr,
        }

    def test_mixed_actions_get_80_20_shaping_split(self) -> None:
        # 1 verdict + 4 investigate steps, total reward = 1.0.
        # Verdict should get 0.8 (the full 80% share, n_verdict=1).
        # Each investigate step should get 0.2 / 4 = 0.05.
        records = [
            self._record("p", "c", "AdReviewAction(action_type='investigate', ...)", 1),
            self._record("p", "c", "AdReviewAction(action_type='investigate', ...)", 2),
            self._record("p", "c", "AdReviewAction(action_type='investigate', ...)", 3),
            self._record("p", "c", "AdReviewAction(action_type='verdict', ...)", 4),
            self._record("p", "c", "AdReviewAction(action_type='investigate', ...)", 5),
        ]
        samples = records_to_samples(
            records,
            episode_result={
                "grader_score": 0.5,
                "rewards_by_role": {"investigator": 1.0},
                "end_reason": "queue_drained",
            },
            task_id="task_2",
            seed=42,
        )

        assert len(samples) == 5
        verdict = next(s for s in samples if s.metadata["action_class"] == "verdict")
        invests = [s for s in samples if s.metadata["action_class"] == "investigate"]
        assert verdict.reward == pytest.approx(0.8, rel=1e-6)
        assert len(invests) == 4
        for s in invests:
            assert s.reward == pytest.approx(0.05, rel=1e-6)
        # Total preserves the episode reward.
        assert sum(s.reward for s in samples) == pytest.approx(1.0, rel=1e-6)
        # Side columns wire through correctly.
        assert all(s.task_id == "task_2" for s in samples)
        assert all(s.seed == 42 for s in samples)
        assert verdict.terminal_grader_score == pytest.approx(0.5, rel=1e-6)

    def test_uniform_split_when_only_one_action_class(self) -> None:
        records = [
            self._record("p", "c", "AdReviewAction(action_type='investigate', ...)", 1),
            self._record("p", "c", "AdReviewAction(action_type='investigate', ...)", 2),
        ]
        samples = records_to_samples(
            records,
            episode_result={"grader_score": 0.0, "rewards_by_role": {"investigator": 0.6}},
            task_id="task_1",
            seed=1,
        )
        assert len(samples) == 2
        for s in samples:
            assert s.reward == pytest.approx(0.3, rel=1e-6)

    def test_fallback_only_records_are_dropped(self) -> None:
        records = [
            {
                "step_idx": 1, "prompt": None, "completion": None,
                "fallback_used": True,
                "action_repr": "AdReviewAction(action_type='verdict', ...)",
            },
        ]
        samples = records_to_samples(
            records,
            episode_result={"rewards_by_role": {"investigator": 1.0}},
            task_id="task_3",
            seed=7,
        )
        assert samples == []

    def test_link_accounts_counts_as_verdict_action_class(self) -> None:
        records = [
            self._record("p", "c", "AdReviewAction(action_type='link_accounts', ...)", 1),
            self._record("p", "c", "AdReviewAction(action_type='investigate', ...)", 2),
        ]
        samples = records_to_samples(
            records,
            episode_result={"rewards_by_role": {"investigator": 1.0}},
            task_id="task_3",
            seed=7,
        )
        link_sample = next(s for s in samples if s.step_idx == 1)
        invest_sample = next(s for s in samples if s.step_idx == 2)
        assert link_sample.metadata["action_class"] == "verdict"
        assert invest_sample.metadata["action_class"] == "investigate"
        assert link_sample.reward == pytest.approx(0.8, rel=1e-6)
        assert invest_sample.reward == pytest.approx(0.2, rel=1e-6)


class TestClassifyAction:
    def test_verdict_recognised(self) -> None:
        assert classify_action("AdReviewAction(action_type='verdict', verdict='reject')") == "verdict"

    def test_link_accounts_recognised_as_verdict(self) -> None:
        assert classify_action("AdReviewAction(action_type='link_accounts', linked_ad_id='ad_002')") == "verdict"

    def test_investigate_default(self) -> None:
        assert classify_action("AdReviewAction(action_type='investigate', ...)") == "investigate"

    def test_empty_input_default_investigate(self) -> None:
        assert classify_action(None) == "investigate"
        assert classify_action("") == "investigate"


# ---------------------------------------------------------------------------
# TracingPolicy + summarise_action are lightweight UX helpers; smoke test.
# ---------------------------------------------------------------------------


class TestSummariseAction:
    def test_handles_action_dict(self) -> None:
        out = summarise_action(
            "investigator",
            {"action_type": "verdict", "verdict": "reject", "confidence": 0.93,
             "rationale": "payment ring"},
        )
        assert "verdict" in out
        assert "reject" in out
        assert "@0.93" in out
        assert '"payment ring"' in out

    def test_handles_action_object(self) -> None:
        action = AdReviewAction(
            action_type="link_accounts",
            ad_id="ad_001",
            linked_ad_id="ad_002",
            link_reason="payment_id collision",
        )
        out = summarise_action("investigator", action)
        assert "link_accounts" in out
        assert "ad_002" in out
        assert "payment_id collision" in out

    def test_truncates_long_rationale(self) -> None:
        long = "x" * 300
        out = summarise_action(
            "investigator",
            {"action_type": "verdict", "verdict": "approve", "rationale": long},
            max_rationale_chars=20,
        )
        assert "..." in out
        # length budget includes leading/trailing quote chars.
        assert len(out) < 80


class TestTracingPolicyForwarding:
    def test_disabled_trace_is_silent_but_forwards(self, capsys) -> None:
        inner = _FakeInvestigator(
            plan=[
                {
                    "prompt": "p", "completion": "c",
                    "action": AdReviewAction(
                        action_type="verdict",
                        ad_id="ad_001",
                        verdict="approve",
                        confidence=0.5,
                        rationale="ok",
                    ),
                }
            ],
        )
        wrapped = TracingPolicy(inner, "investigator", enabled=False)
        action = wrapped.act({})

        captured = capsys.readouterr()
        assert captured.out == ""  # silent
        assert action.action_type == "verdict"
