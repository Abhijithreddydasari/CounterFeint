"""
Tests for the RefereeEnvironment three-agent state machine (in-process).

Covers:
  * turn interleaving (fraudster -> investigator -> fraudster -> ... -> audit)
  * dynamic queue growth (Fraudster proposals reach Investigator)
  * reactive signal (Fraudster observation reflects Investigator's verdicts)
  * phase guards (role-mismatched actions raise PermissionError)
  * three end paths: commit_final, investigator_done, max_rounds
  * grader_score is populated exactly when phase == "done"
"""

from __future__ import annotations

import pytest

from counterfeint.models import (
    AdReviewAction,
    AuditorAction,
    FraudsterAction,
    RefereeState,
)
from counterfeint.scripted import (
    HeuristicAuditor,
    ReactiveFraudster,
    ScriptedInvestigator,
)
from counterfeint.server.referee import RefereeEnvironment


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_referee(**reset_kwargs):
    env = RefereeEnvironment()
    reset_kwargs.setdefault("task_id", "task_1")
    reset_kwargs.setdefault("seed", 42)
    env.reset_match(**reset_kwargs)
    return env


def a_propose(category: str = "fake_giveaway", *, copy: str = "Free iPhone - tap now!"):
    return FraudsterAction(
        action_type="propose_ad",
        ad_copy=copy,
        category=category,
        landing_page_blurb="limited-time giveaway details",
        targeting_summary="adults 18-45",
    )


def a_end_turn():
    return FraudsterAction(action_type="end_turn")


def a_commit():
    return FraudsterAction(action_type="commit_final")


def a_investigate(ad_id: str, target: str = "landing_page"):
    return AdReviewAction(
        action_type="investigate", ad_id=ad_id, investigation_target=target
    )


def a_verdict(ad_id: str, verdict: str = "reject", conf: float = 0.8):
    return AdReviewAction(
        action_type="verdict", ad_id=ad_id, verdict=verdict, confidence=conf,
        rationale=f"Verdict for {ad_id}: {verdict} (confidence {conf})",
    )


def a_submit_audit():
    return AuditorAction(
        action_type="submit_audit_report",
        audit_report={
            "track_a_flags": [],
            "track_b_flags": [],
            "investigator_audit_score": 1.0,
            "fraudster_plausibility_score": 1.0,
            "notes": "test",
        },
    )


# ---------------------------------------------------------------------------
# Turn interleaving + dynamic queue
# ---------------------------------------------------------------------------


class TestTurnInterleaving:
    def test_starts_in_fraudster_turn_round_1(self):
        env = make_referee()
        assert env.phase == "fraudster_turn"
        assert env.state.round_number == 1
        assert env.state.proposals_used == 0

    def test_fraudster_end_turn_flips_to_investigator(self):
        env = make_referee()
        obs = env.step_as_fraudster(a_end_turn())
        assert env.phase == "investigator_turn"
        assert obs.done is False

    def test_fraudster_action_cap_auto_ends_turn(self):
        env = make_referee(max_fraudster_actions_per_turn=2, max_proposals=5)
        env.step_as_fraudster(a_propose("fake_giveaway", copy="ad one"))
        assert env.phase == "fraudster_turn"
        env.step_as_fraudster(a_propose("fake_crypto", copy="ad two"))
        assert env.phase == "investigator_turn"

    def test_investigator_action_cap_flips_to_fraudster_next_round(self):
        env = make_referee(
            max_fraudster_actions_per_turn=3,
            max_investigator_actions_per_turn=3,
        )
        env.step_as_fraudster(a_end_turn())
        assert env.phase == "investigator_turn"
        available = env.build_investigator_observation().available_ads
        for ad_id in available[:3]:
            env.step_as_investigator(a_verdict(ad_id))
        assert env.phase == "fraudster_turn"
        assert env.state.round_number == 2

    def test_fraudster_proposal_reaches_investigator_queue(self):
        env = make_referee()
        before = env.build_investigator_observation().available_ads
        env.step_as_fraudster(a_propose("fake_giveaway"))
        env.step_as_fraudster(a_end_turn())
        after = env.build_investigator_observation().available_ads
        assert len(after) == len(before) + 1


# ---------------------------------------------------------------------------
# Reactive signal — Fraudster sees Investigator's verdicts
# ---------------------------------------------------------------------------


class TestReactiveSignal:
    def test_fraudster_observation_reflects_investigator_verdicts(self):
        env = make_referee(
            max_fraudster_actions_per_turn=3,
            max_investigator_actions_per_turn=3,
        )
        env.step_as_fraudster(a_propose("fake_giveaway", copy="suspicious"))
        proposed_ad_id = env._proposal_slot_to_ad_id[0]
        env.step_as_fraudster(a_end_turn())
        env.step_as_investigator(a_verdict(proposed_ad_id, verdict="reject", conf=0.9))

        remaining = [
            ad_id
            for ad_id in env.build_investigator_observation().available_ads
            if ad_id != proposed_ad_id
        ]
        for ad_id in remaining[:2]:
            env.step_as_investigator(a_verdict(ad_id, verdict="approve", conf=0.7))
        # expected phase flip back to fraudster_turn after action cap
        assert env.phase == "fraudster_turn"

        fraud_obs = env.build_fraudster_observation()
        verdict_map = {v["ad_id"]: v for v in fraud_obs.prior_verdicts}
        assert proposed_ad_id in verdict_map
        assert verdict_map[proposed_ad_id]["verdict"] == "reject"
        assert verdict_map[proposed_ad_id].get("was_my_proposal") is True
        assert any(v["verdict"] == "approve" for v in fraud_obs.prior_verdicts)

    def test_investigation_targets_used_are_visible_to_fraudster(self):
        env = make_referee(
            max_fraudster_actions_per_turn=3,
            max_investigator_actions_per_turn=3,
        )
        env.step_as_fraudster(a_end_turn())
        target_ad = env.build_investigator_observation().available_ads[0]
        env.step_as_investigator(a_investigate(target_ad, "landing_page"))
        env.step_as_investigator(a_verdict(target_ad, verdict="reject", conf=0.9))
        env.step_as_investigator(a_verdict(
            env.build_investigator_observation().available_ads[0],
            verdict="approve", conf=0.7,
        ))
        assert env.phase == "fraudster_turn"
        fraud_obs = env.build_fraudster_observation()
        assert target_ad in fraud_obs.investigation_targets_used
        assert "landing_page" in fraud_obs.investigation_targets_used[target_ad]


# ---------------------------------------------------------------------------
# Phase guards
# ---------------------------------------------------------------------------


class TestPhaseGuards:
    def test_investigator_during_fraudster_turn_raises(self):
        env = make_referee()
        with pytest.raises(PermissionError):
            env.step_as_investigator(a_verdict("ad_001"))

    def test_fraudster_during_investigator_turn_raises(self):
        env = make_referee()
        env.step_as_fraudster(a_end_turn())
        assert env.phase == "investigator_turn"
        with pytest.raises(PermissionError):
            env.step_as_fraudster(a_propose())

    def test_auditor_during_fraudster_turn_raises(self):
        env = make_referee()
        with pytest.raises(PermissionError):
            env.step_as_auditor(a_submit_audit())


# ---------------------------------------------------------------------------
# End paths
# ---------------------------------------------------------------------------


class TestEndPaths:
    def _advance_to_audit(self, env: RefereeEnvironment) -> None:
        loops = 0
        while env.phase not in ("audit_phase", "done"):
            if loops > 200:
                raise AssertionError("episode failed to advance after 200 steps")
            loops += 1
            if env.phase == "fraudster_turn":
                obs = env.build_fraudster_observation()
                policy = ReactiveFraudster(seed=1)
                action = policy.act(obs.model_dump())
                env.step_as_fraudster(action)
            elif env.phase == "investigator_turn":
                obs = env.build_investigator_observation()
                policy = ScriptedInvestigator()
                action = policy.act(obs.model_dump())
                env.step_as_investigator(action)
            else:
                break

    def test_commit_final_jumps_to_audit(self):
        env = make_referee()
        env.step_as_fraudster(a_commit())
        assert env.phase == "audit_phase"
        assert env.state.fraudster_committed is True
        assert env.state.end_reason == "commit_final"

    def test_investigator_done_jumps_to_audit(self):
        env = make_referee(
            max_fraudster_actions_per_turn=1, max_proposals=0,
            max_investigator_actions_per_turn=10, max_rounds=10,
        )
        env.step_as_fraudster(a_end_turn())
        for ad_id in list(env.build_investigator_observation().available_ads):
            env.step_as_investigator(a_verdict(ad_id))
        assert env.phase == "audit_phase"
        assert env.state.end_reason in ("investigator_done", "all_decided")

    def test_max_rounds_jumps_to_audit(self):
        env = make_referee(
            max_rounds=1,
            max_fraudster_actions_per_turn=1,
            max_investigator_actions_per_turn=2,
        )
        env.step_as_fraudster(a_end_turn())
        available = env.build_investigator_observation().available_ads
        for ad_id in available[:2]:
            env.step_as_investigator(a_verdict(ad_id))
        assert env.phase == "audit_phase"
        assert env.state.end_reason in ("max_rounds", "investigator_done", "all_decided")

    def test_audit_submit_flips_to_done_and_sets_grader_score(self):
        env = make_referee()
        env.step_as_fraudster(a_commit())
        assert env.phase == "audit_phase"
        obs = env.step_as_auditor(a_submit_audit())
        assert env.phase == "done"
        assert obs.done is True
        state = env.state
        assert state.grader_score is not None
        assert 0.0 <= state.grader_score <= 1.0


# ---------------------------------------------------------------------------
# Full scripted episode (sanity)
# ---------------------------------------------------------------------------


class TestScriptedFullRun:
    def test_full_episode_terminates_cleanly(self):
        env = make_referee(max_rounds=3)
        fraud = ReactiveFraudster(seed=5)
        inv = ScriptedInvestigator()
        aud = HeuristicAuditor()

        loops = 0
        while env.phase != "done":
            loops += 1
            assert loops <= 400, "episode did not terminate in a reasonable number of steps"

            if env.phase == "fraudster_turn":
                obs = env.build_fraudster_observation().model_dump()
                env.step_as_fraudster(fraud.act(obs))
            elif env.phase == "investigator_turn":
                obs = env.build_investigator_observation().model_dump()
                env.step_as_investigator(inv.act(obs))
            elif env.phase == "audit_phase":
                obs = env.build_auditor_observation().model_dump()
                env.step_as_auditor(aud.act(obs))
            else:
                raise AssertionError(f"unexpected phase {env.phase}")

        state: RefereeState = env.state
        assert state.grader_score is not None
        assert state.audit_report is not None
        assert state.phase == "done"
        assert state.end_reason in (
            "commit_final", "all_decided", "max_rounds", "investigator_done",
        )


class TestTaskConfigCurriculum:
    """Verify TaskConfig knobs flow into the Referee as the default curriculum."""

    def test_task_1_uses_novice_fraudster_budget(self):
        env = RefereeEnvironment()
        env.reset_match(task_id="task_1", seed=42)
        assert env.state.max_rounds == 4
        assert env.state.max_proposals == 5
        allowed = env.build_fraudster_observation().allowed_categories
        assert "fake_giveaway" in allowed
        assert "miracle_cure" in allowed
        assert "counterfeit_goods" not in allowed, (
            "Task 1 should restrict the Fraudster to easy fraud templates"
        )
        assert "network_crypto" not in allowed

    def test_task_2_adds_mid_tier_categories(self):
        env = RefereeEnvironment()
        env.reset_match(task_id="task_2", seed=42)
        assert env.state.max_proposals == 6
        allowed = env.build_fraudster_observation().allowed_categories
        assert "counterfeit_goods" in allowed
        assert "fake_crypto" in allowed
        assert "clone_brand" in allowed
        assert "network_crypto" not in allowed, (
            "Task 2 should not yet allow ring-level categories"
        )

    def test_task_3_opens_full_palette(self):
        env = RefereeEnvironment()
        env.reset_match(task_id="task_3", seed=42)
        assert env.state.max_rounds == 5
        assert env.state.max_proposals == 7
        assert env._max_investigator_actions_per_turn == 7  # not surfaced in RefereeState
        allowed = env.build_fraudster_observation().allowed_categories
        assert "network_crypto" in allowed
        assert "network_ecommerce" in allowed

    def test_explicit_kwarg_still_overrides_task_config(self):
        env = RefereeEnvironment()
        env.reset_match(task_id="task_3", seed=42, max_proposals=2)
        assert env.state.max_proposals == 2, (
            "Explicit reset_match kwargs must still trump the task curriculum"
        )
