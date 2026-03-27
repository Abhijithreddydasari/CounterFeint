"""Tests for the core AdFraudEnvironment."""

from ad_fraud_env.models import AdReviewAction, AdReviewObservation, AdFraudState
from ad_fraud_env.server.environment import AdFraudEnvironment


class TestReset:
    def test_reset_returns_observation(self):
        env = AdFraudEnvironment()
        obs = env.reset(seed=42, task_id="task_1")
        assert isinstance(obs, AdReviewObservation)
        assert obs.done is False
        assert obs.reward == 0.0
        assert len(obs.available_ads) == 5

    def test_reset_clears_state(self):
        env = AdFraudEnvironment()
        env.reset(seed=42, task_id="task_1")
        env.step(AdReviewAction(
            action_type="verdict", ad_id="ad_001",
            verdict="approve", confidence=0.9,
        ))
        obs = env.reset(seed=42, task_id="task_1")
        state = env.state
        assert state.step_count == 0
        assert state.reviewed_count == 0
        assert len(obs.available_ads) == 5

    def test_reset_different_tasks(self):
        env = AdFraudEnvironment()
        for task_id, expected in [("task_1", 5), ("task_2", 12), ("task_3", 20)]:
            obs = env.reset(seed=42, task_id=task_id)
            assert len(obs.available_ads) == expected


class TestStep:
    def test_investigate_returns_findings(self):
        env = AdFraudEnvironment()
        env.reset(seed=42, task_id="task_1")
        obs = env.step(AdReviewAction(
            action_type="investigate",
            ad_id="ad_001",
            investigation_target="advertiser_history",
        ))
        assert obs.done is False
        assert obs.reward == -0.02
        assert "Advertiser" in obs.feedback or "Investigation complete" in obs.feedback

    def test_verdict_correct_rejection(self):
        env = AdFraudEnvironment()
        env.reset(seed=42, task_id="task_1")
        fraud_ads = [
            a for a in env._episode.ads if a.ground_truth_label == "fraud"
        ]
        assert len(fraud_ads) > 0
        ad = fraud_ads[0]
        obs = env.step(AdReviewAction(
            action_type="verdict", ad_id=ad.ad_id,
            verdict="reject", confidence=0.9,
        ))
        assert obs.reward > 0

    def test_verdict_false_negative_penalty(self):
        env = AdFraudEnvironment()
        env.reset(seed=42, task_id="task_1")
        fraud_ads = [
            a for a in env._episode.ads if a.ground_truth_label == "fraud"
        ]
        ad = fraud_ads[0]
        obs = env.step(AdReviewAction(
            action_type="verdict", ad_id=ad.ad_id,
            verdict="approve", confidence=0.9,
        ))
        assert obs.reward < 0

    def test_duplicate_verdict_rejected(self):
        env = AdFraudEnvironment()
        env.reset(seed=42, task_id="task_1")
        env.step(AdReviewAction(
            action_type="verdict", ad_id="ad_001",
            verdict="approve", confidence=0.5,
        ))
        obs = env.step(AdReviewAction(
            action_type="verdict", ad_id="ad_001",
            verdict="reject", confidence=0.9,
        ))
        assert obs.reward == -0.02

    def test_invalid_ad_id(self):
        env = AdFraudEnvironment()
        env.reset(seed=42, task_id="task_1")
        obs = env.step(AdReviewAction(
            action_type="investigate", ad_id="ad_999",
            investigation_target="landing_page",
        ))
        assert obs.reward == -0.05
        assert "Invalid" in obs.feedback

    def test_episode_ends_when_all_reviewed(self):
        env = AdFraudEnvironment()
        obs = env.reset(seed=42, task_id="task_1")
        for ad_id in list(obs.available_ads):
            obs = env.step(AdReviewAction(
                action_type="verdict", ad_id=ad_id,
                verdict="reject", confidence=0.5,
            ))
        assert obs.done is True

    def test_step_after_done_returns_done(self):
        env = AdFraudEnvironment()
        obs = env.reset(seed=42, task_id="task_1")
        for ad_id in list(obs.available_ads):
            obs = env.step(AdReviewAction(
                action_type="verdict", ad_id=ad_id,
                verdict="reject", confidence=0.5,
            ))
        obs = env.step(AdReviewAction(
            action_type="investigate", ad_id="ad_001",
            investigation_target="landing_page",
        ))
        assert obs.done is True
        assert "already complete" in obs.feedback.lower()


class TestState:
    def test_state_tracks_progress(self):
        env = AdFraudEnvironment()
        env.reset(seed=42, task_id="task_1")
        state = env.state
        assert state.task_id == "task_1"
        assert state.total_ads == 5
        assert state.remaining_budget == 25
        assert state.step_count == 0

        env.step(AdReviewAction(
            action_type="investigate", ad_id="ad_001",
            investigation_target="landing_page",
        ))
        state = env.state
        assert state.step_count == 1
        assert state.remaining_budget == 24

    def test_grader_score_set_on_completion(self):
        env = AdFraudEnvironment()
        obs = env.reset(seed=42, task_id="task_1")
        for ad_id in list(obs.available_ads):
            env.step(AdReviewAction(
                action_type="verdict", ad_id=ad_id,
                verdict="reject", confidence=0.5,
            ))
        state = env.state
        assert state.grader_score is not None
        assert 0.0 <= state.grader_score <= 1.0


class TestAntiExploit:
    def test_always_reject_scores_poorly(self):
        """Always-reject on task_2 (5 legit / 5 fraud / 2 escalate) should be punished."""
        env = AdFraudEnvironment()
        obs = env.reset(seed=42, task_id="task_2")
        for ad_id in list(obs.available_ads):
            env.step(AdReviewAction(
                action_type="verdict", ad_id=ad_id,
                verdict="reject", confidence=0.9,
            ))
        score = env.state.grader_score
        assert score < 0.7, f"Always-reject should score poorly, got {score}"

    def test_always_escalate_scores_poorly(self):
        env = AdFraudEnvironment()
        obs = env.reset(seed=42, task_id="task_1")
        for ad_id in list(obs.available_ads):
            env.step(AdReviewAction(
                action_type="verdict", ad_id=ad_id,
                verdict="escalate", confidence=0.5,
            ))
        score = env.state.grader_score
        assert score < 0.7, f"Always-escalate should score poorly, got {score}"
