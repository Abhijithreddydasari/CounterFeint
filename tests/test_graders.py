"""Tests for the grading system."""

from ad_fraud_env.graders.base_grader import (
    BaseGrader,
    EpisodeRecord,
    LinkResult,
    VerdictResult,
    grade_episode,
)


def _make_record(
    task_id: str = "task_1",
    verdicts: list | None = None,
    links: list | None = None,
    total_steps: int = 5,
    action_budget: int = 25,
    ads_metadata: list | None = None,
    n_fraud_rings: int = 0,
) -> EpisodeRecord:
    if verdicts is None:
        verdicts = []
    if links is None:
        links = []
    if ads_metadata is None:
        ads_metadata = [
            {"ad_id": v.ad_id, "severity": 0.8, "ground_truth": v.ground_truth}
            for v in verdicts
        ]
    return EpisodeRecord(
        task_id=task_id,
        total_steps=total_steps,
        action_budget=action_budget,
        verdicts=verdicts,
        links=links,
        ads_metadata=ads_metadata,
        n_fraud_rings=n_fraud_rings,
    )


class TestGraderScoreRange:
    def test_scores_in_valid_range(self):
        verdicts = [
            VerdictResult("ad_001", "reject", 0.9, "fraud"),
            VerdictResult("ad_002", "approve", 0.9, "legit"),
            VerdictResult("ad_003", "reject", 0.8, "fraud"),
        ]
        record = _make_record(verdicts=verdicts, total_steps=3)
        score = grade_episode(record)
        assert 0.0 <= score <= 1.0

    def test_perfect_score_is_high(self):
        verdicts = [
            VerdictResult("ad_001", "reject", 0.95, "fraud"),
            VerdictResult("ad_002", "approve", 0.95, "legit"),
            VerdictResult("ad_003", "reject", 0.95, "fraud"),
            VerdictResult("ad_004", "approve", 0.95, "legit"),
            VerdictResult("ad_005", "reject", 0.95, "fraud"),
        ]
        record = _make_record(verdicts=verdicts, total_steps=5)
        score = grade_episode(record)
        assert score > 0.7, f"Perfect verdicts should score high, got {score}"

    def test_all_wrong_scores_low(self):
        verdicts = [
            VerdictResult("ad_001", "approve", 0.9, "fraud"),
            VerdictResult("ad_002", "reject", 0.9, "legit"),
            VerdictResult("ad_003", "approve", 0.9, "fraud"),
        ]
        record = _make_record(verdicts=verdicts, total_steps=3)
        score = grade_episode(record)
        assert score < 0.3, f"All wrong verdicts should score low, got {score}"


class TestTask2Grader:
    def test_calibration_bonus(self):
        well_calibrated = [
            VerdictResult("ad_001", "reject", 0.9, "fraud"),
            VerdictResult("ad_002", "approve", 0.9, "legit"),
            VerdictResult("ad_003", "reject", 0.8, "fraud"),
            VerdictResult("ad_004", "approve", 0.85, "legit"),
        ]
        poorly_calibrated = [
            VerdictResult("ad_001", "reject", 0.2, "fraud"),
            VerdictResult("ad_002", "approve", 0.2, "legit"),
            VerdictResult("ad_003", "reject", 0.2, "fraud"),
            VerdictResult("ad_004", "approve", 0.2, "legit"),
        ]
        r1 = _make_record(task_id="task_2", verdicts=well_calibrated, total_steps=4, action_budget=30)
        r2 = _make_record(task_id="task_2", verdicts=poorly_calibrated, total_steps=4, action_budget=30)
        s1 = grade_episode(r1)
        s2 = grade_episode(r2)
        assert s1 >= s2, f"Well calibrated ({s1}) should score >= poorly calibrated ({s2})"


class TestTask3Grader:
    def test_network_link_bonus(self):
        verdicts = [
            VerdictResult("ad_001", "reject", 0.9, "fraud"),
            VerdictResult("ad_002", "reject", 0.9, "fraud"),
            VerdictResult("ad_003", "reject", 0.9, "fraud"),
        ]
        links_correct = [
            LinkResult("ad_001", "ad_002", True),
            LinkResult("ad_002", "ad_003", True),
        ]
        links_wrong = [
            LinkResult("ad_001", "ad_002", False),
        ]

        r1 = _make_record(
            task_id="task_3", verdicts=verdicts, links=links_correct,
            total_steps=5, action_budget=40, n_fraud_rings=1,
        )
        r2 = _make_record(
            task_id="task_3", verdicts=verdicts, links=links_wrong,
            total_steps=4, action_budget=40, n_fraud_rings=1,
        )
        s1 = grade_episode(r1)
        s2 = grade_episode(r2)
        assert s1 > s2, f"Correct links ({s1}) should score > wrong links ({s2})"
