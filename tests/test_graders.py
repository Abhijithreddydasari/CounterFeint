"""Tests for the grading system."""

from counterfeint.graders.base_grader import (
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
    ring_sizes: list | None = None,
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
        ring_sizes=ring_sizes,
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
            total_steps=5, action_budget=35, n_fraud_rings=1,
            ring_sizes=[3],
        )
        r2 = _make_record(
            task_id="task_3", verdicts=verdicts, links=links_wrong,
            total_steps=4, action_budget=35, n_fraud_rings=1,
            ring_sizes=[3],
        )
        s1 = grade_episode(r1)
        s2 = grade_episode(r2)
        assert s1 > s2, f"Correct links ({s1}) should score > wrong links ({s2})"

    def test_graph_based_scoring(self):
        """Task 3 grader should use edge coverage from ground truth graph."""
        verdicts = [
            VerdictResult("ad_001", "reject", 0.9, "fraud"),
            VerdictResult("ad_002", "reject", 0.9, "fraud"),
            VerdictResult("ad_003", "reject", 0.9, "fraud"),
            VerdictResult("ad_004", "reject", 0.9, "fraud"),
        ]
        # 4 ads in a ring of 4 -> 6 ground truth edges
        # Discover 3 of them
        links = [
            LinkResult("ad_001", "ad_002", True),
            LinkResult("ad_002", "ad_003", True),
            LinkResult("ad_003", "ad_004", True),
        ]
        r = _make_record(
            task_id="task_3", verdicts=verdicts, links=links,
            total_steps=7, action_budget=35, n_fraud_rings=1,
            ring_sizes=[4],
        )
        score = grade_episode(r)
        assert 0.0 <= score <= 1.0

    def test_false_links_penalized(self):
        """False link_accounts should reduce score."""
        verdicts = [
            VerdictResult("ad_001", "reject", 0.9, "fraud"),
            VerdictResult("ad_002", "approve", 0.9, "legit"),
        ]
        no_links = _make_record(
            task_id="task_3", verdicts=verdicts, links=[],
            total_steps=2, action_budget=35, n_fraud_rings=1,
            ring_sizes=[3],
        )
        false_links = _make_record(
            task_id="task_3", verdicts=verdicts,
            links=[LinkResult("ad_001", "ad_002", False)],
            total_steps=3, action_budget=35, n_fraud_rings=1,
            ring_sizes=[3],
        )
        s_none = grade_episode(no_links)
        s_false = grade_episode(false_links)
        assert s_none >= s_false, (
            f"No links ({s_none}) should score >= false links ({s_false})"
        )

    def test_coverage_bonus(self):
        """Agents that review more ads should get a coverage bonus."""
        few_verdicts = [
            VerdictResult("ad_001", "reject", 0.9, "fraud"),
        ]
        many_verdicts = [
            VerdictResult("ad_001", "reject", 0.9, "fraud"),
            VerdictResult("ad_002", "approve", 0.9, "legit"),
            VerdictResult("ad_003", "reject", 0.9, "fraud"),
            VerdictResult("ad_004", "approve", 0.9, "legit"),
        ]
        ads_meta = [
            {"ad_id": "ad_001", "severity": 0.8, "ground_truth": "fraud"},
            {"ad_id": "ad_002", "severity": 0.5, "ground_truth": "legit"},
            {"ad_id": "ad_003", "severity": 0.8, "ground_truth": "fraud"},
            {"ad_id": "ad_004", "severity": 0.5, "ground_truth": "legit"},
            {"ad_id": "ad_005", "severity": 0.5, "ground_truth": "legit"},
        ]
        r_few = _make_record(
            task_id="task_3", verdicts=few_verdicts, total_steps=1,
            action_budget=35, ads_metadata=ads_meta, ring_sizes=[3],
        )
        r_many = _make_record(
            task_id="task_3", verdicts=many_verdicts, total_steps=4,
            action_budget=35, ads_metadata=ads_meta, ring_sizes=[3],
        )
        s_few = grade_episode(r_few)
        s_many = grade_episode(r_many)
        assert s_many > s_few, (
            f"More coverage ({s_many}) should score > less coverage ({s_few})"
        )
