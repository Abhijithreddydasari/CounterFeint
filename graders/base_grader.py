"""
Shared grading logic for all tasks.

Each grader produces a 0.0-1.0 score by normalizing raw reward
between theoretical worst and best cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class VerdictResult:
    ad_id: str
    verdict: str          # approve, reject, escalate
    confidence: float
    ground_truth: str     # fraud, legit, escalate
    auto_approved: bool = False


@dataclass
class LinkResult:
    ad_id_1: str
    ad_id_2: str
    correct: bool


@dataclass
class EpisodeRecord:
    """All data needed for grading a completed episode."""
    task_id: str
    total_steps: int
    action_budget: int
    verdicts: List[VerdictResult]
    links: List[LinkResult]
    ads_metadata: List[Dict[str, Any]]  # [{ad_id, ground_truth, severity, ...}]
    n_fraud_rings: int = 0
    ring_sizes: List[int] = None

    def __post_init__(self):
        if self.ring_sizes is None:
            self.ring_sizes = []


class BaseGrader(ABC):
    """Abstract grader that scores an episode 0.0-1.0."""

    @abstractmethod
    def grade(self, record: EpisodeRecord) -> float:
        """Return a score in [0.0, 1.0]."""
        ...

    def _count_correct_verdicts(self, verdicts: List[VerdictResult]) -> int:
        return sum(1 for v in verdicts if self._is_correct(v))

    def _count_false_positives(self, verdicts: List[VerdictResult]) -> int:
        return sum(
            1 for v in verdicts
            if v.verdict == "reject" and v.ground_truth == "legit"
        )

    def _count_false_negatives(self, verdicts: List[VerdictResult]) -> int:
        return sum(
            1 for v in verdicts
            if v.verdict == "approve" and v.ground_truth == "fraud"
        )

    def _is_correct(self, v: VerdictResult) -> bool:
        return (
            (v.verdict == "reject" and v.ground_truth == "fraud")
            or (v.verdict == "approve" and v.ground_truth == "legit")
            or (v.verdict == "escalate" and v.ground_truth == "escalate")
        )

    def _verdict_reward(self, v: VerdictResult, severity: float = 0.5) -> float:
        if v.verdict == "reject" and v.ground_truth == "fraud":
            return 0.3 + 0.1 * severity
        elif v.verdict == "approve" and v.ground_truth == "legit":
            return 0.1
        elif v.verdict == "escalate" and v.ground_truth == "escalate":
            return 0.15
        elif v.verdict == "reject" and v.ground_truth == "legit":
            return -0.4
        elif v.verdict == "approve" and v.ground_truth == "fraud":
            return -0.5
        elif v.verdict == "escalate":
            return -0.05
        elif v.verdict == "approve" and v.ground_truth == "escalate":
            return -0.15
        elif v.verdict == "reject" and v.ground_truth == "escalate":
            return -0.1
        return -0.05

    def _normalize(self, raw: float, best: float, worst: float) -> float:
        score_range = best - worst
        if score_range <= 0:
            return 0.5
        normalized = (raw - worst) / score_range
        return max(0.0, min(1.0, normalized))


def grade_episode(record: EpisodeRecord) -> float:
    """Grade an episode using the appropriate task grader."""
    from . import GRADERS
    grader = GRADERS.get(record.task_id)
    if grader is None:
        raise ValueError(f"Unknown task_id: {record.task_id}")
    return grader.grade(record)
