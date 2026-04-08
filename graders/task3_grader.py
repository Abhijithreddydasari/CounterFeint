"""
Task 3 Grader: Coordinated Fraud Network Detection (Hard).

Adds graph-based network detection scoring on top of verdict accuracy,
budget efficiency, and calibration. 20 ads, 35 budget, 3 fraud rings
with varied topologies. Expected baseline 0.1-0.3.

Network scoring uses ground truth edge coverage: what fraction of
ground-truth ring connections did the agent discover via link_accounts?
"""

from __future__ import annotations

from .base_grader import BaseGrader, EpisodeRecord


class Task3Grader(BaseGrader):
    def grade(self, record: EpisodeRecord) -> float:
        raw = 0.0
        best = 0.0
        worst = 0.0

        severity_map = {
            m["ad_id"]: m.get("severity", 0.5) for m in record.ads_metadata
        }

        for v in record.verdicts:
            severity = severity_map.get(v.ad_id, 0.5)
            raw += self._verdict_reward(v, severity)

            if v.ground_truth == "fraud":
                best += 0.3 + 0.1 * severity
                worst -= 0.5
            elif v.ground_truth == "legit":
                best += 0.1
                worst -= 0.35
            elif v.ground_truth == "escalate":
                best += 0.15
                worst -= 0.15

        n_investigations = max(0, record.total_steps - len([
            v for v in record.verdicts if not v.auto_approved
        ]))
        raw -= n_investigations * 0.02
        worst -= record.action_budget * 0.02

        # Budget efficiency bonus
        best += 0.2
        if record.total_steps > 0:
            correct = self._count_correct_verdicts(record.verdicts)
            raw += (correct / record.total_steps) * 0.2

        # Calibration bonus
        calibration = self._compute_calibration(record)
        raw += calibration * 0.15
        best += 0.15

        # Graph-based network detection scoring
        network_reward = self._compute_network_score(record)
        raw += network_reward
        # Best case: discover all ground truth edges
        total_gt_edges = self._count_ground_truth_edges(record)
        best += max(total_gt_edges, 1) * 0.4
        worst -= max(len(record.links), 1) * 0.2

        # Investigation coverage bonus: reward breadth over depth
        coverage = self._compute_coverage_bonus(record)
        raw += coverage * 0.1
        best += 0.1

        return self._normalize(raw, best, worst)

    def _compute_calibration(self, record: EpisodeRecord) -> float:
        manual = [v for v in record.verdicts if not v.auto_approved]
        if len(manual) < 3:
            return 0.5

        bins = {"low": [], "mid": [], "high": []}
        for v in manual:
            if v.confidence < 0.4:
                bins["low"].append(v)
            elif v.confidence < 0.7:
                bins["mid"].append(v)
            else:
                bins["high"].append(v)

        errors = []
        for _label, group in bins.items():
            if not group:
                continue
            avg_conf = sum(v.confidence for v in group) / len(group)
            avg_acc = sum(1 for v in group if self._is_correct(v)) / len(group)
            errors.append(abs(avg_conf - avg_acc))

        if not errors:
            return 0.5

        avg_error = sum(errors) / len(errors)
        return max(0.0, 1.0 - 2.0 * avg_error)

    def _count_ground_truth_edges(self, record: EpisodeRecord) -> int:
        """Count the total number of ground truth edges across all rings."""
        total = 0
        for ring_size in record.ring_sizes:
            total += ring_size * (ring_size - 1) // 2
        return total

    def _compute_network_score(self, record: EpisodeRecord) -> float:
        """Score link_accounts actions against ground truth fraud rings.

        Uses edge-coverage: correct links earn +0.4 each, incorrect
        links incur -0.25 each (heavier penalty since decoys exist).
        """
        correct_links = sum(1 for l in record.links if l.correct)
        incorrect_links = sum(1 for l in record.links if not l.correct)

        reward = correct_links * 0.4
        reward -= incorrect_links * 0.25

        return reward

    def _compute_coverage_bonus(self, record: EpisodeRecord) -> float:
        """Reward agents that investigate across multiple ads (breadth).

        Agents that only deep-dive a single ad miss network signals.
        """
        if not record.ads_metadata:
            return 0.0

        total_ads = len(record.ads_metadata)
        reviewed = sum(1 for v in record.verdicts if not v.auto_approved)
        coverage_ratio = reviewed / total_ads if total_ads > 0 else 0.0
        return min(1.0, coverage_ratio)
