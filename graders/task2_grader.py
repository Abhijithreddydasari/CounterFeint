"""
Task 2 Grader: Sophisticated Fraud Under Budget Pressure (Medium).

Adds budget efficiency bonus and calibration scoring on top of verdict accuracy.
12 ads, 30 budget — requires triage, expected baseline 0.3-0.5.
"""

from __future__ import annotations

from .base_grader import BaseGrader, EpisodeRecord


class Task2Grader(BaseGrader):
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
                worst -= 0.4
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

        # Calibration bonus: reward agents whose stated confidence
        # correlates with actual accuracy
        calibration = self._compute_calibration(record)
        raw += calibration * 0.15
        best += 0.15

        return self._normalize(raw, best, worst)

    def _compute_calibration(self, record: EpisodeRecord) -> float:
        """
        Measure how well confidence tracks correctness.

        Bins verdicts by confidence and checks if the fraction correct
        in each bin roughly matches the stated confidence.
        """
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
