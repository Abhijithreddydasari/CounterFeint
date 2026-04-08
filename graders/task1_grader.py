"""
Task 1 Grader: Basic Ad Triage (Easy).

Scores based on verdict accuracy only. No network or calibration bonuses.
5 ads, 25 budget — a decent LLM should score 0.6-0.8.
"""

from __future__ import annotations

from .base_grader import BaseGrader, EpisodeRecord


class Task1Grader(BaseGrader):
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

        best += 0.2

        if record.total_steps > 0:
            correct = self._count_correct_verdicts(record.verdicts)
            raw += (correct / record.total_steps) * 0.2

        return self._normalize(raw, best, worst)
