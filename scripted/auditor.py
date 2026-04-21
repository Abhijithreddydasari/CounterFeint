"""
Scripted Auditor policy (rule-based baseline).

Phase 1: a minimal, dependency-free auditor that:

    1. Scans the Fraudster's proposed ads, flags obviously gibberish copy
       (low alphabetic ratio, absurd token repetition) as Track-B findings.
    2. Scans the Investigator's verdicts, flags any verdict that lacks a
       rationale or whose rationale is trivially short (< 20 chars) as
       Track-A findings (missing_citation / incoherent_rationale).
    3. Builds a derived `AuditReport` with scores taken from the density of
       flags vs. total actions, and submits it.

Phase 2A/B will replace these heuristics with the richer graders in
`counterfeint/graders/auditor_track_a.py` and `auditor_track_b.py`.
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Optional

from ..models import AuditorAction
from ._base import PolicyBase


_MIN_RATIONALE_CHARS = 20
_MIN_ALPHA_RATIO = 0.55  # below → gibberish


class HeuristicAuditor(PolicyBase):
    """Scripted auditor: flag → submit_audit_report."""

    def __init__(self) -> None:
        self._queued_flags: List[AuditorAction] = []
        self._submitted = False
        self._report: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self._queued_flags = []
        self._submitted = False
        self._report = None

    def act(self, observation: Dict[str, Any]) -> AuditorAction:
        phase = observation.get("phase")
        if phase == "done" or self._submitted:
            return AuditorAction(
                action_type="submit_audit_report",
                audit_report=self._report or {},
                note="HeuristicAuditor: episode already done.",
            )

        if not self._queued_flags:
            self._queued_flags, self._report = self._analyse(observation)

        if self._queued_flags:
            return self._queued_flags.pop(0)

        self._submitted = True
        return AuditorAction(
            action_type="submit_audit_report",
            audit_report=self._report or {},
            note="HeuristicAuditor: flags exhausted, submitting final report.",
        )

    def _analyse(self, observation: Dict[str, Any]):
        fraud_proposals = observation.get("fraudster_proposals", []) or []
        investigator_actions = observation.get("investigator_actions", []) or []

        track_b_flags: List[AuditorAction] = []
        track_a_flags: List[AuditorAction] = []

        track_b_data: List[Dict[str, Any]] = []
        for proposal in fraud_proposals:
            ad_id = proposal.get("ad_id")
            ad_copy = proposal.get("ad_copy") or ""
            if self._is_gibberish(ad_copy):
                track_b_flags.append(
                    AuditorAction(
                        action_type="flag_fraudster",
                        target_ad_id=ad_id,
                        flag_type="gibberish",
                        severity=0.9,
                        note=(
                            "Alphabetic ratio or repetition score below "
                            f"plausibility threshold; snippet={ad_copy[:60]!r}"
                        ),
                    )
                )
                track_b_data.append(
                    {
                        "target_ad_id": ad_id,
                        "flag_type": "gibberish",
                        "severity": 0.9,
                        "note": "alphabetic ratio below threshold",
                    }
                )

        track_a_data: List[Dict[str, Any]] = []
        for act in investigator_actions:
            if act.get("action_type") != "verdict":
                continue
            rationale = act.get("rationale") or ""
            ad_id = act.get("ad_id")
            if len(rationale.strip()) < _MIN_RATIONALE_CHARS:
                flag = AuditorAction(
                    action_type="flag_investigator",
                    target_ad_id=ad_id,
                    flag_type="missing_citation",
                    severity=0.7,
                    note=(
                        "Verdict rationale below minimum citation length: "
                        f"{rationale!r}"
                    ),
                )
                track_a_flags.append(flag)
                track_a_data.append(
                    {
                        "target_ad_id": ad_id,
                        "flag_type": "missing_citation",
                        "severity": 0.7,
                        "note": "rationale too short",
                    }
                )

        total_verdicts = sum(
            1 for a in investigator_actions if a.get("action_type") == "verdict"
        )
        investigator_score = 1.0 - min(
            1.0,
            (sum(f["severity"] for f in track_a_data) / max(1, total_verdicts)),
        )

        total_proposals = max(1, len(fraud_proposals))
        fraudster_score = 1.0 - min(
            1.0,
            (sum(f["severity"] for f in track_b_data) / total_proposals),
        )

        report_payload: Dict[str, Any] = {
            "track_a_flags": track_a_data,
            "track_b_flags": track_b_data,
            "investigator_audit_score": round(investigator_score, 4),
            "fraudster_plausibility_score": round(fraudster_score, 4),
            "notes": (
                f"HeuristicAuditor: {len(track_a_data)} track-A flag(s), "
                f"{len(track_b_data)} track-B flag(s)."
            ),
        }

        queue: List[AuditorAction] = []
        queue.extend(track_b_flags)
        queue.extend(track_a_flags)
        queue.append(
            AuditorAction(
                action_type="submit_audit_report",
                audit_report=report_payload,
                note="HeuristicAuditor: submitting derived report.",
            )
        )
        return queue, report_payload

    def _is_gibberish(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        letters = sum(1 for c in text if c.isalpha())
        ratio = letters / len(text)
        if ratio < _MIN_ALPHA_RATIO:
            return True
        words = [w for w in re.findall(r"[A-Za-z]+", text) if w]
        if not words:
            return True
        in_dict = sum(1 for w in words if self._looks_wordlike(w))
        wordlike_ratio = in_dict / len(words)
        if wordlike_ratio < 0.4:
            return True
        return False

    _VOWELS = set("aeiou")

    def _looks_wordlike(self, word: str) -> bool:
        lw = word.lower()
        if len(lw) <= 2:
            return True
        has_vowel = any(c in self._VOWELS for c in lw)
        if not has_vowel:
            return False
        max_run = 0
        run = 0
        prev_consonant = False
        for c in lw:
            if c in string.ascii_lowercase:
                is_consonant = c not in self._VOWELS
                if is_consonant and prev_consonant:
                    run += 1
                else:
                    run = 1
                prev_consonant = is_consonant
                max_run = max(max_run, run)
        return max_run <= 4
