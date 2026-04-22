"""
Scripted Auditor policy backed by the real Track A / Track B graders.

The Auditor consumes the full audit-phase observation, runs the
rule-based audit checks in `graders/auditor_track_a.py` and
`graders/auditor_track_b.py`, queues up one `flag_investigator` /
`flag_fraudster` action per flag, and finally submits an audit report
containing the flag payloads and aggregated audit scores.

This means the scripted Auditor is no longer a toy — it is the exact
same decision surface the LLM Auditor will fill in during Phase 3.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..graders.auditor_track_a import (
    bias_audit,
    calibration_audit,
    cross_ad_consistency_audit,
    investigator_audit_score,
    rationale_citation_audit,
    rationale_verdict_coherence_audit,
)
from ..graders.base_grader import (
    EpisodeRecord,
    LinkResult,
    VerdictResult,
)
from ..graders.plausibility_score import compute_queue_plausibility
from ..models import AuditFlag, AuditorAction
from ._base import PolicyBase


class HeuristicAuditor(PolicyBase):
    """Scripted Auditor — runs rule-based graders and submits a final report."""

    def __init__(self) -> None:
        self._queued: List[AuditorAction] = []
        self._submitted: bool = False
        self._report: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        self._queued = []
        self._submitted = False
        self._report = None

    def act(self, observation: Dict[str, Any]) -> AuditorAction:
        if observation.get("phase") == "done" or self._submitted:
            return AuditorAction(
                action_type="submit_audit_report",
                audit_report=self._report or {},
                note="HeuristicAuditor: episode already done.",
            )

        if not self._queued:
            self._queued, self._report = self._plan(observation)

        if self._queued:
            action = self._queued.pop(0)
            if action.action_type == "submit_audit_report":
                self._submitted = True
            return action

        self._submitted = True
        return AuditorAction(
            action_type="submit_audit_report",
            audit_report=self._report or {},
            note="HeuristicAuditor: queue empty, submitting final report.",
        )

    # ------------------------------------------------------------------
    # Internal planning
    # ------------------------------------------------------------------

    def _plan(
        self, observation: Dict[str, Any]
    ) -> tuple[List[AuditorAction], Dict[str, Any]]:
        record = self._build_record(observation)
        investigator_actions = observation.get("investigator_actions", []) or []
        fraudster_proposals = observation.get("fraudster_proposals", []) or []
        investigation_data_seen = observation.get("investigation_data_seen", {}) or {}

        # Track A — audit the Investigator.
        track_a_flags: List[AuditFlag] = []
        if record is not None:
            track_a_flags.extend(calibration_audit(record))
            track_a_flags.extend(cross_ad_consistency_audit(record))
            track_a_flags.extend(bias_audit(record))
        track_a_flags.extend(
            rationale_citation_audit(investigator_actions, investigation_data_seen)
        )
        track_a_flags.extend(
            rationale_verdict_coherence_audit(investigator_actions)
        )

        # Track B — audit the Fraudster.
        per_ad_scores, track_b_flags, queue_plaus = compute_queue_plausibility(
            fraudster_proposals
        )

        # Convert flags to Auditor actions (one flag per action).
        actions: List[AuditorAction] = []
        for f in track_b_flags:
            actions.append(
                AuditorAction(
                    action_type="flag_fraudster",
                    target_ad_id=f.target_ad_id,
                    flag_type=f.flag_type,
                    severity=f.severity,
                    note=f.note,
                )
            )
        for f in track_a_flags:
            actions.append(
                AuditorAction(
                    action_type="flag_investigator",
                    target_ad_id=f.target_ad_id,
                    flag_type=f.flag_type,
                    severity=f.severity,
                    note=f.note,
                )
            )

        # Aggregate scores.
        inv_score = investigator_audit_score(track_a_flags)

        report: Dict[str, Any] = {
            "track_a_flags": [f.model_dump() for f in track_a_flags],
            "track_b_flags": [f.model_dump() for f in track_b_flags],
            "investigator_audit_score": round(inv_score, 4),
            "fraudster_plausibility_score": round(queue_plaus, 4),
            "per_ad_plausibility": {k: round(v, 4) for k, v in per_ad_scores.items()},
            "notes": (
                f"HeuristicAuditor: {len(track_a_flags)} Track A flag(s), "
                f"{len(track_b_flags)} Track B flag(s); "
                f"inv_audit={inv_score:.2f}, fraud_plaus={queue_plaus:.2f}."
            ),
        }

        actions.append(
            AuditorAction(
                action_type="submit_audit_report",
                audit_report=report,
                note="HeuristicAuditor: submitting derived report.",
            )
        )
        return actions, report

    # ------------------------------------------------------------------
    # EpisodeRecord reconstruction from the auditor observation
    # ------------------------------------------------------------------

    def _build_record(self, observation: Dict[str, Any]) -> Optional[EpisodeRecord]:
        record_payload = observation.get("full_episode_record") or {}
        if not record_payload:
            return None

        verdict_entries = record_payload.get("verdicts") or []
        link_entries = record_payload.get("links") or []
        ad_entries = record_payload.get("ads") or []

        verdicts: List[VerdictResult] = []
        for v in verdict_entries:
            if "verdict" not in v:
                continue
            verdicts.append(
                VerdictResult(
                    ad_id=v.get("ad_id", ""),
                    verdict=v.get("verdict", "approve"),
                    confidence=float(v.get("confidence", 0.5) or 0.5),
                    ground_truth=v.get("ground_truth", "legit"),
                    auto_approved=bool(v.get("auto_approved", False)),
                )
            )

        links: List[LinkResult] = [
            LinkResult(
                ad_id_1=l.get("ad_id_1", ""),
                ad_id_2=l.get("ad_id_2", ""),
                correct=bool(l.get("correct", False)),
            )
            for l in link_entries
            if l.get("ad_id_1") and l.get("ad_id_2")
        ]

        ads_metadata: List[Dict[str, Any]] = []
        for ad in ad_entries:
            if not ad.get("ad_id"):
                continue
            ads_metadata.append(
                {
                    "ad_id": ad.get("ad_id"),
                    "ground_truth": ad.get("ground_truth", "legit"),
                    "severity": float(ad.get("severity", 0.5) or 0.5),
                    "fraud_type": ad.get("fraud_type", ""),
                    "category": ad.get("category", ""),
                    "country": ad.get("country", ""),
                }
            )

        return EpisodeRecord(
            task_id=record_payload.get("task_id", ""),
            total_steps=int(record_payload.get("total_steps", 0) or 0),
            action_budget=int(record_payload.get("action_budget", 0) or 0),
            verdicts=verdicts,
            links=links,
            ads_metadata=ads_metadata,
        )


__all__ = ["HeuristicAuditor"]
