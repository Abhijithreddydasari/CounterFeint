"""
Single-entry-point orchestrator for CounterFeint's full Auditor pipeline.

The Auditor surface is split across two grader modules so each track is unit-
testable in isolation:

* ``auditor_track_a`` — five rule-based audits over the Investigator's
  decisions (calibration, cross-ad consistency, bias, rationale citation,
  rationale ↔ verdict coherence).
* ``auditor_track_b`` — five plausibility audits over the Fraudster's ad
  surface (intrinsic, grounding, real-world, signal-realism, novelty),
  combined into a per-ad plausibility score in ``plausibility_score``.

Outside of unit tests, every consumer (the scripted ``HeuristicAuditor``,
the Referee's ``_compute_default_track_scores``, and
``multi_agent_rewards.compute_episode_rewards``) needs *both* tracks plus
the aggregate scores.  Without a single orchestrator each consumer ended up
re-implementing the same dispatch sequence by hand and they drifted out of
sync over Round 2.

``run_full_audit`` is that single orchestrator.  It runs every audit once,
populates a shared ``RewardCache`` so the downstream reward functions don't
re-do the O(N²) plausibility/novelty work, and returns a typed
``FullAuditResult`` bundle.

This file is a *pure refactor* — every audit it calls already exists.  The
only behavioural difference compared to the Round-2 hand-rolled call sites
is that the per-ad plausibility / novelty pipeline is computed exactly
once instead of once per consumer (the reward-cache pattern, lifted from
``multi_agent_rewards.build_reward_cache``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..models import AuditFlag
from .auditor_track_a import (
    bias_audit,
    calibration_audit,
    cross_ad_consistency_audit,
    investigator_audit_score,
    rationale_citation_audit,
    rationale_verdict_coherence_audit,
)
from .base_grader import EpisodeRecord
from .multi_agent_rewards import RewardCache, build_reward_cache


@dataclass
class FullAuditResult:
    """Bundle returned by :func:`run_full_audit`.

    Attributes
    ----------
    track_a_flags
        All Track A flags (Investigator-side audits) in a stable order:
        calibration → cross-ad consistency → bias → rationale citation →
        rationale↔verdict coherence.  Empty if ``record`` is ``None``
        (the Track A audits all need an EpisodeRecord).
    track_b_flags
        All Track B flags (Fraudster-side plausibility audits), produced
        by ``compute_queue_plausibility`` over the Fraudster proposal log.
    investigator_audit_score
        Aggregate Track A score in [0, 1] from
        ``auditor_track_a.investigator_audit_score``.
    fraudster_plausibility_score
        Aggregate Track B score in [0, 1] from
        ``plausibility_score.compute_queue_plausibility``.
    per_ad_plausibility
        ad_id → plausibility ∈ [0, 1].  Used by ``investigator_reward``
        as a difficulty modulator and by ``fraudster_reward`` as the
        survival-credit gate.
    reward_cache
        The shared ``RewardCache`` so downstream reward functions
        (``multi_agent_rewards.compute_episode_rewards``) can be
        threaded through without re-computing plausibility / novelty.
    """

    track_a_flags: List[AuditFlag]
    track_b_flags: List[AuditFlag]
    investigator_audit_score: float
    fraudster_plausibility_score: float
    per_ad_plausibility: Dict[str, float]
    reward_cache: RewardCache


def run_full_audit(
    *,
    record: Optional[EpisodeRecord],
    investigator_action_log: Sequence[Mapping[str, Any]],
    investigation_data_seen: Mapping[str, Mapping[str, str]],
    fraudster_proposal_log: Sequence[Mapping[str, Any]],
    country_by_ad_id: Optional[Mapping[str, str]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> FullAuditResult:
    """Run all Track A + Track B audits in one pass.

    See module docstring for context.  All inputs are the same shapes the
    Referee already exposes via the audit-phase observation, so callers
    don't need to massage anything.

    Parameters
    ----------
    record
        ``EpisodeRecord`` reconstructed from the audit observation.  May
        be ``None`` if the episode terminated before any verdicts were
        recorded (the Track A audits are skipped in that case; Track B
        still runs over whatever the Fraudster proposed).
    investigator_action_log
        Investigator action log (each entry is a dict produced by the
        Referee, with at minimum ``action_type`` / ``ad_id`` /
        ``rationale`` / ``investigation_target_used`` fields).
    investigation_data_seen
        ad_id → {target → findings}.  Used by ``rationale_citation_audit``
        to verify the Investigator cited findings it actually pulled.
    fraudster_proposal_log
        Fraudster proposal log (one entry per ``propose_ad`` /
        ``modify_pending_ad`` / ``commit_final``).  Used to drive the
        Track B plausibility computation.
    country_by_ad_id, weights
        Forwarded to :func:`build_reward_cache` →
        :func:`compute_queue_plausibility` for the Track B pass.

    Returns
    -------
    FullAuditResult
        Typed bundle with both flag lists, both aggregate scores, the
        per-ad plausibility map, and the reward cache for downstream
        reuse.
    """
    track_a: List[AuditFlag] = []
    if record is not None:
        track_a.extend(calibration_audit(record))
        track_a.extend(cross_ad_consistency_audit(record))
        track_a.extend(bias_audit(record))
    track_a.extend(
        rationale_citation_audit(investigator_action_log, investigation_data_seen)
    )
    track_a.extend(rationale_verdict_coherence_audit(investigator_action_log))

    cache = build_reward_cache(
        fraudster_proposal_log,
        country_by_ad_id=country_by_ad_id,
        weights=weights,
    )

    return FullAuditResult(
        track_a_flags=track_a,
        track_b_flags=list(cache.track_b_flags),
        investigator_audit_score=investigator_audit_score(track_a),
        fraudster_plausibility_score=cache.queue_plausibility,
        per_ad_plausibility=dict(cache.per_ad_plausibility),
        reward_cache=cache,
    )


__all__ = ["FullAuditResult", "run_full_audit"]
