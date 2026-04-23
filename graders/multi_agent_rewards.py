"""
Multi-agent reward integration for CounterFeint R2.

Combines:
  • the R1 single-agent grader score (investigator base reward)
  • Track A audit flags (investigator reasoning quality)
  • Track B plausibility score (fraudster surface realism)
  • Auditor "ground truth" accounting (did Auditor flag *real* issues?)

and produces a dict `{"fraudster", "investigator", "auditor"}` of episode-
end rewards that the Referee stashes on `RefereeState`.

The formulas implement the weight schema from the plan §2C while staying
deterministic (no LLM judge) and computable from the `EpisodeRecord` +
`AuditReport` + Fraudster proposal log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..models import AuditFlag, AuditReport
from .auditor_track_a import (
    bias_audit,
    calibration_audit,
    cross_ad_consistency_audit,
    investigator_audit_score,
    rationale_citation_audit,
    rationale_verdict_coherence_audit,
)
from .auditor_track_b import pattern_novelty_check
from .base_grader import EpisodeRecord, VerdictResult, grade_episode
from .plausibility_score import (
    compute_plausibility_score,
    compute_queue_plausibility,
)

# -----------------------------------------------------------------------------
# Reward-formula coefficients (mirrors plan §2C).  Kept in one place so Phase
# 3 training can sweep them.
# -----------------------------------------------------------------------------

FRAUDSTER_PER_AD_SEVERITY_WEIGHT = 1.0   # per surviving ad, scaled by plausibility
FRAUDSTER_BANNED_PENALTY = 1.0           # per ad Investigator rejected
# Deprecated as of the pre-training cleanup — the Track B flag count double-
# counts with the plausibility gate (a gibberish ad both drags its plausibility
# to ~0 *and* emits a Track B flag, getting subtracted twice).  Kept at 0.0 for
# backwards compatibility with anything still importing the symbol; see
# ROUND_2_Q5_REALISM_REWARDS_TRAINING.md §3.1 for the rationale.
FRAUDSTER_UNREALISTIC_PENALTY = 0.0

INVESTIGATOR_RATIONALE_BONUS = 0.2       # per inferred-approved rationale
INVESTIGATOR_INCONSISTENCY_PENALTY = 0.3 # per flagged inconsistency

AUDITOR_TRUE_MISCAL = 1.0
AUDITOR_TRUE_UNREALISTIC = 1.0
AUDITOR_TRUE_INCONSISTENCY = 0.5
AUDITOR_FALSE_POSITIVE_PENALTY = 0.5


# -----------------------------------------------------------------------------
# Shared cache so the O(N²) Track B plausibility / novelty pipeline only runs
# once per episode instead of 3× (fraudster_reward + auditor_ground_truth +
# compute_episode_rewards each previously invoked compute_queue_plausibility,
# which in turn re-ran pattern_novelty_check *per ad*).  Referee builds the
# cache once at audit→done; downstream reward functions read from it.
# -----------------------------------------------------------------------------


@dataclass
class RewardCache:
    """Cached artefacts from a single compute_queue_plausibility pass."""

    per_ad_plausibility: Dict[str, float] = field(default_factory=dict)
    track_b_flags: List[AuditFlag] = field(default_factory=list)
    queue_plausibility: float = 1.0
    # Single queue-level novelty_check result; passed into every per-ad
    # compute_plausibility_score call so the O(N²) loop runs exactly once.
    novelty_cache: Tuple[float, List[AuditFlag]] = field(
        default_factory=lambda: (1.0, [])
    )


def build_reward_cache(
    fraudster_proposal_log: Iterable[Mapping[str, Any]],
    *,
    country_by_ad_id: Optional[Mapping[str, str]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> RewardCache:
    """Compute queue plausibility + novelty once and bundle the results."""
    queue = [
        p
        for p in fraudster_proposal_log
        if p.get("action_type") not in ("skip",)
    ]
    novelty = pattern_novelty_check(queue) if queue else (1.0, [])
    per_ad, track_b, queue_plaus = compute_queue_plausibility(
        queue,
        country_by_ad_id=country_by_ad_id,
        weights=weights,
        novelty_cache=novelty,
    )
    return RewardCache(
        per_ad_plausibility=per_ad,
        track_b_flags=track_b,
        queue_plausibility=queue_plaus,
        novelty_cache=novelty,
    )


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------


@dataclass
class RewardInputs:
    """All data the reward computer needs.  Referee builds this once at audit→done.

    ``cache`` is an optional ``RewardCache`` shared between all reward
    computations in a single episode.  If ``None`` (e.g. unit-test callers
    that invoke a single reward function), the reward functions build a
    per-call cache on demand.
    """

    record: EpisodeRecord
    audit_report: AuditReport
    fraudster_proposal_log: List[Dict[str, Any]]
    investigator_action_log: List[Dict[str, Any]]
    investigation_data_seen: Dict[str, Dict[str, str]]
    fraudster_ad_ids: List[str]
    cache: Optional[RewardCache] = None

    def get_or_build_cache(self) -> RewardCache:
        """Return the shared cache, building one on-demand if absent."""
        if self.cache is None:
            self.cache = build_reward_cache(self.fraudster_proposal_log)
        return self.cache


# -----------------------------------------------------------------------------
# Ground-truth accounting for auditor flags
# -----------------------------------------------------------------------------


def compute_auditor_ground_truth(inputs: RewardInputs) -> Dict[str, int]:
    """
    Re-run the Track A audit over the same EpisodeRecord to get the set of
    flags the Auditor *should* have raised.  We then compare against what
    the Auditor actually submitted and count:

      • `true_miscalibrations_flagged`  - matching flag_type="miscalibration"
      • `true_unrealistic_flagged`      - matching Track B "gibberish" /
                                          "template_repetition" etc.
      • `true_inconsistencies_flagged`  - matching flag_type="inconsistency"
      • `true_citation_flagged`         - matching flag_type="missing_citation"
      • `true_incoherent_flagged`       - matching flag_type="incoherent_rationale"
      • `true_bias_flagged`             - matching flag_type="bias"
      • `false_positives`               - flags Auditor raised that ground
                                          truth did not.

    We match by `flag_type` + `target_ad_id` so two flags of the same type
    on different ads can't count as one.
    """
    ground_truth_a = []
    ground_truth_a.extend(calibration_audit(inputs.record))
    ground_truth_a.extend(
        rationale_citation_audit(
            inputs.investigator_action_log, inputs.investigation_data_seen
        )
    )
    ground_truth_a.extend(
        rationale_verdict_coherence_audit(inputs.investigator_action_log)
    )
    ground_truth_a.extend(cross_ad_consistency_audit(inputs.record))
    ground_truth_a.extend(bias_audit(inputs.record))

    cache = inputs.get_or_build_cache()
    ground_truth_b_flags = cache.track_b_flags

    gt_set = {(f.flag_type, f.target_ad_id) for f in ground_truth_a + ground_truth_b_flags}

    # Tally Auditor's submitted flags.
    submitted = list(inputs.audit_report.track_a_flags) + list(
        inputs.audit_report.track_b_flags
    )
    counts = {
        "true_miscalibrations_flagged": 0,
        "true_unrealistic_flagged": 0,
        "true_inconsistencies_flagged": 0,
        "true_citation_flagged": 0,
        "true_incoherent_flagged": 0,
        "true_bias_flagged": 0,
        "false_positives": 0,
    }

    for flag in submitted:
        key = (flag.flag_type, flag.target_ad_id)
        is_true_positive = key in gt_set
        if is_true_positive:
            ft = flag.flag_type
            if ft == "miscalibration":
                counts["true_miscalibrations_flagged"] += 1
            elif ft in {"gibberish", "parameter_mismatch", "template_repetition", "branding_anomaly", "market_implausible"}:
                counts["true_unrealistic_flagged"] += 1
            elif ft == "inconsistency":
                counts["true_inconsistencies_flagged"] += 1
            elif ft == "missing_citation":
                counts["true_citation_flagged"] += 1
            elif ft == "incoherent_rationale":
                counts["true_incoherent_flagged"] += 1
            elif ft == "bias":
                counts["true_bias_flagged"] += 1
        else:
            counts["false_positives"] += 1
    return counts


# -----------------------------------------------------------------------------
# Per-role reward formulas
# -----------------------------------------------------------------------------


def fraudster_reward(inputs: RewardInputs) -> float:
    """
    Fraudster reward is a *productivity* metric: how many unbanned fraud ads
    slipped past the Investigator, weighted by severity and plausibility.

    reward = Σ severity_i * plausibility_i  (for each unbanned fraudster ad)
           - FRAUDSTER_BANNED_PENALTY * n_banned

    Note: the old ``FRAUDSTER_UNREALISTIC_PENALTY * n_track_b_flags`` term was
    removed in the pre-training cleanup — Track B flag counts double-counted
    with the plausibility gate.  Plausibility is now the single realism
    chokepoint (gibberish ads collapse plaus_i → 0 so the first term already
    zeroes-out survival credit).
    """
    severity_by_ad: Dict[str, float] = {}
    for meta in inputs.record.ads_metadata:
        severity_by_ad[meta.get("ad_id", "")] = float(meta.get("severity", 0.5) or 0.5)

    cache = inputs.get_or_build_cache()
    per_ad_plaus = cache.per_ad_plausibility

    verdict_by_ad = {v.ad_id: v.verdict for v in inputs.record.verdicts}

    fraudster_ads = set(inputs.fraudster_ad_ids)
    unbanned, banned = [], []
    for ad_id in fraudster_ads:
        verdict = verdict_by_ad.get(ad_id)
        if verdict == "reject":
            banned.append(ad_id)
        else:
            # Anything not explicitly rejected survives — approved, escalated,
            # or left un-reviewed (end-of-episode default).
            unbanned.append(ad_id)

    reward = 0.0
    for ad_id in unbanned:
        plaus = per_ad_plaus.get(ad_id, 0.0)
        sev = severity_by_ad.get(ad_id, 0.6)
        reward += FRAUDSTER_PER_AD_SEVERITY_WEIGHT * sev * plaus

    reward -= FRAUDSTER_BANNED_PENALTY * len(banned)

    return reward


def investigator_reward(inputs: RewardInputs) -> float:
    """
    Investigator reward = R1 grader_score + Track A audit signal.

    R1 grader_score ∈ [0, 1] is the base (calibration, triage, correctness).
    We add bonuses for coherent/citing rationales and penalties for
    inconsistencies found by the Auditor.
    """
    base = grade_episode(inputs.record)

    # "Approved rationales" == verdicts for which *no* Track A flag fired.
    n_verdicts = sum(
        1 for a in inputs.investigator_action_log if a.get("action_type") == "verdict"
    )
    flagged_ids = {
        f.target_ad_id
        for f in inputs.audit_report.track_a_flags
        if f.target_ad_id
    }
    n_approved = max(0, n_verdicts - len(flagged_ids))
    n_inconsistencies = sum(
        1 for f in inputs.audit_report.track_a_flags if f.flag_type == "inconsistency"
    )

    reward = (
        base
        + INVESTIGATOR_RATIONALE_BONUS * n_approved
        - INVESTIGATOR_INCONSISTENCY_PENALTY * n_inconsistencies
    )
    return reward


def auditor_reward(
    inputs: RewardInputs,
    *,
    ground_truth_counts: Optional[Mapping[str, int]] = None,
) -> float:
    """
    Auditor reward = credit for true positives - penalty for false positives.

    `ground_truth_counts` is the dict returned by `compute_auditor_ground_truth`.
    Passing it in avoids recomputation when the caller already has it.
    """
    counts = dict(ground_truth_counts or compute_auditor_ground_truth(inputs))
    reward = (
        AUDITOR_TRUE_MISCAL * counts.get("true_miscalibrations_flagged", 0)
        + AUDITOR_TRUE_UNREALISTIC * counts.get("true_unrealistic_flagged", 0)
        + AUDITOR_TRUE_INCONSISTENCY * counts.get("true_inconsistencies_flagged", 0)
        + 0.5 * counts.get("true_citation_flagged", 0)
        + 0.5 * counts.get("true_incoherent_flagged", 0)
        + 0.5 * counts.get("true_bias_flagged", 0)
        - AUDITOR_FALSE_POSITIVE_PENALTY * counts.get("false_positives", 0)
    )
    return reward


# -----------------------------------------------------------------------------
# Top-level entry points
# -----------------------------------------------------------------------------


def compute_episode_rewards(inputs: RewardInputs) -> Dict[str, float]:
    """
    Compute all three role-scoped rewards plus the per-ad plausibility scores.

    Returns a dict:
      {
        "fraudster":  float,
        "investigator":  float,
        "auditor":  float,
        "grader_score":  float,           # R1 base grader score
        "per_ad_plausibility":  dict,     # ad_id -> plausibility
        "audit_ground_truth":  dict,      # true_*/false_* counters
      }

    Builds a single ``RewardCache`` shared across all reward computations, so
    ``compute_queue_plausibility`` and ``pattern_novelty_check`` each run
    exactly once per episode (the pre-cleanup version ran them 3× + O(N) per
    caller).
    """
    # Prime the shared cache once, on the Inputs object.
    inputs.get_or_build_cache()

    gt = compute_auditor_ground_truth(inputs)
    return {
        "fraudster": fraudster_reward(inputs),
        "investigator": investigator_reward(inputs),
        "auditor": auditor_reward(inputs, ground_truth_counts=gt),
        "grader_score": grade_episode(inputs.record),
        "per_ad_plausibility": inputs.cache.per_ad_plausibility if inputs.cache else {},
        "audit_ground_truth": gt,
    }


__all__ = [
    "AUDITOR_FALSE_POSITIVE_PENALTY",
    "AUDITOR_TRUE_INCONSISTENCY",
    "AUDITOR_TRUE_MISCAL",
    "AUDITOR_TRUE_UNREALISTIC",
    "FRAUDSTER_BANNED_PENALTY",
    "FRAUDSTER_PER_AD_SEVERITY_WEIGHT",
    "FRAUDSTER_UNREALISTIC_PENALTY",
    "INVESTIGATOR_INCONSISTENCY_PENALTY",
    "INVESTIGATOR_RATIONALE_BONUS",
    "RewardCache",
    "RewardInputs",
    "auditor_reward",
    "build_reward_cache",
    "compute_auditor_ground_truth",
    "compute_episode_rewards",
    "fraudster_reward",
    "investigator_reward",
]
