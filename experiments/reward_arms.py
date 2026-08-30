"""Shared reward computation under verifier ablation arms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set

from counterfeint.graders.auditor_pipeline import FullAuditResult, run_full_audit
from counterfeint.graders.base_grader import grade_episode
from counterfeint.graders.multi_agent_rewards import (
    FRAUDSTER_BANNED_PENALTY,
    FRAUDSTER_PER_AD_SEVERITY_WEIGHT,
    INVESTIGATOR_INCONSISTENCY_CAP,
    INVESTIGATOR_INCONSISTENCY_PENALTY,
    INVESTIGATOR_RATIONALE_BONUS,
    INVESTIGATOR_RATIONALE_FLAG_TYPES,
    build_reward_cache,
)
from counterfeint.models import AuditFlag

from .episode_bundle import EpisodeBundle, bundle_to_episode_record
from .llm_auditor import merge_hybrid

TRACK_A_TYPES = {
    "miscalibration",
    "missing_citation",
    "incoherent_rationale",
    "inconsistency",
    "bias",
}
TRACK_B_TYPES = {
    "gibberish",
    "parameter_mismatch",
    "template_repetition",
    "branding_anomaly",
    "market_implausible",
}


class AblationArm(str, Enum):
    FULL = "full"
    NO_TRACK_A = "no_track_a"
    NO_TRACK_B = "no_track_b"
    NO_PLAUSIBILITY_GATE = "no_plausibility_gate"
    OUTCOME_ONLY = "outcome_only"
    NO_MISCALIBRATION = "no_miscalibration"
    NO_MISSING_CITATION = "no_missing_citation"
    NO_INCOHERENT = "no_incoherent_rationale"
    NO_INCONSISTENCY = "no_inconsistency"
    NO_BIAS = "no_bias"
    NO_GIBBERISH = "no_gibberish"
    NO_PARAMETER_MISMATCH = "no_parameter_mismatch"
    NO_TEMPLATE_REPETITION = "no_template_repetition"
    NO_BRANDING_ANOMALY = "no_branding_anomaly"
    LLM_ONLY = "llm_only"
    HYBRID = "hybrid"


DROP_TYPE: Dict[AblationArm, Optional[str]] = {
    AblationArm.NO_MISCALIBRATION: "miscalibration",
    AblationArm.NO_MISSING_CITATION: "missing_citation",
    AblationArm.NO_INCOHERENT: "incoherent_rationale",
    AblationArm.NO_INCONSISTENCY: "inconsistency",
    AblationArm.NO_BIAS: "bias",
    AblationArm.NO_GIBBERISH: "gibberish",
    AblationArm.NO_PARAMETER_MISMATCH: "parameter_mismatch",
    AblationArm.NO_TEMPLATE_REPETITION: "template_repetition",
    AblationArm.NO_BRANDING_ANOMALY: "branding_anomaly",
}

# Zeroing a Track B dimension weight disables that check in plausibility.
ARM_PLAUS_WEIGHTS: Dict[AblationArm, Dict[str, float]] = {
    AblationArm.NO_GIBBERISH: {"real_world_existence": 0.0},
    AblationArm.NO_PARAMETER_MISMATCH: {
        "intrinsic_consistency": 0.0,
        "parameter_grounding": 0.0,
    },
    AblationArm.NO_TEMPLATE_REPETITION: {"pattern_novelty": 0.0},
    AblationArm.NO_BRANDING_ANOMALY: {"signal_realism": 0.0},
}


@dataclass
class RoleRewards:
    arm: str
    fraudster_reward: float
    investigator_reward: float
    grader_score: float
    mean_plausibility: float
    n_track_a: int
    n_track_b: int


def run_audit(bundle: EpisodeBundle) -> FullAuditResult:
    record = bundle_to_episode_record(bundle)
    return run_full_audit(
        record=record,
        investigator_action_log=bundle.investigator_actions,
        investigation_data_seen=bundle.investigation_data_seen,
        fraudster_proposal_log=bundle.fraudster_proposals,
    )


def _filter_flags(
    flags: List[AuditFlag],
    arm: AblationArm,
    extra_flags: Optional[List[AuditFlag]] = None,
) -> List[AuditFlag]:
    out = list(flags)
    if extra_flags:
        seen = {(f.flag_type, f.target_ad_id) for f in out}
        for f in extra_flags:
            key = (f.flag_type, f.target_ad_id)
            if key not in seen:
                out.append(f)
                seen.add(key)
    if arm == AblationArm.NO_TRACK_A:
        out = [f for f in out if f.track != "A"]
    if arm in (AblationArm.NO_TRACK_B, AblationArm.NO_PLAUSIBILITY_GATE):
        out = [f for f in out if f.track != "B"]
    drop = DROP_TYPE.get(arm)
    if drop:
        out = [f for f in out if f.flag_type != drop]
    if arm == AblationArm.LLM_ONLY:
        # Caller should pass only LLM flags via extra_flags and empty det flags.
        pass
    return out


def compute_role_rewards(
    bundle: EpisodeBundle,
    arm: AblationArm,
    *,
    llm_flags: Optional[List[AuditFlag]] = None,
    audit: Optional[FullAuditResult] = None,
) -> RoleRewards:
    record = bundle_to_episode_record(bundle)
    audit = audit or run_audit(bundle)
    grader_score = grade_episode(record)

    if arm == AblationArm.OUTCOME_ONLY:
        return RoleRewards(
            arm=arm.value,
            fraudster_reward=0.0,
            investigator_reward=grader_score,
            grader_score=grader_score,
            mean_plausibility=1.0,
            n_track_a=0,
            n_track_b=0,
        )

    if arm == AblationArm.LLM_ONLY:
        track_a = [f for f in (llm_flags or []) if f.track == "A"]
        track_b = [f for f in (llm_flags or []) if f.track == "B"]
        per_ad_plaus = {k: 1.0 for k in audit.per_ad_plausibility}
        for f in track_b:
            if f.target_ad_id:
                per_ad_plaus[f.target_ad_id] = min(
                    per_ad_plaus.get(f.target_ad_id, 1.0),
                    max(0.0, 1.0 - (f.severity or 0.5)),
                )
    elif arm == AblationArm.HYBRID:
        det = list(audit.track_a_flags) + list(audit.track_b_flags)
        merged = merge_hybrid(det, llm_flags or [])
        track_a = [f for f in merged if f.track == "A"]
        track_b = [f for f in merged if f.track == "B"]
        per_ad_plaus = dict(audit.per_ad_plausibility)
        extra_b = {(f.flag_type, f.target_ad_id) for f in audit.track_b_flags}
        for f in track_b:
            key = (f.flag_type, f.target_ad_id)
            if key in extra_b or not f.target_ad_id:
                continue
            per_ad_plaus[f.target_ad_id] = min(
                per_ad_plaus.get(f.target_ad_id, 1.0),
                max(0.0, 1.0 - (f.severity or 0.5)),
            )
    else:
        track_a = _filter_flags(list(audit.track_a_flags), arm)
        if arm in ARM_PLAUS_WEIGHTS:
            cache = build_reward_cache(
                bundle.fraudster_proposals, weights=ARM_PLAUS_WEIGHTS[arm]
            )
            per_ad_plaus = dict(cache.per_ad_plausibility)
            track_b = _filter_flags(list(cache.track_b_flags), arm)
        else:
            track_b = _filter_flags(list(audit.track_b_flags), arm)
            per_ad_plaus = dict(audit.per_ad_plausibility)
        if arm in (AblationArm.NO_TRACK_B, AblationArm.NO_PLAUSIBILITY_GATE):
            per_ad_plaus = {k: 1.0 for k in per_ad_plaus}

    severity_by_ad = {
        m.get("ad_id", ""): float(m.get("severity", 0.5) or 0.5)
        for m in record.ads_metadata
    }
    verdict_by_ad = {v.ad_id: v.verdict for v in record.verdicts}
    fraudster_ads = set(bundle.fraudster_ad_ids)
    unbanned = [ad_id for ad_id in fraudster_ads if verdict_by_ad.get(ad_id) != "reject"]
    fraud_r = 0.0
    for ad_id in unbanned:
        sev = severity_by_ad.get(ad_id, 0.6)
        plaus = per_ad_plaus.get(ad_id, 0.0)
        if arm == AblationArm.NO_PLAUSIBILITY_GATE:
            fraud_r += FRAUDSTER_PER_AD_SEVERITY_WEIGHT * sev
        else:
            fraud_r += FRAUDSTER_PER_AD_SEVERITY_WEIGHT * sev * plaus
    banned = sum(1 for ad_id in fraudster_ads if verdict_by_ad.get(ad_id) == "reject")
    fraud_r -= FRAUDSTER_BANNED_PENALTY * banned

    flagged_ids = {
        f.target_ad_id
        for f in track_a
        if f.target_ad_id and f.flag_type in INVESTIGATOR_RATIONALE_FLAG_TYPES
    }
    if arm == AblationArm.NO_TRACK_A:
        flagged_ids = set()
    weighted_clean = 0.0
    for v in record.verdicts:
        if v.ad_id in flagged_ids:
            continue
        weighted_clean += per_ad_plaus.get(v.ad_id, 1.0)
    n_incons = min(
        sum(1 for f in track_a if f.flag_type == "inconsistency"),
        INVESTIGATOR_INCONSISTENCY_CAP,
    )
    if arm == AblationArm.NO_TRACK_A or arm == AblationArm.NO_INCONSISTENCY:
        n_incons = 0
    inv_r = (
        grader_score
        + INVESTIGATOR_RATIONALE_BONUS * weighted_clean
        - INVESTIGATOR_INCONSISTENCY_PENALTY * n_incons
    )
    mean_plaus = (
        sum(per_ad_plaus.values()) / len(per_ad_plaus) if per_ad_plaus else 1.0
    )
    return RoleRewards(
        arm=arm.value,
        fraudster_reward=fraud_r,
        investigator_reward=inv_r,
        grader_score=grader_score,
        mean_plausibility=mean_plaus,
        n_track_a=len(track_a),
        n_track_b=len(track_b),
    )
