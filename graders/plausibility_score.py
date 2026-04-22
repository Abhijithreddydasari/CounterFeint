"""
Aggregation layer for Track B plausibility.

`compute_plausibility_score` is the single value consumed by the
multi-agent reward formula (graders/multi_agent_rewards.py).  It
combines the five per-dimension checks into a weighted score in
[0, 1] with configurable dimension weights and an optional fallback
mode (intrinsic + grounding only) if overall false-positive rate is
too high during audit calibration.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..models import AuditFlag
from .auditor_track_b import (
    intrinsic_consistency_check,
    parameter_grounding_check,
    pattern_novelty_check,
    real_world_existence_check,
    signal_realism_check,
)

# -----------------------------------------------------------------------------
# Default weights.  Start equal (0.2 each) per the plan; tune during Phase 3.
# -----------------------------------------------------------------------------

DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "intrinsic_consistency": 0.2,
    "parameter_grounding": 0.2,
    "real_world_existence": 0.2,
    "signal_realism": 0.2,
    "pattern_novelty": 0.2,
}

# Fallback mode reduces Track B to its two most FP-resilient dimensions.
FALLBACK_DIMENSION_WEIGHTS: Dict[str, float] = {
    "intrinsic_consistency": 0.5,
    "parameter_grounding": 0.5,
    "real_world_existence": 0.0,
    "signal_realism": 0.0,
    "pattern_novelty": 0.0,
}


# Weighted-mean alone lets one perfect dimension drag the overall score above
# thresholds even when another dimension vetoes (e.g. ad contains a deny
# token → real_world_existence = 0 but intrinsic_consistency = 1).  To make
# "one catastrophic check = low plausibility" the correct default behaviour,
# we subtract a veto penalty that kicks in whenever the worst dimension is
# below this threshold.
VETO_DIMENSION_THRESHOLD: float = 0.6


def compute_plausibility_score(
    proposal: Mapping[str, Any],
    *,
    queue_context: Optional[Iterable[Mapping[str, Any]]] = None,
    advertiser_country: Optional[str] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> Tuple[float, List[AuditFlag]]:
    """
    Compute a single-ad plausibility score in [0, 1].

    The queue context is optional; when provided we apply the queue-level
    pattern-novelty dimension as a multiplicative factor on the per-ad
    score so ad 3 of a 12-ad clone attack gets dragged down even if it
    *individually* looks plausible.
    """
    w = dict(DEFAULT_DIMENSION_WEIGHTS)
    if weights:
        w.update(weights)

    flags: List[AuditFlag] = []
    components: Dict[str, float] = {}

    s1, f1 = intrinsic_consistency_check(proposal)
    flags.extend(f1)
    components["intrinsic_consistency"] = s1

    s2, f2 = parameter_grounding_check(proposal, advertiser_country=advertiser_country)
    flags.extend(f2)
    components["parameter_grounding"] = s2

    s3, f3 = real_world_existence_check(proposal)
    flags.extend(f3)
    components["real_world_existence"] = s3

    s4, f4 = signal_realism_check(proposal)
    flags.extend(f4)
    components["signal_realism"] = s4

    queue = list(queue_context or [])
    s5, f5 = pattern_novelty_check(queue) if queue else (1.0, [])
    flags.extend(f5)
    components["pattern_novelty"] = s5

    # Weighted mean over *active* (non-zero-weight) dimensions so disabling a
    # dimension via the fallback weights also removes it from the veto check.
    active = {k: v for k, v in components.items() if w.get(k, 0.0) > 0}
    total_w = sum(w[k] for k in active) or 1.0
    weighted = sum(active[k] * w[k] for k in active) / total_w

    # Worst-dimension veto: any active dimension below the threshold drags
    # the score down proportionally.
    worst = min(active.values()) if active else 1.0
    veto = max(0.0, VETO_DIMENSION_THRESHOLD - worst)

    return max(0.0, min(1.0, weighted - veto)), flags


def compute_queue_plausibility(
    proposals: Iterable[Mapping[str, Any]],
    *,
    country_by_ad_id: Optional[Mapping[str, str]] = None,
    weights: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, float], List[AuditFlag], float]:
    """
    Compute per-ad plausibility + a queue-level aggregate.

    Returns
    -------
    per_ad_scores : dict[ad_id, float]
    flags : list[AuditFlag]  (Track B only)
    queue_plausibility : float
        Mean of per-ad scores. Used by the Auditor as
        `fraudster_plausibility_score` on the AuditReport.
    """
    # The list may contain the full Fraudster action log (propose / modify /
    # skip) or a pre-filtered list of proposal dicts.  Treat anything that
    # isn't an explicit "skip" as proposal-like content so scripted / test
    # callers can pass minimal dicts without an `action_type` key.
    queue = [
        p
        for p in proposals
        if p.get("action_type") not in ("skip",)
    ]
    per_ad_scores: Dict[str, float] = {}
    all_flags: List[AuditFlag] = []

    for p in queue:
        ad_id = str(p.get("ad_id") or f"slot_{p.get('slot_index')}")
        country = None
        if country_by_ad_id and ad_id in country_by_ad_id:
            country = country_by_ad_id[ad_id]
        score, flags = compute_plausibility_score(
            p,
            queue_context=queue,
            advertiser_country=country,
            weights=weights,
        )
        per_ad_scores[ad_id] = score
        all_flags.extend(flags)

    queue_plausibility = (
        sum(per_ad_scores.values()) / len(per_ad_scores)
        if per_ad_scores
        else 1.0
    )
    return per_ad_scores, all_flags, queue_plausibility


__all__ = [
    "DEFAULT_DIMENSION_WEIGHTS",
    "FALLBACK_DIMENSION_WEIGHTS",
    "compute_plausibility_score",
    "compute_queue_plausibility",
]
