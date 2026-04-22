"""
Auditor — Track A: rule-based audit of the *Investigator's reasoning*.

All five audit checks are fully deterministic — no LLM-judge, no
external API, no randomness.  Each function returns a `list[AuditFlag]`
describing any violations found in the investigator's behavior during
an episode.

Design notes:
- Each check is self-contained and can be unit-tested in isolation.
- Flags carry a severity in [0, 1] so downstream reward logic can weight
  minor issues (e.g. one slightly-short rationale) differently from
  catastrophic ones (e.g. fully miscalibrated confidence).
- Track A flags all have track="A"; Track B flags (plausibility audit
  of the Fraudster) live in `auditor_track_b.py`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..data.audit_heuristics import (
    GREEN_FLAG_PHRASES,
    RED_FLAG_PHRASES,
    count_citations_matching_findings,
    count_phrase_hits,
    extract_evidence_tokens,
    hash_ad_pair,
    severity_slice,
)
from ..models import AuditFlag
from .base_grader import EpisodeRecord, VerdictResult

# -----------------------------------------------------------------------------
# Tunable thresholds (keep in one place so tests can import and verify).
# -----------------------------------------------------------------------------

# Calibration bins.
_CAL_BINS = (
    ("low", 0.0, 0.4),
    ("mid", 0.4, 0.7),
    ("high", 0.7, 1.01),
)
CALIBRATION_GAP_THRESHOLD = 0.3  # confidence-vs-accuracy gap per bin
MIN_BIN_SIZE_FOR_CALIB = 2

MIN_RATIONALE_CHARS = 24
CITATION_MIN_TOKENS = 1

RED_COUNT_FOR_CONTRADICTION = 2
GREEN_COUNT_FOR_CONTRADICTION = 2

BIAS_ACC_GAP = 0.25
BIAS_MIN_SLICE_SIZE = 3

MIN_PAIRS_FOR_CONSISTENCY = 1  # any matched pair with divergent verdicts is enough


# -----------------------------------------------------------------------------
# 1. Calibration audit
# -----------------------------------------------------------------------------


def calibration_audit(record: EpisodeRecord) -> List[AuditFlag]:
    """
    Flag the Investigator when stated confidence does not track accuracy.

    Bins verdicts into low / mid / high confidence and computes
    |avg_confidence - avg_accuracy| per bin. Any bin with gap >
    CALIBRATION_GAP_THRESHOLD and enough samples gets flagged.
    """
    manual = [v for v in record.verdicts if not v.auto_approved]
    if len(manual) < 3:
        return []

    flags: List[AuditFlag] = []
    for label, lo, hi in _CAL_BINS:
        group = [v for v in manual if lo <= (v.confidence or 0.0) < hi]
        if len(group) < MIN_BIN_SIZE_FOR_CALIB:
            continue
        avg_conf = sum((v.confidence or 0.0) for v in group) / len(group)
        avg_acc = sum(1 for v in group if _is_correct(v)) / len(group)
        # Only flag over-confidence: the Investigator claimed certainty it didn't
        # have. Under-confidence (being right more often than you say) is sub-
        # optimal but not an audit failure.
        gap = avg_conf - avg_acc
        if gap > CALIBRATION_GAP_THRESHOLD:
            flags.append(
                AuditFlag(
                    track="A",
                    target_ad_id=None,
                    flag_type="miscalibration",
                    severity=min(1.0, gap),
                    note=(
                        f"Calibration bin '{label}' ({lo:.2f}-{hi:.2f}): "
                        f"avg_conf={avg_conf:.2f}, avg_acc={avg_acc:.2f}, "
                        f"over-confidence gap={gap:.2f} over {len(group)} verdict(s)."
                    ),
                )
            )
    return flags


# -----------------------------------------------------------------------------
# 2. Rationale citation audit
# -----------------------------------------------------------------------------


def rationale_citation_audit(
    investigator_actions: Iterable[Mapping[str, Any]],
    investigation_data_seen: Mapping[str, Mapping[str, str]],
) -> List[AuditFlag]:
    """
    Flag verdicts whose rationale cites nothing from the investigation data.

    For each `verdict` action we:
      1. Concatenate all findings the Investigator pulled for that ad.
      2. Count evidence-shaped tokens in the rationale that also appear
         in the findings (payment ids, domains, registrars, numeric
         claims, and red/green phrases).
      3. Flag if fewer than `CITATION_MIN_TOKENS` tokens match OR if the
         rationale is trivially short.
    """
    flags: List[AuditFlag] = []
    for act in investigator_actions:
        if act.get("action_type") != "verdict":
            continue
        ad_id = act.get("ad_id")
        rationale = (act.get("rationale") or "").strip()
        findings_map = investigation_data_seen.get(ad_id or "", {}) or {}
        findings = "\n".join(str(v) for v in findings_map.values())

        if len(rationale) < MIN_RATIONALE_CHARS:
            flags.append(
                AuditFlag(
                    track="A",
                    target_ad_id=ad_id,
                    flag_type="missing_citation",
                    severity=0.7,
                    note=(
                        f"Rationale too short ({len(rationale)} chars < "
                        f"{MIN_RATIONALE_CHARS}); nothing to audit."
                    ),
                )
            )
            continue

        if not findings:
            # Nothing was investigated — can't require citations, but the
            # verdict is still reasoned about *something*; weak flag.
            if not extract_evidence_tokens(rationale):
                flags.append(
                    AuditFlag(
                        track="A",
                        target_ad_id=ad_id,
                        flag_type="missing_citation",
                        severity=0.4,
                        note=(
                            "Verdict rendered with no investigation pulled "
                            "and no evidence-shaped tokens in rationale."
                        ),
                    )
                )
            continue

        hits = count_citations_matching_findings(rationale, findings)
        if hits < CITATION_MIN_TOKENS:
            flags.append(
                AuditFlag(
                    track="A",
                    target_ad_id=ad_id,
                    flag_type="missing_citation",
                    severity=0.6,
                    note=(
                        f"Rationale cites 0 tokens present in findings "
                        f"(len={len(rationale)})."
                    ),
                )
            )
    return flags


# -----------------------------------------------------------------------------
# 3. Rationale ↔ verdict coherence audit
# -----------------------------------------------------------------------------


def rationale_verdict_coherence_audit(
    investigator_actions: Iterable[Mapping[str, Any]],
) -> List[AuditFlag]:
    """
    Flag verdicts whose rationale contradicts the issued verdict.

    Heuristic (deterministic, keyword-driven):
      • ≥2 red-flag phrases in rationale AND verdict == "approve" → contradiction
      • ≥2 green-flag phrases in rationale AND verdict == "reject" → contradiction
    Escalations get a pass since they are explicitly the "uncertain" verdict.
    """
    flags: List[AuditFlag] = []
    for act in investigator_actions:
        if act.get("action_type") != "verdict":
            continue
        verdict = (act.get("verdict") or "").lower()
        rationale = act.get("rationale") or ""
        if verdict == "escalate" or not rationale:
            continue

        red = count_phrase_hits(rationale, RED_FLAG_PHRASES)
        green = count_phrase_hits(rationale, GREEN_FLAG_PHRASES)

        if verdict == "approve" and red >= RED_COUNT_FOR_CONTRADICTION:
            flags.append(
                AuditFlag(
                    track="A",
                    target_ad_id=act.get("ad_id"),
                    flag_type="incoherent_rationale",
                    severity=min(1.0, 0.3 + 0.2 * red),
                    note=(
                        f"Verdict='approve' but rationale cites {red} "
                        "red-flag phrase(s) — contradictory."
                    ),
                )
            )
        elif verdict == "reject" and green >= GREEN_COUNT_FOR_CONTRADICTION:
            flags.append(
                AuditFlag(
                    track="A",
                    target_ad_id=act.get("ad_id"),
                    flag_type="incoherent_rationale",
                    severity=min(1.0, 0.3 + 0.2 * green),
                    note=(
                        f"Verdict='reject' but rationale cites {green} "
                        "green-flag phrase(s) — contradictory."
                    ),
                )
            )
    return flags


# -----------------------------------------------------------------------------
# 4. Cross-ad consistency audit
# -----------------------------------------------------------------------------


def cross_ad_consistency_audit(
    record: EpisodeRecord,
) -> List[AuditFlag]:
    """
    Flag the Investigator when structurally-similar ads get diverging verdicts.

    `hash_ad_pair` decides which ads are "structurally similar" (same
    ground truth, same fraud_type/category, same severity bucket).  Any
    such pair with diverging verdicts is an inconsistency.
    """
    verdict_by_ad = {v.ad_id: v for v in record.verdicts if not v.auto_approved}
    ads_by_id = {m.get("ad_id"): m for m in record.ads_metadata if m.get("ad_id")}

    flags: List[AuditFlag] = []
    seen_pairs: set = set()
    ad_ids = sorted(verdict_by_ad.keys())

    for i, ad_id_a in enumerate(ad_ids):
        meta_a = ads_by_id.get(ad_id_a)
        if not meta_a:
            continue
        v_a = verdict_by_ad.get(ad_id_a)
        if not v_a:
            continue
        for ad_id_b in ad_ids[i + 1 :]:
            meta_b = ads_by_id.get(ad_id_b)
            if not meta_b:
                continue
            v_b = verdict_by_ad.get(ad_id_b)
            if not v_b:
                continue
            pair_key = hash_ad_pair(meta_a, meta_b)
            if pair_key is None or pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            if v_a.verdict != v_b.verdict:
                flags.append(
                    AuditFlag(
                        track="A",
                        target_ad_id=ad_id_a,
                        flag_type="inconsistency",
                        severity=0.7,
                        note=(
                            f"Structurally-similar pair ({ad_id_a}, {ad_id_b}) "
                            f"received diverging verdicts "
                            f"('{v_a.verdict}' vs '{v_b.verdict}')."
                        ),
                    )
                )
    return flags


# -----------------------------------------------------------------------------
# 5. Bias audit
# -----------------------------------------------------------------------------


def bias_audit(record: EpisodeRecord) -> List[AuditFlag]:
    """
    Flag the Investigator when per-slice accuracy differs by > BIAS_ACC_GAP.

    Slices used:
      • `country`   (if present in ads_metadata)
      • `category`  (if present in ads_metadata)
      • `severity_bucket` (always derived)

    A slice with at least BIAS_MIN_SLICE_SIZE ads is compared against
    the complement. Gap above threshold → flag.
    """
    verdict_by_ad = {v.ad_id: v for v in record.verdicts if not v.auto_approved}
    ads_by_id = {m.get("ad_id"): m for m in record.ads_metadata if m.get("ad_id")}
    if len(verdict_by_ad) < 2 * BIAS_MIN_SLICE_SIZE:
        return []

    def _per_ad_correct(ad_id: str) -> Optional[bool]:
        v = verdict_by_ad.get(ad_id)
        if not v:
            return None
        return _is_correct(v)

    flags: List[AuditFlag] = []
    slice_extractors = {
        "country": lambda m: (m.get("country") or None),
        "category": lambda m: (m.get("category") or None),
        "severity_bucket": lambda m: severity_slice(m.get("severity")),
    }

    for dimension, extractor in slice_extractors.items():
        buckets: Dict[str, List[bool]] = {}
        for ad_id, meta in ads_by_id.items():
            key = extractor(meta)
            if key is None:
                continue
            correct = _per_ad_correct(ad_id)
            if correct is None:
                continue
            buckets.setdefault(str(key), []).append(correct)

        bucket_accs = {
            k: (sum(v) / len(v)) for k, v in buckets.items()
            if len(v) >= BIAS_MIN_SLICE_SIZE
        }
        if len(bucket_accs) < 2:
            continue

        best_k = max(bucket_accs, key=bucket_accs.get)
        worst_k = min(bucket_accs, key=bucket_accs.get)
        gap = bucket_accs[best_k] - bucket_accs[worst_k]
        if gap > BIAS_ACC_GAP:
            flags.append(
                AuditFlag(
                    track="A",
                    target_ad_id=None,
                    flag_type="bias",
                    severity=min(1.0, gap),
                    note=(
                        f"Bias on dimension '{dimension}': best='{best_k}' "
                        f"(acc={bucket_accs[best_k]:.2f}, n={len(buckets[best_k])}) "
                        f"vs worst='{worst_k}' "
                        f"(acc={bucket_accs[worst_k]:.2f}, n={len(buckets[worst_k])}); "
                        f"gap={gap:.2f}."
                    ),
                )
            )
    return flags


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def run_track_a(
    record: EpisodeRecord,
    *,
    investigator_actions: Iterable[Mapping[str, Any]],
    investigation_data_seen: Mapping[str, Mapping[str, str]],
) -> List[AuditFlag]:
    """
    Run all five Track A checks and return the combined flag list.

    Order is stable so reports are reproducible in tests.
    """
    flags: List[AuditFlag] = []
    flags.extend(calibration_audit(record))
    flags.extend(
        rationale_citation_audit(investigator_actions, investigation_data_seen)
    )
    flags.extend(rationale_verdict_coherence_audit(investigator_actions))
    flags.extend(cross_ad_consistency_audit(record))
    flags.extend(bias_audit(record))
    return flags


def investigator_audit_score(flags: Iterable[AuditFlag]) -> float:
    """
    Aggregate Track A flags into a single [0, 1] score.

    1.0 = clean. Score decays linearly in total severity, but with a
    weight ceiling so one big-severity flag doesn't tank the score as
    much as several medium flags (which is the failure mode we care
    about most).
    """
    a_flags = [f for f in flags if f.track == "A"]
    if not a_flags:
        return 1.0

    # Weight each flag type so consistency/inconsistency and
    # miscalibration carry more weight than a single short rationale.
    weights = {
        "miscalibration": 0.25,
        "incoherent_rationale": 0.22,
        "inconsistency": 0.20,
        "bias": 0.18,
        "missing_citation": 0.12,
    }

    loss = 0.0
    for flag in a_flags:
        w = weights.get(flag.flag_type, 0.15)
        loss += w * (flag.severity or 0.5)

    return max(0.0, min(1.0, 1.0 - loss))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _is_correct(v: VerdictResult) -> bool:
    return (
        (v.verdict == "reject" and v.ground_truth == "fraud")
        or (v.verdict == "approve" and v.ground_truth == "legit")
        or (v.verdict == "escalate" and v.ground_truth == "escalate")
    )


__all__ = [
    "BIAS_ACC_GAP",
    "BIAS_MIN_SLICE_SIZE",
    "CALIBRATION_GAP_THRESHOLD",
    "CITATION_MIN_TOKENS",
    "GREEN_COUNT_FOR_CONTRADICTION",
    "MIN_BIN_SIZE_FOR_CALIB",
    "MIN_RATIONALE_CHARS",
    "RED_COUNT_FOR_CONTRADICTION",
    "bias_audit",
    "calibration_audit",
    "cross_ad_consistency_audit",
    "investigator_audit_score",
    "rationale_citation_audit",
    "rationale_verdict_coherence_audit",
    "run_track_a",
]
