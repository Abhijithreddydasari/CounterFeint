"""
Auditor — Track B: rule-based plausibility audit of the *Fraudster's ads*.

Five dimensions, each returning `(score ∈ [0,1], flags: list[AuditFlag])`:

1. intrinsic_consistency_check   — category ↔ targeting compatibility
2. parameter_grounding_check     — country ↔ TLD, copy-vs-targeting
3. real_world_existence_check    — grammar / non-ASCII / repetition / deny list
4. signal_realism_check          — no R1 grader-trigger leakage
5. pattern_novelty_check         — queue-level diversity (edit distance + Jaccard)

The five checks feed into `plausibility_score.compute_plausibility_score`
which aggregates them into a single Fraudster-plausibility score used by
the multi-agent reward model.

All checks are fully deterministic — safe to run at judging time.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..data.plausibility_references import (
    contains_deny_token,
    contains_fraud_marker,
    extract_tlds_from_text,
    is_adult_only_category,
    is_high_risk_tld,
    is_tld_plausible_for_country,
    targeting_matches_category,
    targeting_mentions_minors,
)
from ..models import AuditFlag

# -----------------------------------------------------------------------------
# Tunables
# -----------------------------------------------------------------------------

NON_ASCII_MAX_RATIO = 0.2
REPEATED_NGRAM_MAX = 4  # > this many repeats of any 5-gram → suspicious
MIN_AD_COPY_CHARS = 20
MIN_WORDLIKE_RATIO = 0.5
PATTERN_NOVELTY_MIN_QUEUE = 2
PATTERN_NOVELTY_JACCARD_FLAG = 0.7
PATTERN_NOVELTY_EDITDIST_FLAG = 0.2  # normalized edit distance below this → clones


# -----------------------------------------------------------------------------
# Proposal shape
#
# All checks expect a dict-like "proposal" record containing any of:
#   ad_id, ad_copy, category, landing_page_blurb, targeting_summary, rationale
#
# This matches `_serialize_fraudster_action` in server/referee.py so we can
# audit the fraudster_log directly.
# -----------------------------------------------------------------------------


def _get_text(proposal: Mapping[str, Any], *keys: str) -> str:
    for k in keys:
        v = proposal.get(k)
        if v:
            return str(v)
    return ""


# -----------------------------------------------------------------------------
# 1. Intrinsic consistency
# -----------------------------------------------------------------------------


def intrinsic_consistency_check(
    proposal: Mapping[str, Any],
) -> Tuple[float, List[AuditFlag]]:
    """
    Flag ads whose declared category doesn't match the targeting / copy
    it actually uses.
    """
    ad_id = proposal.get("ad_id")
    category = str(proposal.get("category") or "").lower()
    targeting = _get_text(proposal, "targeting_summary")
    ad_copy = _get_text(proposal, "ad_copy", "new_ad_copy")

    if not category:
        return 0.5, [
            AuditFlag(
                track="B",
                target_ad_id=ad_id,
                flag_type="parameter_mismatch",
                severity=0.5,
                note="Ad has no declared category; cannot assess intrinsic consistency.",
            )
        ]

    flags: List[AuditFlag] = []
    score = 1.0

    if is_adult_only_category(category) and targeting_mentions_minors(targeting):
        flags.append(
            AuditFlag(
                track="B",
                target_ad_id=ad_id,
                flag_type="parameter_mismatch",
                severity=1.0,
                note=(
                    f"Category '{category}' targets minors in targeting "
                    f"string: {targeting[:120]!r}"
                ),
            )
        )
        score -= 0.7

    if targeting and not targeting_matches_category(category, targeting):
        flags.append(
            AuditFlag(
                track="B",
                target_ad_id=ad_id,
                flag_type="parameter_mismatch",
                severity=0.6,
                note=(
                    f"Targeting string does not mention any expected keyword "
                    f"for category '{category}'."
                ),
            )
        )
        score -= 0.3

    combined = f"{ad_copy}\n{targeting}".lower()
    if category in {"fake_crypto", "network_crypto"} and "crypto" not in combined and "investing" not in combined:
        flags.append(
            AuditFlag(
                track="B",
                target_ad_id=ad_id,
                flag_type="parameter_mismatch",
                severity=0.4,
                note=(
                    f"Category '{category}' but ad copy / targeting never "
                    "mentions crypto or investing — implausible surface."
                ),
            )
        )
        score -= 0.2

    return max(0.0, min(1.0, score)), flags


# -----------------------------------------------------------------------------
# 2. Parameter grounding
# -----------------------------------------------------------------------------


def parameter_grounding_check(
    proposal: Mapping[str, Any],
    *,
    advertiser_country: Optional[str] = None,
) -> Tuple[float, List[AuditFlag]]:
    """
    Flag country↔TLD mismatches and high-risk TLDs.

    `advertiser_country` is optional — if the caller (typically the Referee)
    knows the advertiser profile it can pass it in; otherwise we fall back
    to scanning ad_copy / landing_page_blurb for explicit country claims.
    """
    ad_id = proposal.get("ad_id")
    blurb = _get_text(proposal, "landing_page_blurb", "new_landing_page_blurb")
    ad_copy = _get_text(proposal, "ad_copy", "new_ad_copy")

    country = (advertiser_country or "").upper()
    if not country:
        # Try to infer from ad_copy via simple ISO tokens.
        tokens = re.findall(r"\b(US|USA|UK|GB|DE|FR|ES|CA|AU|NL|BR|IN|JP|CN|RU|NG)\b", ad_copy)
        if tokens:
            raw = tokens[0].upper()
            country = "GB" if raw in ("UK", "GB") else raw

    flags: List[AuditFlag] = []
    score = 1.0
    tlds = extract_tlds_from_text(blurb) | extract_tlds_from_text(ad_copy)

    for tld in tlds:
        if country and not is_tld_plausible_for_country(country, tld):
            flags.append(
                AuditFlag(
                    track="B",
                    target_ad_id=ad_id,
                    flag_type="parameter_mismatch",
                    severity=0.8,
                    note=(
                        f"TLD '.{tld}' is implausible for advertiser country "
                        f"'{country}'."
                    ),
                )
            )
            score -= 0.4

        if is_high_risk_tld(tld):
            flags.append(
                AuditFlag(
                    track="B",
                    target_ad_id=ad_id,
                    flag_type="branding_anomaly",
                    severity=0.4,
                    note=f"High-risk TLD '.{tld}' used in landing page surface.",
                )
            )
            score -= 0.15

    return max(0.0, min(1.0, score)), flags


# -----------------------------------------------------------------------------
# 3. Real-world existence
# -----------------------------------------------------------------------------


def real_world_existence_check(
    proposal: Mapping[str, Any],
) -> Tuple[float, List[AuditFlag]]:
    """
    Flag gibberish / nonsense ads: non-ASCII floods, repeated n-grams,
    deny-list tokens, non-wordlike alphabetic ratios.
    """
    ad_id = proposal.get("ad_id")
    text = _get_text(proposal, "ad_copy", "new_ad_copy")
    blurb = _get_text(proposal, "landing_page_blurb", "new_landing_page_blurb")
    combined = f"{text} {blurb}".strip()

    flags: List[AuditFlag] = []
    score = 1.0

    if not combined or len(text) < MIN_AD_COPY_CHARS:
        flags.append(
            AuditFlag(
                track="B",
                target_ad_id=ad_id,
                flag_type="gibberish",
                severity=0.8,
                note=f"Ad copy is too short ({len(text)} chars) to be plausible.",
            )
        )
        score -= 0.6
        return max(0.0, score), flags

    if contains_deny_token(combined):
        flags.append(
            AuditFlag(
                track="B",
                target_ad_id=ad_id,
                flag_type="gibberish",
                severity=1.0,
                note="Ad contains adversarial / prompt-injection tokens.",
            )
        )
        score -= 0.7

    cleaned = unicodedata.normalize("NFKC", combined)
    if any(ord(c) < 0x20 and c not in "\t\n\r" for c in cleaned):
        flags.append(
            AuditFlag(
                track="B",
                target_ad_id=ad_id,
                flag_type="gibberish",
                severity=0.9,
                note="Ad contains control characters.",
            )
        )
        score -= 0.5

    non_ascii = sum(1 for c in combined if ord(c) > 127)
    if non_ascii / max(1, len(combined)) > NON_ASCII_MAX_RATIO:
        flags.append(
            AuditFlag(
                track="B",
                target_ad_id=ad_id,
                flag_type="gibberish",
                severity=0.7,
                note=(
                    f"Non-ASCII ratio {non_ascii/len(combined):.2f} exceeds "
                    f"{NON_ASCII_MAX_RATIO:.2f}."
                ),
            )
        )
        score -= 0.3

    words = [w for w in re.findall(r"[A-Za-z]+", combined) if len(w) > 1]
    if words:
        wordlike = sum(1 for w in words if _looks_wordlike(w)) / len(words)
        if wordlike < MIN_WORDLIKE_RATIO:
            # Scale the penalty with how gibberish the queue actually looks.
            # A 10% word-like ratio is much worse than a 45% word-like ratio.
            severity = min(1.0, 0.6 + (MIN_WORDLIKE_RATIO - wordlike))
            flags.append(
                AuditFlag(
                    track="B",
                    target_ad_id=ad_id,
                    flag_type="gibberish",
                    severity=severity,
                    note=(
                        f"Only {wordlike:.0%} of words look pronounceable; "
                        "likely gibberish."
                    ),
                )
            )
            score -= min(0.8, 0.4 + (MIN_WORDLIKE_RATIO - wordlike) * 1.2)

    ngram_repeats = _max_ngram_repeat(combined, n=5)
    if ngram_repeats > REPEATED_NGRAM_MAX:
        flags.append(
            AuditFlag(
                track="B",
                target_ad_id=ad_id,
                flag_type="template_repetition",
                severity=min(1.0, 0.2 + 0.1 * ngram_repeats),
                note=(
                    f"Same 5-gram repeated {ngram_repeats}× — template collapse."
                ),
            )
        )
        score -= 0.2

    return max(0.0, min(1.0, score)), flags


# -----------------------------------------------------------------------------
# 4. Signal realism
# -----------------------------------------------------------------------------


def signal_realism_check(
    proposal: Mapping[str, Any],
) -> Tuple[float, List[AuditFlag]]:
    """
    Flag Fraudsters who copy R1 grader-trigger strings into their own ads
    (pmt_ring_*, tmpl_*, ring_*, cfg_*, adv_*_fraud).  Those are the exact
    patterns the R1 data generators use to signal fraud — a Fraudster
    copying them is trying to trip graders, not fool a reviewer.
    """
    ad_id = proposal.get("ad_id")
    text = _get_text(proposal, "ad_copy", "new_ad_copy") + "\n" + _get_text(
        proposal, "landing_page_blurb", "new_landing_page_blurb"
    )

    if contains_fraud_marker(text):
        return 0.0, [
            AuditFlag(
                track="B",
                target_ad_id=ad_id,
                flag_type="branding_anomaly",
                severity=1.0,
                note=(
                    "Ad surfaces contain R1 grader-trigger patterns "
                    "(e.g. tmpl_*, pmt_ring_*, ring_*)."
                ),
            )
        ]
    return 1.0, []


# -----------------------------------------------------------------------------
# 5. Pattern novelty
# -----------------------------------------------------------------------------


def pattern_novelty_check(
    queue: Iterable[Mapping[str, Any]],
) -> Tuple[float, List[AuditFlag]]:
    """
    Flag Fraudsters who submit near-identical ads across the whole queue.

    We compute pairwise normalized edit distance and 5-gram Jaccard
    across ad_copy strings.  If the average Jaccard is too high or the
    average normalized edit distance is too low, the queue lacks novelty.
    """
    ads = [p for p in queue if p.get("action_type") == "propose_ad"]
    if len(ads) < PATTERN_NOVELTY_MIN_QUEUE:
        return 1.0, []

    texts = [
        (p.get("ad_id"), str(p.get("ad_copy") or "")) for p in ads
    ]

    # Pairwise similarity.
    pair_count = 0
    jaccard_sum = 0.0
    edit_sum = 0.0
    for i in range(len(texts)):
        a_id, a = texts[i]
        if not a:
            continue
        for j in range(i + 1, len(texts)):
            b_id, b = texts[j]
            if not b:
                continue
            pair_count += 1
            jaccard_sum += _ngram_jaccard(a, b, n=5)
            edit_sum += _normalized_edit_distance(a, b)

    if pair_count == 0:
        return 1.0, []

    mean_jaccard = jaccard_sum / pair_count
    mean_edit = edit_sum / pair_count

    flags: List[AuditFlag] = []
    score = 1.0
    if mean_jaccard >= PATTERN_NOVELTY_JACCARD_FLAG:
        flags.append(
            AuditFlag(
                track="B",
                target_ad_id=None,
                flag_type="template_repetition",
                severity=min(1.0, mean_jaccard),
                note=(
                    f"Queue mean 5-gram Jaccard={mean_jaccard:.2f} — "
                    "ads reuse the same template surface."
                ),
            )
        )
        score -= 0.4 + 0.2 * (mean_jaccard - PATTERN_NOVELTY_JACCARD_FLAG)

    if mean_edit <= PATTERN_NOVELTY_EDITDIST_FLAG:
        flags.append(
            AuditFlag(
                track="B",
                target_ad_id=None,
                flag_type="template_repetition",
                severity=min(1.0, 1.0 - mean_edit),
                note=(
                    f"Queue mean normalized edit distance={mean_edit:.2f} — "
                    "ads differ by only a few tokens."
                ),
            )
        )
        score -= 0.3

    return max(0.0, min(1.0, score)), flags


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def run_track_b(
    proposals: Iterable[Mapping[str, Any]],
    *,
    country_by_ad_id: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, float], List[AuditFlag]]:
    """
    Run all five Track B checks over a Fraudster's proposal queue.

    Returns
    -------
    per_ad_scores : dict[ad_id, float]
        Mean per-ad plausibility in [0, 1].
    flags : list[AuditFlag]
        All Track B flags raised by any of the five checks.
    """
    queue = [p for p in proposals if p.get("action_type") == "propose_ad"]
    per_ad_scores: Dict[str, float] = {}
    all_flags: List[AuditFlag] = []

    for proposal in queue:
        ad_id = str(proposal.get("ad_id") or f"slot_{proposal.get('slot_index')}")
        country = None
        if country_by_ad_id and ad_id in country_by_ad_id:
            country = country_by_ad_id[ad_id]

        s1, f1 = intrinsic_consistency_check(proposal)
        s2, f2 = parameter_grounding_check(proposal, advertiser_country=country)
        s3, f3 = real_world_existence_check(proposal)
        s4, f4 = signal_realism_check(proposal)
        per_ad_scores[ad_id] = sum([s1, s2, s3, s4]) / 4.0
        all_flags.extend(f1 + f2 + f3 + f4)

    s5, f5 = pattern_novelty_check(queue)
    # Multiply every per-ad score by queue-level novelty so template-collapse
    # drags down the whole queue.
    for ad_id in list(per_ad_scores.keys()):
        per_ad_scores[ad_id] *= s5
    all_flags.extend(f5)

    return per_ad_scores, all_flags


# -----------------------------------------------------------------------------
# Internal helpers (pure functions; unit-tested indirectly via the tests)
# -----------------------------------------------------------------------------

_VOWELS = set("aeiou")


def _looks_wordlike(word: str) -> bool:
    lw = word.lower()
    if len(lw) <= 2:
        return True
    if not any(c in _VOWELS for c in lw):
        return False
    # No single cluster of >4 consonants in a row.
    run, max_run = 0, 0
    for c in lw:
        if c in string.ascii_lowercase:
            if c not in _VOWELS:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
    return max_run <= 4


def _max_ngram_repeat(text: str, *, n: int = 5) -> int:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    if len(tokens) < n:
        return 0
    counts: Counter = Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )
    return max(counts.values()) if counts else 0


def _ngram_set(text: str, *, n: int = 5) -> set:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    if len(tokens) < n:
        return set(tuple(tokens),) if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _ngram_jaccard(a: str, b: str, *, n: int = 5) -> float:
    sa, sb = _ngram_set(a, n=n), _ngram_set(b, n=n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union)


def _normalized_edit_distance(a: str, b: str) -> float:
    """
    Normalized Levenshtein distance in [0, 1] (1 = totally different).
    Implemented iteratively so we don't pull in external deps.
    """
    a = a.strip().lower()
    b = b.strip().lower()
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    if a == b:
        return 0.0

    m, n = len(a), len(b)
    # Cap the comparison at 400 chars so long blurbs don't blow up CPU.
    if m > 400 or n > 400:
        a = a[:400]
        b = b[:400]
        m, n = len(a), len(b)

    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, cur = cur, prev

    return prev[n] / max(m, n)


__all__ = [
    "MIN_AD_COPY_CHARS",
    "MIN_WORDLIKE_RATIO",
    "NON_ASCII_MAX_RATIO",
    "PATTERN_NOVELTY_EDITDIST_FLAG",
    "PATTERN_NOVELTY_JACCARD_FLAG",
    "PATTERN_NOVELTY_MIN_QUEUE",
    "REPEATED_NGRAM_MAX",
    "intrinsic_consistency_check",
    "parameter_grounding_check",
    "pattern_novelty_check",
    "real_world_existence_check",
    "run_track_b",
    "signal_realism_check",
]
