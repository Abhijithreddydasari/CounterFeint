"""
Static reference tables for Track B plausibility audit.

These are deliberately **small, hand-curated lookups** rather than scraped
from the web — they run fully offline at judging time.  The tables are
tuned against CounterFeint's R1 synthetic data (see `fraud_patterns.py`,
`advertiser_profiles.py`, `landing_pages.py`) so a realistic R1-generated
fraud ad should *not* trip them, while obviously absurd / gibberish ads
clearly will.
"""

from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Set

# -----------------------------------------------------------------------------
# Country ↔ TLD plausibility.
#
# Map ISO country codes to the set of TLDs that are "plausible" (a common
# ccTLD plus the gTLDs anyone uses).  Ads claiming a US advertiser with a
# `.cn` landing page in a fake-crypto category is classic fraudster-
# signal-mismatch.
# -----------------------------------------------------------------------------

_GLOBAL_TLDS: FrozenSet[str] = frozenset(
    {
        "com",
        "net",
        "org",
        "io",
        "co",
        "shop",
        "store",
        "xyz",
        "online",
        "site",
    }
)

VALID_COUNTRY_TLD_PAIRS: Dict[str, FrozenSet[str]] = {
    "US": _GLOBAL_TLDS | frozenset({"us"}),
    "UK": _GLOBAL_TLDS | frozenset({"uk", "co.uk"}),
    "GB": _GLOBAL_TLDS | frozenset({"uk", "co.uk"}),
    "DE": _GLOBAL_TLDS | frozenset({"de"}),
    "FR": _GLOBAL_TLDS | frozenset({"fr"}),
    "ES": _GLOBAL_TLDS | frozenset({"es"}),
    "IT": _GLOBAL_TLDS | frozenset({"it"}),
    "NL": _GLOBAL_TLDS | frozenset({"nl"}),
    "CA": _GLOBAL_TLDS | frozenset({"ca"}),
    "AU": _GLOBAL_TLDS | frozenset({"au", "com.au"}),
    "IN": _GLOBAL_TLDS | frozenset({"in"}),
    "JP": _GLOBAL_TLDS | frozenset({"jp"}),
    "CN": _GLOBAL_TLDS | frozenset({"cn", "com.cn"}),
    "RU": _GLOBAL_TLDS | frozenset({"ru"}),
    "NG": _GLOBAL_TLDS | frozenset({"ng"}),
    "BR": _GLOBAL_TLDS | frozenset({"br", "com.br"}),
    "MX": _GLOBAL_TLDS | frozenset({"mx"}),
}

# TLDs that should make us suspicious whenever they co-occur with a
# Western advertiser country in a financial / crypto / health category.
_HIGH_RISK_TLDS: FrozenSet[str] = frozenset({"cn", "ru", "tk", "ml", "ga", "cf", "xyz"})


# -----------------------------------------------------------------------------
# Category ↔ targeting compatibility.
#
# Each category has a list of *token* substrings we expect to appear in
# plausible targeting strings.  E.g. weight-loss targeting kids is an
# obvious parameter mismatch.  Lookups are lower-cased substring `in`
# checks so any reasonable phrasing matches.
# -----------------------------------------------------------------------------

CATEGORY_TARGETING_COMPATIBILITY: Dict[str, List[str]] = {
    "ecommerce": [
        "adults",
        "shoppers",
        "shopping",
        "fashion",
        "home",
        "kitchen",
        "beauty",
        "gift",
    ],
    "saas": [
        "adults",
        "professionals",
        "business",
        "developers",
        "technology",
        "it ",
        "b2b",
    ],
    "local_service": [
        "local",
        "homeowners",
        "neighborhood",
        "residents",
        "adults",
    ],
    "education": [
        "students",
        "learners",
        "adults",
        "teachers",
        "parents",
        "kids ",  # note trailing space so we don't match "kidsafe"
    ],
    "fitness": [
        "adults",
        "fitness",
        "athletes",
        "gym",
        "workout",
        "health",
    ],
    "fake_giveaway": [
        "adults",
        "18+",
        "sweepstakes",
        "rewards",
        "gift",
    ],
    "counterfeit_goods": [
        "shoppers",
        "fashion",
        "adults",
        "deals",
    ],
    "miracle_cure": [
        "adults",
        "health",
        "wellness",
        "weight loss",
        "senior",
    ],
    "advance_fee": [
        "adults",
        "finance",
        "investing",
        "entrepreneurs",
    ],
    "fake_crypto": [
        "adults",
        "crypto",
        "investing",
        "finance",
    ],
    "celebrity_endorsement_fraud": [
        "adults",
        "fans",
        "investing",
        "lifestyle",
    ],
    "clone_brand": [
        "shoppers",
        "fashion",
        "adults",
        "bargain",
    ],
    "gray_area_supplements": [
        "adults",
        "wellness",
        "fitness",
        "health",
    ],
    "network_crypto": [
        "adults",
        "crypto",
        "investing",
        "finance",
    ],
    "network_ecommerce": [
        "adults",
        "shoppers",
        "shopping",
    ],
    "network_fintech": [
        "adults",
        "finance",
        "investing",
        "business",
    ],
    "network_health": [
        "adults",
        "health",
        "wellness",
    ],
}

# Categories we *never* want to see targeting minors.
_ADULT_ONLY_CATEGORIES: FrozenSet[str] = frozenset(
    {
        "fake_giveaway",
        "miracle_cure",
        "advance_fee",
        "fake_crypto",
        "celebrity_endorsement_fraud",
        "gray_area_supplements",
        "network_crypto",
        "network_fintech",
    }
)

# Explicit "minor" targeting signals that should trigger a flag for adult-only cats.
_MINOR_TARGETING_MARKERS: FrozenSet[str] = frozenset(
    {
        "kids",
        "teens",
        "teenagers",
        "children",
        "minors",
        "under 18",
        "under-18",
        "schoolers",
    }
)


# -----------------------------------------------------------------------------
# Known fraud markers — patterns the R1 environment uses to *signal* fraud to
# the Investigator.  A Fraudster copying these verbatim into its ad_copy /
# landing_page_blurb is "grader-hacking" rather than generating plausible
# adversarial surface text.
# -----------------------------------------------------------------------------

KNOWN_FRAUD_MARKER_REGEXES: List[re.Pattern[str]] = [
    re.compile(r"\btmpl_\d{3,}\b", re.IGNORECASE),
    re.compile(r"\bpmt_ring_\d{3,}\b", re.IGNORECASE),
    re.compile(r"\bring_[a-z0-9]{3,}\b", re.IGNORECASE),
    re.compile(r"\bcfg_[a-f0-9]{3,}\b", re.IGNORECASE),
    re.compile(r"\badv_[a-z0-9]{3,}_fraud\b", re.IGNORECASE),
]


# -----------------------------------------------------------------------------
# Deny-list tokens — obvious "adversarial" / prompt-injection markers and
# control characters.  If the Fraudster tries to emit these it has clearly
# abandoned the "pretend to be a real advertiser" pretence.
# -----------------------------------------------------------------------------

DENY_LIST_TOKENS: FrozenSet[str] = frozenset(
    {
        "<bypass>",
        "<adversarial>",
        "<jailbreak>",
        "<system>",
        "<<</>>",
        "</instructions>",
        "ignore previous",
        "ignore all previous",
        "disregard instructions",
        "disregard previous",
    }
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def extract_tlds_from_text(text: str) -> Set[str]:
    """Lower-case TLDs found as domain suffixes in free text."""
    if not text:
        return set()
    out: Set[str] = set()
    for match in re.findall(
        r"\b[a-z0-9][a-z0-9\-]*\.([a-z]{2,6}(?:\.[a-z]{2})?)\b",
        text,
        re.IGNORECASE,
    ):
        out.add(match.lower())
    return out


def is_tld_plausible_for_country(country: str, tld: str) -> bool:
    country = (country or "").upper()
    tld = (tld or "").lower().lstrip(".")
    if not tld:
        return True
    allowed = VALID_COUNTRY_TLD_PAIRS.get(country)
    if allowed is None:
        return True  # unknown country, can't flag
    return tld in allowed


def is_high_risk_tld(tld: str) -> bool:
    return (tld or "").lower().lstrip(".") in _HIGH_RISK_TLDS


def is_adult_only_category(category: str) -> bool:
    return (category or "").lower() in _ADULT_ONLY_CATEGORIES


def targeting_mentions_minors(targeting: str) -> bool:
    if not targeting:
        return False
    lowered = targeting.lower()
    return any(marker in lowered for marker in _MINOR_TARGETING_MARKERS)


def targeting_matches_category(category: str, targeting: str) -> bool:
    """
    Soft compatibility check: True if the targeting string contains ≥1
    category-appropriate keyword, or if the category has no configured
    expectations (unknown category → don't flag).
    """
    expected = CATEGORY_TARGETING_COMPATIBILITY.get((category or "").lower())
    if expected is None:
        return True
    if not targeting:
        return False
    lowered = targeting.lower()
    return any(tok in lowered for tok in expected)


def contains_fraud_marker(text: str) -> bool:
    if not text:
        return False
    return any(rx.search(text) for rx in KNOWN_FRAUD_MARKER_REGEXES)


def contains_deny_token(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(tok in lowered for tok in DENY_LIST_TOKENS)


__all__ = [
    "CATEGORY_TARGETING_COMPATIBILITY",
    "DENY_LIST_TOKENS",
    "KNOWN_FRAUD_MARKER_REGEXES",
    "VALID_COUNTRY_TLD_PAIRS",
    "contains_deny_token",
    "contains_fraud_marker",
    "extract_tlds_from_text",
    "is_adult_only_category",
    "is_high_risk_tld",
    "is_tld_plausible_for_country",
    "targeting_matches_category",
    "targeting_mentions_minors",
]
