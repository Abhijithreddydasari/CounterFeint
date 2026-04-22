"""
Static heuristic tables consumed by `graders/auditor_track_a.py`.

These are deliberately **rule-based** (not LLM-judge) so Track A can run
deterministically inside the grader during on-site judging — no API keys,
no non-determinism, no cost.
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Red / green phrases for rationale-coherence checks.
#
# Red phrases strongly suggest the ad is fraudulent; green phrases strongly
# suggest it is legitimate.  A rationale that cites ≥2 red phrases but ends
# with `approve` (or ≥2 green phrases and ends with `reject`) is incoherent.
# -----------------------------------------------------------------------------

RED_FLAG_PHRASES: FrozenSet[str] = frozenset(
    {
        # Landing-page / domain signals
        "no ssl",
        "expired certificate",
        "privacy registrar",
        "privacy proxy",
        "whois privacy",
        "new domain",
        "young domain",
        "scam similarity",
        "known scam template",
        "phishing",
        "redirect chain",
        "suspicious redirect",
        # Advertiser / account signals
        "prior ban",
        "previous ban",
        "policy violation",
        "unverified business",
        "unverified account",
        "new account",
        "young account",
        "burst submission",
        "burst activity",
        "high spend velocity",
        "budget/age ratio",
        "budget to age ratio",
        # Ring / network signals
        "shared payment",
        "shared registrar",
        "shared template",
        "ring member",
        "network overlap",
        "template reuse",
        "near-duplicate",
        "clone template",
        # Category markers
        "miracle cure",
        "guaranteed returns",
        "advance fee",
        "celebrity endorsement fraud",
        "counterfeit",
    }
)

GREEN_FLAG_PHRASES: FrozenSet[str] = frozenset(
    {
        "verified business",
        "valid ssl",
        "ssl valid",
        "clean record",
        "no violations",
        "established domain",
        "aged domain",
        "long tenure",
        "historical approval",
        "high approval rate",
        "steady cadence",
        "consistent activity",
        "legitimate domain",
        "mainstream registrar",
        "reputable registrar",
        "recognized brand",
        "organic growth",
        "typical targeting",
    }
)


# -----------------------------------------------------------------------------
# Regex patterns used by `rationale_citation_audit`.
#
# We scan the Investigator's rationale for *evidence tokens* that actually
# appear in the investigation findings text (e.g. a payment_id like
# "pmt_12345" or a domain like "shady.cn").  A rationale that cites nothing
# is suspect — reasoning should ground in evidence.
# -----------------------------------------------------------------------------

# These are "interesting tokens" we consider citations when they appear in
# both the rationale and the findings text.
PAYMENT_ID_RE = re.compile(r"\bpmt_[a-z0-9_]{4,}\b", re.IGNORECASE)
DOMAIN_RE = re.compile(r"\b[a-z0-9][a-z0-9\-]{1,}\.[a-z]{2,24}\b", re.IGNORECASE)
TEMPLATE_HASH_RE = re.compile(r"\btmpl_[a-z0-9]{3,}\b", re.IGNORECASE)
RING_ID_RE = re.compile(r"\bring_[a-z0-9_]{2,}\b", re.IGNORECASE)
ADVERTISER_ACCT_RE = re.compile(r"\badv_[a-z0-9_]{3,}\b", re.IGNORECASE)
CURRENCY_RE = re.compile(r"\$[\d,]{1,}(?:\.\d{1,2})?")
PCT_RE = re.compile(r"\d{1,3}\s?%")
REGISTRAR_RE = re.compile(
    r"\b(njalla|epik|namesilo|godaddy|cloudflare|tucows)\b", re.IGNORECASE
)

_EVIDENCE_REGEXES: Tuple[re.Pattern[str], ...] = (
    PAYMENT_ID_RE,
    DOMAIN_RE,
    TEMPLATE_HASH_RE,
    RING_ID_RE,
    ADVERTISER_ACCT_RE,
    CURRENCY_RE,
    PCT_RE,
    REGISTRAR_RE,
)


def extract_evidence_tokens(text: str) -> List[str]:
    """Return the list of citation-shaped tokens found in `text` (lower-cased)."""
    if not text:
        return []
    seen: List[str] = []
    for rx in _EVIDENCE_REGEXES:
        for match in rx.findall(text):
            tok = match.lower() if isinstance(match, str) else match
            if tok and tok not in seen:
                seen.append(tok)
    return seen


def count_citations_matching_findings(rationale: str, findings: str) -> int:
    """
    Count tokens in `rationale` that also appear in `findings`.

    Pure word-match over lower-cased evidence tokens — deterministic,
    no LLM, safe to run at judging time.
    """
    if not rationale or not findings:
        return 0
    findings_lower = findings.lower()
    rationale_tokens = extract_evidence_tokens(rationale)
    hits = 0
    for tok in rationale_tokens:
        if tok in findings_lower:
            hits += 1
    # Also count red/green phrase echoes — they are weaker but still "citation-like".
    rationale_lower = rationale.lower()
    for phrase in RED_FLAG_PHRASES | GREEN_FLAG_PHRASES:
        if phrase in rationale_lower and phrase in findings_lower:
            hits += 1
    return hits


def count_phrase_hits(text: str, phrases: Iterable[str]) -> int:
    """Case-insensitive whole-phrase hit count."""
    if not text:
        return 0
    lower = text.lower()
    return sum(1 for phrase in phrases if phrase in lower)


# -----------------------------------------------------------------------------
# Structural pair hashing for cross-ad consistency audit.
#
# Two ads are "structurally similar" if the Investigator *should* treat
# them the same way (because they share a meaningful attribute bundle).
# Diverging verdicts on matched pairs → consistency flag.
# -----------------------------------------------------------------------------


def hash_ad_pair(ad_a: Dict[str, Any], ad_b: Dict[str, Any]) -> Optional[str]:
    """
    Return a stable structural-similarity key if `ad_a` and `ad_b` should
    be treated as equivalent by a consistent Investigator.

    Returns `None` if the ads are not structurally similar.

    Definition of "structurally similar" (rule-of-thumb, tuned to R1 data):
      • Same `ground_truth` (both fraud, both legit, or both escalate)
      • Same `fraud_type` OR same `category`
      • Same `severity` bucket  (low: <0.33, mid: <0.67, high: >=0.67)
    """
    if ad_a.get("ad_id") == ad_b.get("ad_id"):
        return None

    if ad_a.get("ground_truth") != ad_b.get("ground_truth"):
        return None

    fraud_match = (
        (ad_a.get("fraud_type") or "") == (ad_b.get("fraud_type") or "")
        and (ad_a.get("fraud_type") or "") != ""
    )
    cat_match = (
        (ad_a.get("category") or "") == (ad_b.get("category") or "")
        and (ad_a.get("category") or "") != ""
    )
    if not (fraud_match or cat_match):
        return None

    def sev_bucket(x: Any) -> str:
        try:
            s = float(x or 0.0)
        except (TypeError, ValueError):
            return "unk"
        if s < 0.33:
            return "low"
        if s < 0.67:
            return "mid"
        return "high"

    if sev_bucket(ad_a.get("severity")) != sev_bucket(ad_b.get("severity")):
        return None

    key_parts = sorted([ad_a.get("ad_id", ""), ad_b.get("ad_id", "")])
    marker = "F" if fraud_match else "C"
    return (
        f"{marker}|"
        f"{ad_a.get('ground_truth')}|"
        f"{ad_a.get('fraud_type') if fraud_match else ad_a.get('category')}|"
        f"{sev_bucket(ad_a.get('severity'))}|"
        f"{'-'.join(key_parts)}"
    )


# -----------------------------------------------------------------------------
# Slice definitions for bias audit
# -----------------------------------------------------------------------------


def severity_slice(severity: Any) -> str:
    try:
        s = float(severity or 0.0)
    except (TypeError, ValueError):
        return "unknown"
    if s < 0.33:
        return "low"
    if s < 0.67:
        return "mid"
    return "high"


__all__ = [
    "ADVERTISER_ACCT_RE",
    "CURRENCY_RE",
    "DOMAIN_RE",
    "GREEN_FLAG_PHRASES",
    "PAYMENT_ID_RE",
    "PCT_RE",
    "RED_FLAG_PHRASES",
    "REGISTRAR_RE",
    "RING_ID_RE",
    "TEMPLATE_HASH_RE",
    "count_citations_matching_findings",
    "count_phrase_hits",
    "extract_evidence_tokens",
    "hash_ad_pair",
    "severity_slice",
]
