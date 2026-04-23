"""
Llama Guard 3 / Purple Llama mock classifier data.

Mocks Meta's open safety classifier (https://github.com/meta-llama/PurpleLlama)
for ad content.  Used by the Investigator's `policy_classifier` investigation
target and exposed as a REST endpoint at `POST /api/v1/tools/policy_classifier`.

Design goals:

- **Deterministic per ad_id**: same input → same output, for judge reproducibility.
- **Ground-truth correlated**: fraud ads are labelled "unsafe" ~75% of the time;
  legit ads are false-flagged ~5% of the time.  The classifier is noisy on
  purpose — an Investigator that over-trusts it will still make mistakes.
- **Surface-marker grounded**: uses regex heuristics (urgency, fake authority,
  get-rich-quick patterns) modelled after real Meta T&S enforcement signals so
  the explanation text is specific enough for the Investigator's rationale to
  cite.

The classifier is intentionally a mock: it's not loading the real Llama Guard 3
weights (which would blow the container size).  The shape of the output exactly
matches what a real Llama Guard 3-8B call would return so switching to the real
model is a one-function swap.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Llama Guard 3 hazard taxonomy (public categories).
# Source: https://github.com/meta-llama/PurpleLlama/blob/main/Llama-Guard3/8B/MODEL_CARD.md
# --------------------------------------------------------------------------
LLAMA_GUARD_CATEGORIES: Dict[str, str] = {
    "S1": "violent_crimes",
    "S2": "non_violent_crimes",
    "S3": "sex_related_crimes",
    "S4": "child_sexual_exploitation",
    "S5": "defamation",
    "S6": "specialized_advice",
    "S7": "privacy",
    "S8": "intellectual_property",
    "S9": "indiscriminate_weapons",
    "S10": "hate",
    "S11": "suicide_self_harm",
    "S12": "sexual_content",
    "S13": "elections",
}


# --------------------------------------------------------------------------
# CounterFeint-specific "TS-Fraud" head — surface markers a production T&S
# classifier would stack on top of Llama Guard for ad-fraud specifically.
# --------------------------------------------------------------------------
TS_FRAUD_MARKERS: Dict[str, str] = {
    "high_pressure_urgency": "Act-now / limited-time / expires-in manipulation",
    "fake_authority_claim": "Doctor-recommended / FDA-approved / certified (unverified)",
    "unrealistic_guarantee": "100% guaranteed / risk-free / no-questions-asked",
    "exclusivity_manipulation": "Secret / hidden / exclusive-access framing",
    "get_rich_quick": "Earn $X daily / make $X from home",
    "phishing_credential_request": "Login / verify / click-here patterns",
    "counterfeit_indicator": "Replica / dupe / authentic-looking",
}


_URGENCY_RE = re.compile(
    r"\b(act\s+now|limited\s+time|expires?\s+(?:in|soon|today)|"
    r"only\s+\d+\s+(?:left|spots|hours?|days?)|hurry|last\s+chance|"
    r"while\s+supplies?\s+last|ends?\s+(?:tonight|soon))\b",
    re.IGNORECASE,
)
_AUTHORITY_RE = re.compile(
    r"\b(fda[\- ]approved|doctor[\- ]recommended|clinically\s+proven|"
    r"certified(?:\s+by|\s+professional)?|endorsed\s+by|"
    r"board[\- ]certified|medical[\- ]grade)\b",
    re.IGNORECASE,
)
_GUARANTEE_RE = re.compile(
    r"\b(100%?\s*(?:guaranteed|money[\- ]?back)|risk[\- ]free|"
    r"no\s+questions?\s+asked|satisfaction\s+guaranteed|guaranteed\s+results?)\b",
    re.IGNORECASE,
)
_EXCLUSIVITY_RE = re.compile(
    r"\b(secret|hidden|exclusive\s+access|members?\s+only|"
    r"invitation\s+only|insider|behind[\- ]the[\- ]scenes)\b",
    re.IGNORECASE,
)
_GET_RICH_RE = re.compile(
    r"(?:earn|make)\s*\$?\d+[kKmM]?\s*(?:daily|per\s+day|a\s+day|"
    r"weekly|per\s+week|from\s+home|doing\s+nothing)",
    re.IGNORECASE,
)
_PHISHING_RE = re.compile(
    r"\b(verify\s+(?:your|account)|login\s+(?:required|to)|"
    r"click\s+here\s+to\s+(?:claim|verify|unlock)|"
    r"confirm\s+(?:your|account|identity)|password\s+reset)\b",
    re.IGNORECASE,
)
_COUNTERFEIT_RE = re.compile(
    r"\b(replica|dupe|authentic[\- ]looking|copy\s+of|knockoff|"
    r"inspired\s+by\s+(?:the\s+)?original|designer[\- ]inspired)\b",
    re.IGNORECASE,
)


MARKER_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("high_pressure_urgency", _URGENCY_RE),
    ("fake_authority_claim", _AUTHORITY_RE),
    ("unrealistic_guarantee", _GUARANTEE_RE),
    ("exclusivity_manipulation", _EXCLUSIVITY_RE),
    ("get_rich_quick", _GET_RICH_RE),
    ("phishing_credential_request", _PHISHING_RE),
    ("counterfeit_indicator", _COUNTERFEIT_RE),
]


# --------------------------------------------------------------------------
# Fraud-type → Llama Guard category mapping.
# Drives *which* LG category fires when a given fraud type is flagged unsafe.
# Keys match the values produced by `counterfeint/data/fraud_patterns.py`.
# --------------------------------------------------------------------------
FRAUD_TYPE_TO_LG_CATEGORIES: Dict[str, List[str]] = {
    "fake_giveaway": ["S2"],
    "counterfeit": ["S2", "S8"],
    "miracle_cure": ["S2", "S6"],
    "advance_fee_scam": ["S2", "S7"],
    "fake_crypto": ["S2"],
    "fake_endorsement": ["S2", "S5"],
    "brand_impersonation": ["S5", "S8"],
    "gray_area": ["S2"],
    "coordinated_network": ["S2"],
}
_DEFAULT_FRAUD_LG: List[str] = ["S2"]


# Ground-truth probabilities: how reliably the classifier agrees with the
# true label.  Noisy on purpose — the Investigator has to corroborate with
# other tools, not blindly trust policy_classifier.
P_UNSAFE_GIVEN_FRAUD: float = 0.75
P_UNSAFE_GIVEN_LEGIT: float = 0.05


@dataclass
class PolicyClassifierResult:
    """Structured mock Llama Guard 3 output."""

    ad_id: str
    verdict: str  # "unsafe" | "safe"
    confidence: float  # 0.0..1.0
    triggered_lg_categories: List[str] = field(default_factory=list)
    triggered_fraud_markers: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "ad_id": self.ad_id,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "triggered_lg_categories": [
                {"code": code, "name": LLAMA_GUARD_CATEGORIES[code]}
                for code in self.triggered_lg_categories
            ],
            "triggered_fraud_markers": [
                {"code": m, "description": TS_FRAUD_MARKERS[m]}
                for m in self.triggered_fraud_markers
            ],
            "explanation": self.explanation,
        }

    def to_investigation_text(self) -> str:
        """Render as a multi-line block suitable for the Investigator's findings."""
        lines = [f"Llama Guard 3 Classification for {self.ad_id}:"]
        lines.append(f"  Verdict: {self.verdict} (confidence {self.confidence:.2f})")
        if self.triggered_lg_categories:
            cats_str = ", ".join(
                f"{k} ({LLAMA_GUARD_CATEGORIES[k]})"
                for k in self.triggered_lg_categories
            )
            lines.append(f"  Triggered LG categories: {cats_str}")
        else:
            lines.append("  Triggered LG categories: none")
        if self.triggered_fraud_markers:
            markers_str = ", ".join(
                f"TS-Fraud ({m})" for m in self.triggered_fraud_markers
            )
            lines.append(f"  Triggered custom markers: {markers_str}")
        else:
            lines.append("  Triggered custom markers: none")
        lines.append(f"  Policy explanation: {self.explanation}")
        return "\n".join(lines)


def _seeded_rng(ad_id: str, salt: str = "lg3") -> random.Random:
    """Deterministic per-ad RNG seed (so same ad_id → same verdict every time)."""
    h = hashlib.sha256(f"{salt}:{ad_id}".encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def detect_fraud_markers(text: str) -> List[str]:
    """Scan ad_copy + landing_page text for TS-Fraud markers (regex-based)."""
    matches: List[str] = []
    for name, pattern in MARKER_PATTERNS:
        if pattern.search(text):
            matches.append(name)
    return matches


def classify_ad(
    ad_id: str,
    ad_copy: str,
    landing_page_text: str = "",
    ground_truth_label: Optional[str] = None,
    fraud_type: Optional[str] = None,
) -> PolicyClassifierResult:
    """Deterministic mock Llama Guard 3 classification.

    Parameters
    ----------
    ad_id :
        Identifier — used as RNG seed so same ad always yields the same output.
    ad_copy :
        Ad body text (scanned for surface markers).
    landing_page_text :
        Optional landing page blurb, also scanned for surface markers.
    ground_truth_label :
        "fraud" | "legit" | "escalate" | None.  If provided (internal
        episode generation), biases the verdict toward the label with the
        P_UNSAFE_GIVEN_* noise rates.  If None (e.g. external REST endpoint
        call without a label), falls back to surface-marker heuristic.
    fraud_type :
        Optional.  Drives *which* LG category is triggered when the verdict
        is "unsafe" for a ground-truth fraud ad.
    """
    rng = _seeded_rng(ad_id)
    combined = f"{ad_copy}\n{landing_page_text}"
    surface_markers = detect_fraud_markers(combined)

    if ground_truth_label == "fraud":
        unsafe = rng.random() < P_UNSAFE_GIVEN_FRAUD
    elif ground_truth_label == "legit":
        unsafe = rng.random() < P_UNSAFE_GIVEN_LEGIT
    elif ground_truth_label == "escalate":
        # Escalate is genuinely ambiguous → 50/50 with slight skew to unsafe
        # if surface markers exist.
        base = 0.35 + 0.15 * min(len(surface_markers), 3)
        unsafe = rng.random() < base
    else:
        # No ground-truth hint (public API / external curl).  Relies purely on
        # surface markers: 2+ → unsafe; 1 → coin flip biased unsafe; 0 → safe.
        if len(surface_markers) >= 2:
            unsafe = True
        elif len(surface_markers) == 1:
            unsafe = rng.random() < 0.35
        else:
            unsafe = False

    if unsafe:
        if ground_truth_label == "fraud" and fraud_type:
            lg_cats = list(
                FRAUD_TYPE_TO_LG_CATEGORIES.get(fraud_type, _DEFAULT_FRAUD_LG)
            )
        else:
            # Infer from surface markers.
            lg_cats = ["S2"]
            if "phishing_credential_request" in surface_markers:
                lg_cats.append("S7")
            if "counterfeit_indicator" in surface_markers:
                lg_cats.append("S8")
            # Dedup while preserving order.
            seen = set()
            lg_cats = [c for c in lg_cats if not (c in seen or seen.add(c))]

        triggered_markers = list(surface_markers)
        if not triggered_markers and ground_truth_label == "fraud":
            # Latent pattern match — classifier fires on embeddings even with
            # no surface regex hits.  Pick a random marker for readability.
            triggered_markers = [rng.choice(list(TS_FRAUD_MARKERS.keys()))]

        confidence = rng.uniform(0.68, 0.92)
        primary_label = LLAMA_GUARD_CATEGORIES[lg_cats[0]]
        markers_str = (
            ", ".join(triggered_markers) if triggered_markers else "latent embedding match"
        )
        explanation = (
            f"Model flags {primary_label} pattern at confidence {confidence:.2f}. "
            f"Signals: {markers_str}."
        )
        return PolicyClassifierResult(
            ad_id=ad_id,
            verdict="unsafe",
            confidence=confidence,
            triggered_lg_categories=lg_cats,
            triggered_fraud_markers=triggered_markers,
            explanation=explanation,
        )
    else:
        confidence = rng.uniform(0.55, 0.90)
        markers_str = (
            ", ".join(surface_markers) if surface_markers else "none"
        )
        explanation = (
            f"No high-confidence policy violations. Surface signals: {markers_str}."
        )
        return PolicyClassifierResult(
            ad_id=ad_id,
            verdict="safe",
            confidence=confidence,
            triggered_lg_categories=[],
            triggered_fraud_markers=list(surface_markers),
            explanation=explanation,
        )
