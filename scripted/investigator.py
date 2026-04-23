"""
Scripted Investigator policy (rule-based baseline).

Consumes an `AdReviewObservation` dict and returns an `AdReviewAction`.

Strategy (deterministic):
    1. For each new ad_id, pull `landing_page` investigation once.
    2. Based on the resulting feedback + current ad info, issue a verdict:
         - suspicious signals ⇒ reject (confidence 0.85)
         - obviously legit     ⇒ approve (confidence 0.8)
         - ambiguous           ⇒ escalate (confidence 0.5)
    3. Never investigate the same ad twice; keeps budget efficient.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Set

from ..data.meta_policy_taxonomy import citation_blurb_for, is_legit_category
from ..models import AdReviewAction
from ._base import PolicyBase


_SUSPICIOUS_COPY_MARKERS = (
    "free iphone",
    "guaranteed",
    "one weird trick",
    "miracle",
    "lose 20 pounds",
    "50x return",
    "risk-free",
    "giveaway ends",
    "first 500 customers",
    "doctors hate",
    "mooncoin",
    "pre-sale bonus",
)
_SUSPICIOUS_CATEGORY_PREFIXES = ("fake_", "gray_area_")
_LEGIT_COPY_MARKERS = (
    "14-day free trial",
    "free shipping",
    "warranty",
    "return policy",
    "refurbished",
    "certified",
    "office hours",
    "insured",
    "verified",
)


class ScriptedInvestigator(PolicyBase):
    """Heuristic investigator. Investigates each new ad once, then verdicts."""

    def __init__(self) -> None:
        self._investigated: Set[str] = set()
        self._verdicted: Set[str] = set()

    def reset(self) -> None:
        self._investigated.clear()
        self._verdicted.clear()

    def act(self, observation: Dict[str, Any]) -> AdReviewAction:
        available = [
            ad for ad in observation.get("available_ads", []) if ad not in self._verdicted
        ]
        if not available:
            return AdReviewAction(
                action_type="verdict",
                ad_id="__noop__",
                verdict="escalate",
                confidence=0.3,
            )
        ad_id = available[0]

        if ad_id not in self._investigated:
            self._investigated.add(ad_id)
            return AdReviewAction(
                action_type="investigate",
                ad_id=ad_id,
                investigation_target="landing_page",
                rationale=(
                    "ScriptedInvestigator: pulling landing_page for "
                    f"{ad_id} before verdict."
                ),
            )

        verdict, confidence, why, category = self._decide_verdict(ad_id, observation)
        self._verdicted.add(ad_id)

        rationale = (
            f"ScriptedInvestigator: {verdict} {ad_id} (conf={confidence:.2f}). {why}"
        )
        if verdict == "reject" and not is_legit_category(category):
            rationale = f"{rationale} {citation_blurb_for(category)}"

        return AdReviewAction(
            action_type="verdict",
            ad_id=ad_id,
            verdict=verdict,
            confidence=confidence,
            rationale=rationale,
        )

    def _decide_verdict(self, ad_id: str, observation: Dict[str, Any]):
        findings = (observation.get("investigation_findings") or "").lower()
        ad_info = (observation.get("current_ad_info") or "").lower()
        queue_meta = observation.get("queue_status") or {}
        category = self._extract_category(ad_info, queue_meta, ad_id)

        suspicious = False
        legit = False
        reasons = []

        if any(marker in findings for marker in _SUSPICIOUS_COPY_MARKERS):
            suspicious = True
            reasons.append("landing-page copy contains fraud markers")
        if any(marker in ad_info for marker in _SUSPICIOUS_COPY_MARKERS):
            suspicious = True
            reasons.append("ad copy contains fraud markers")
        if category and any(category.startswith(p) for p in _SUSPICIOUS_CATEGORY_PREFIXES):
            suspicious = True
            reasons.append(f"category={category} is fraud template")
        if "domain age" in findings and "<1 year" in findings:
            suspicious = True
            reasons.append("new domain (<1y)")
        if any(marker in findings for marker in _LEGIT_COPY_MARKERS):
            legit = True
            reasons.append("landing page shows legit markers")
        if any(marker in ad_info for marker in _LEGIT_COPY_MARKERS):
            legit = True

        if suspicious and not legit:
            return (
                "reject",
                0.85,
                "; ".join(reasons) or "multiple fraud markers",
                category,
            )
        if legit and not suspicious:
            return (
                "approve",
                0.8,
                "; ".join(reasons) or "legit markers present",
                category,
            )
        return "escalate", 0.5, "ambiguous signals; escalating", category

    def _extract_category(
        self, ad_info: str, queue_meta: Dict[str, Any], ad_id: str
    ) -> str:
        m = re.search(r"category:\s*([a-z_]+)", ad_info)
        if m:
            return m.group(1)
        info = queue_meta.get(ad_id) or {}
        return (info.get("category") or "").lower()
