"""
Meta policy taxonomy layer for CounterFeint.

Maps each internal CounterFeint ad category (e.g. ``fake_giveaway``,
``counterfeit_goods``, ``network_crypto``) to the Meta-published policy section
it violates — section + subsection names are pulled verbatim from Meta's public
transparency site, citation IDs are short synthesized mnemonics used for
inline rationale formatting in agent output.

Sources (all public):

- Fraud, Scams and Deceptive Practices
  https://transparency.meta.com/policies/community-standards/fraud-scams/
- Advertising Standards
  https://transparency.meta.com/en-us/policies/ad-standards/
- Intellectual Property
  https://transparency.meta.com/policies/community-standards/intellectual-property/
- Coordinating Harm and Publicizing Crime (inauthentic coordinated behaviour)
  https://transparency.meta.com/policies/community-standards/coordinating-harm-publicizing-crime/

Used by:

- :mod:`counterfeint.server.environment` — injects ``meta_policy_citation`` into
  ``AdReviewObservation.current_ad_info`` so the Investigator LLM sees the
  exact Meta-grounded policy context for the ad in focus.
- :mod:`counterfeint.scripted.investigator` — scripted Investigator's reject
  rationales cite the appropriate policy ID.
- :mod:`counterfeint.graders.auditor_track_a` — the coherence audit rewards
  rationales that reference the expected citation for obvious-fraud categories.
- ``counterfeint/server/static/investigate_hq.html`` — displays the citation in
  the ad detail panel (judges-visible).

The ``LEGIT_CITATION_ID`` value (``"N/A"``) is returned for benign categories
(``ecommerce``, ``saas``, etc.); this keeps the observation payload shape
uniform and makes it easy for agents to distinguish "no violation claimed" from
"we forgot to look it up".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


LEGIT_CITATION_ID = "N/A"


@dataclass(frozen=True)
class MetaPolicyEntry:
    """Frozen description of one Meta-published policy lens on an ad category."""

    section: str
    """Top-level Meta policy section (e.g. 'Fraud, Scams and Deceptive Practices')."""

    subsection: str
    """Named subsection within the section (e.g. 'Inauthentic Identity - Fake Giveaways')."""

    citation_id: str
    """Short mnemonic used in inline rationales (e.g. 'FSDP-IF-03')."""

    url: str
    """Public transparency.meta.com URL for the policy section."""

    def citation_blurb(self) -> str:
        """One-line citation ready to paste into a rationale or UI card."""
        if self.citation_id == LEGIT_CITATION_ID:
            return "No Meta policy violation claimed (legitimate ad category)."
        return (
            f"Meta policy citation: {self.citation_id} — "
            f"{self.section} > {self.subsection}"
        )


FRAUD_URL = (
    "https://transparency.meta.com/policies/community-standards/fraud-scams/"
)
AD_STANDARDS_URL = "https://transparency.meta.com/en-us/policies/ad-standards/"
IP_URL = (
    "https://transparency.meta.com/policies/community-standards/"
    "intellectual-property/"
)
CIB_URL = (
    "https://transparency.meta.com/policies/community-standards/"
    "coordinating-harm-publicizing-crime/"
)


_LEGIT = MetaPolicyEntry(
    section="N/A",
    subsection="Legitimate ad",
    citation_id=LEGIT_CITATION_ID,
    url="",
)


META_TAXONOMY: Dict[str, MetaPolicyEntry] = {
    # Legit camouflage categories
    "ecommerce": _LEGIT,
    "saas": _LEGIT,
    "local_service": _LEGIT,
    "education": _LEGIT,
    "fitness": _LEGIT,
    # Fraud / deceptive practice templates
    "fake_giveaway": MetaPolicyEntry(
        section="Fraud, Scams and Deceptive Practices",
        subsection="Inauthentic Identity - Fake Giveaways",
        citation_id="FSDP-IF-03",
        url=FRAUD_URL,
    ),
    "counterfeit_goods": MetaPolicyEntry(
        section="Intellectual Property",
        subsection="Counterfeit Goods",
        citation_id="IP-CG-01",
        url=IP_URL,
    ),
    "miracle_cure": MetaPolicyEntry(
        section="Advertising Standards",
        subsection="Unrealistic Health Claims",
        citation_id="AS-HC-07",
        url=AD_STANDARDS_URL,
    ),
    "advance_fee": MetaPolicyEntry(
        section="Fraud, Scams and Deceptive Practices",
        subsection="Advance-Fee Fraud",
        citation_id="FSDP-AF-02",
        url=FRAUD_URL,
    ),
    "fake_crypto": MetaPolicyEntry(
        section="Advertising Standards",
        subsection="Cryptocurrency Products and Services",
        citation_id="AS-CR-05",
        url=AD_STANDARDS_URL,
    ),
    "celebrity_endorsement_fraud": MetaPolicyEntry(
        section="Fraud, Scams and Deceptive Practices",
        subsection="Impersonation - Celebrity",
        citation_id="FSDP-IP-04",
        url=FRAUD_URL,
    ),
    "clone_brand": MetaPolicyEntry(
        section="Intellectual Property",
        subsection="Trademark Infringement",
        citation_id="IP-TM-02",
        url=IP_URL,
    ),
    "gray_area_supplements": MetaPolicyEntry(
        section="Advertising Standards",
        subsection="Unsafe Supplements and Weight-Loss Products",
        citation_id="AS-SUP-03",
        url=AD_STANDARDS_URL,
    ),
    # Coordinated / ring-level categories used by Task 3 network fraud
    "network_crypto": MetaPolicyEntry(
        section="Coordinating Harm and Publicizing Crime",
        subsection="Coordinated Inauthentic Behaviour - Investment Fraud Ring",
        citation_id="CH-CIB-01",
        url=CIB_URL,
    ),
    "network_ecommerce": MetaPolicyEntry(
        section="Coordinating Harm and Publicizing Crime",
        subsection="Coordinated Inauthentic Behaviour - Commerce Ring",
        citation_id="CH-CIB-02",
        url=CIB_URL,
    ),
    "network_fintech": MetaPolicyEntry(
        section="Coordinating Harm and Publicizing Crime",
        subsection="Coordinated Inauthentic Behaviour - Financial Services Ring",
        citation_id="CH-CIB-03",
        url=CIB_URL,
    ),
    "network_health": MetaPolicyEntry(
        section="Coordinating Harm and Publicizing Crime",
        subsection="Coordinated Inauthentic Behaviour - Health Products Ring",
        citation_id="CH-CIB-04",
        url=CIB_URL,
    ),
}


def lookup(category: Optional[str]) -> MetaPolicyEntry:
    """Return the MetaPolicyEntry for a category (legit-default for unknowns)."""
    if not category:
        return _LEGIT
    return META_TAXONOMY.get(category, _LEGIT)


def citation_id_for(category: Optional[str]) -> str:
    """Convenience shortcut: just the citation_id string."""
    return lookup(category).citation_id


def citation_blurb_for(category: Optional[str]) -> str:
    """Convenience shortcut: one-line citation blurb for a category."""
    return lookup(category).citation_blurb()


def is_legit_category(category: Optional[str]) -> bool:
    """True if the category maps to the legit-placeholder (no policy violation)."""
    return lookup(category).citation_id == LEGIT_CITATION_ID


__all__ = [
    "MetaPolicyEntry",
    "META_TAXONOMY",
    "LEGIT_CITATION_ID",
    "lookup",
    "citation_id_for",
    "citation_blurb_for",
    "is_legit_category",
]
