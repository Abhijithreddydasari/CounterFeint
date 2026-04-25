"""
Per-ad structured evidence ledger.

Both the Investigator and the Fraudster benefit from a structured cross-ad
view of the underlying account / payment / landing-page signals. The
Investigator uses it to decide ``link_accounts`` (collisions on
``payment_id`` or ``targeting_fingerprint`` indicate a ring); the Fraudster
uses it to decide which of its own pending proposals to ``modify_pending_ad``
based on which signals correlate with the Investigator's rejections.

We deliberately mix HIGH-signal columns (payment_id, targeting_fingerprint)
with LOW-signal columns (country, category, account_age_days) so neither
LLM can shortcut to "any collision = ring" — it has to *learn* which
columns are discriminative. That's the "more parameters in the summary so
the model picks the right ones" framing the user asked for.

The ledger is *derived* from already-revealed data only:
  - For the Investigator, a field appears only after the matching
    ``investigate`` target has been pulled on that ad.
  - For the Fraudster, fields are present for every Fraudster-proposed ad
    because the env gates Fraudster proposals through
    ``extend_episode_with_proposal`` which auto-assigns and immediately
    surfaces all underlying signals back to the proposer (the Fraudster
    never sees signals for *Investigator*-side / synthetic ads).

We extract structured fields by regex from the same investigation text the
agents already see — so the ledger and the free-form ``investigation_findings``
stay in lock-step (no extra info is leaked, only re-shaped).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Optional


# Regex extractors keyed by investigation target name. Patterns parse the
# rendered investigation text produced by:
#   - counterfeint/data/ad_generator._generate_payment_investigation
#   - counterfeint/data/landing_pages.LandingPageData.to_investigation_text
#   - counterfeint/data/ad_generator._generate_targeting_investigation
# Keep these in sync if the rendering changes.
_LEDGER_EXTRACTORS: Dict[str, Dict[str, "re.Pattern[str]"]] = {
    "payment_method": {
        "payment_id": re.compile(r"Payment ID:\s*(\S+)"),
        "payment_type": re.compile(r"Method type:\s*(\S+)"),
    },
    "landing_page": {
        "domain": re.compile(r"^Domain:\s*(\S+)", re.MULTILINE),
        "registrar": re.compile(r"Registrar:\s*([^\n]+)"),
        "domain_age_days": re.compile(r"Domain age:\s*(\d+)"),
    },
    "targeting_overlap": {
        "targeting_fingerprint": re.compile(r"Targeting fingerprint:\s*(\S+)"),
    },
    "advertiser_history": {
        # advertiser_id and verified_business are surfaced from the
        # AdvertiserProfile dataclass directly — see _build_entry below.
    },
}


def build_evidence_ledger(
    *,
    episode: Any,
    registry: Optional[Any],
    ad_ids: Iterable[str],
    investigations: Mapping[str, Iterable[str]],
) -> Dict[str, Dict[str, Any]]:
    """Build a {ad_id: {field: value}} ledger over the given ad_ids.

    Parameters
    ----------
    episode
        ``GeneratedEpisode`` providing ``ads``, ``advertiser_profiles`` and
        ``investigation_data``.
    registry
        Optional ``InvestigationToolRegistry`` — preferred source of
        already-rendered investigation text. Falls back to
        ``episode.investigation_data`` if not provided.
    ad_ids
        Which ads to include in the ledger. Caller decides scoping
        (Investigator: all ads it has touched; Fraudster: its own
        proposals).
    investigations
        ``{ad_id: [investigated_target, ...]}``. Determines which extractor
        sets to run per ad.
    """
    if episode is None:
        return {}

    ads_by_id = {ad.ad_id: ad for ad in episode.ads}
    profiles = getattr(episode, "advertiser_profiles", {}) or {}
    inv_data = getattr(episode, "investigation_data", {}) or {}

    ledger: Dict[str, Dict[str, Any]] = {}
    for ad_id in ad_ids:
        entry = _build_entry(
            ad_id=ad_id,
            ad=ads_by_id.get(ad_id),
            profile=profiles.get(ad_id),
            investigated_targets=list(investigations.get(ad_id, []) or []),
            registry=registry,
            inv_data_for_ad=inv_data.get(ad_id, {}) or {},
        )
        if entry:
            ledger[ad_id] = entry
    return ledger


def _build_entry(
    *,
    ad_id: str,
    ad: Any,
    profile: Any,
    investigated_targets: Iterable[str],
    registry: Optional[Any],
    inv_data_for_ad: Mapping[str, str],
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {}

    if ad is not None:
        entry["category"] = ad.category

    if profile is not None:
        entry["country"] = profile.country
        entry["account_age_days"] = profile.account_age_days

    for target in investigated_targets:
        if target == "advertiser_history" and profile is not None:
            entry["advertiser_id"] = profile.advertiser_id
            entry["verified_business"] = bool(profile.verified_business)
            continue

        extractors = _LEDGER_EXTRACTORS.get(target)
        if not extractors:
            continue
        text = ""
        if registry is not None:
            text = registry.lookup(ad_id, target) or ""
        if not text:
            text = inv_data_for_ad.get(target, "") or ""
        for field_name, pattern in extractors.items():
            m = pattern.search(text)
            if m:
                value: Any = m.group(1).strip()
                if field_name == "domain_age_days":
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                entry[field_name] = value

    return entry


__all__ = ["build_evidence_ledger"]
