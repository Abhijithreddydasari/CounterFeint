"""
Investigation tool registry.

Owns the canonical list of investigation targets and the per-(ad_id, target)
findings text.  Lets the Investigator look up findings without caring about
where the data came from (synthetic episode vs. dynamically Fraudster-proposed
ad), and lets the Referee register data for new ads on the fly.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple


# Canonical investigation targets.  Mirrors the Literal in models.py so the
# two sources of truth can be kept in lockstep via the unit tests.
INVESTIGATION_TARGETS: Tuple[str, ...] = (
    "advertiser_history",
    "landing_page",
    "payment_method",
    "targeting_overlap",
    "creative_similarity",
    "campaign_structure",
)


class InvestigationToolRegistry:
    """
    Lookup table mapping (ad_id, investigation_target) -> findings text.

    Wraps a dict[ad_id -> dict[target -> text]] so the Referee can hot-add
    Fraudster proposals.
    """

    def __init__(
        self, base_data: Optional[Dict[str, Dict[str, str]]] = None
    ) -> None:
        self._data: Dict[str, Dict[str, str]] = {
            ad_id: dict(per_ad) for ad_id, per_ad in (base_data or {}).items()
        }

    @classmethod
    def from_episode(cls, episode) -> "InvestigationToolRegistry":  # noqa: ANN001
        """Build a registry from a `GeneratedEpisode.investigation_data` dict."""
        return cls(base_data=getattr(episode, "investigation_data", {}))

    @property
    def targets(self) -> Tuple[str, ...]:
        """Canonical investigation target names."""
        return INVESTIGATION_TARGETS

    def has_ad(self, ad_id: str) -> bool:
        return ad_id in self._data

    def known_ads(self) -> Iterable[str]:
        return self._data.keys()

    def lookup(self, ad_id: str, target: str) -> str:
        """
        Return the findings text for an investigation.

        Returns a sentinel string if either the target name is unknown or no
        data is registered for the (ad_id, target) pair.
        """
        if target not in INVESTIGATION_TARGETS:
            return f"Unknown investigation target '{target}'."
        per_ad = self._data.get(ad_id)
        if per_ad is None:
            return "No data available for this ad."
        return per_ad.get(target, "No data available for this investigation type.")

    def register_ad(self, ad_id: str, findings: Dict[str, str]) -> None:
        """Replace (or add) the entire findings dict for `ad_id`."""
        self._data[ad_id] = dict(findings)

    def update_ad(self, ad_id: str, findings: Dict[str, str]) -> None:
        """Merge new findings into an existing (or new) ad's dict."""
        self._data.setdefault(ad_id, {}).update(findings)

    def remove_ad(self, ad_id: str) -> None:
        self._data.pop(ad_id, None)

    def to_dict(self) -> Dict[str, Dict[str, str]]:
        """Return a deep copy suitable for the Auditor's `investigation_data_seen` field."""
        return {
            ad_id: dict(per_ad) for ad_id, per_ad in self._data.items()
        }
