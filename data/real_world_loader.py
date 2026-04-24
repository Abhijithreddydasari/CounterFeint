"""
Read-only loader for the Meta-CIB-modeled holdout dataset.

This module is the **only** sanctioned import surface for
``counterfeint/data/real_world_test_set.json``. It exists to enforce the
core constraint of the eval lane:

    The synthetic-but-CIB-grounded ads in real_world_test_set.json are a
    HOLDOUT set. They MUST NEVER be used in training rollouts.

To keep that boundary visible at the import level, the loader functions
take an explicit ``confirm_eval_only=True`` argument. Any caller passing
``False`` (or omitting it) gets a :class:`PermissionError`. Training
code paths simply never call this module, so there is no realistic way
to leak holdout data into the training distribution by accident.

Returned ads conform to the existing :class:`counterfeint.data.ad_generator.Ad`
dataclass shape, so the eval lane can drop them straight into the
existing observation builder.

Cross-references
----------------

* Per-ad fields ``case_study_source``, ``provenance_quarter``, and
  ``ring_membership`` mirror the shape used by
  :class:`counterfeint.data.network_generator.FraudRing` and align with
  the three CIB topologies named in
  :data:`counterfeint.data.network_generator.RING_CASE_STUDIES`
  (Ghana DigitSol, Benin Digited, China-Russia hub).
* The :func:`count_by_ring` summary helper feeds the README "Evaluated
  against Meta-CIB-modeled ads" section.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .ad_generator import Ad
except ImportError:  # pragma: no cover - script-level fallback
    from counterfeint.data.ad_generator import Ad  # type: ignore[no-redef]


HOLDOUT_PATH: Path = Path(__file__).resolve().parent / "real_world_test_set.json"


# ---------------------------------------------------------------------------
# Extended Ad row carrying provenance
# ---------------------------------------------------------------------------


@dataclass
class HoldoutAd:
    """Wraps an :class:`Ad` with the CIB provenance fields the eval lane uses.

    The wrapped :class:`Ad` is exposed via :attr:`ad` so existing observation
    builders can consume it without changes; the new fields live on the
    wrapper so they're never accidentally written back into the procedural
    generator's state.
    """

    ad: Ad
    case_study_source: str
    provenance_quarter: str
    ring_membership: Optional[str]
    shared_signals: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ad_id": self.ad.ad_id,
            "ad_copy": self.ad.ad_copy,
            "category": self.ad.category,
            "targeting_summary": self.ad.targeting_summary,
            "initial_risk_signals": list(self.ad.initial_risk_signals),
            "ground_truth_label": self.ad.ground_truth_label,
            "fraud_type": self.ad.fraud_type,
            "severity": self.ad.severity,
            "difficulty": self.ad.difficulty,
            "case_study_source": self.case_study_source,
            "provenance_quarter": self.provenance_quarter,
            "ring_membership": self.ring_membership,
            "shared_signals": dict(self.shared_signals),
        }


# ---------------------------------------------------------------------------
# Public loader API
# ---------------------------------------------------------------------------


class HoldoutAccessError(PermissionError):
    """Raised when the holdout dataset is requested without an eval-only confirmation."""


@lru_cache(maxsize=1)
def _read_raw(path_str: str) -> Dict[str, Any]:
    """Cached JSON read so the test suite doesn't re-parse on every call."""
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def _coerce_ad(raw: Dict[str, Any]) -> HoldoutAd:
    ad = Ad(
        ad_id=str(raw["ad_id"]),
        ad_copy=str(raw["ad_copy"]),
        category=str(raw["category"]),
        targeting_summary=str(raw["targeting_summary"]),
        initial_risk_signals=list(raw.get("initial_risk_signals") or []),
        ground_truth_label=str(raw["ground_truth_label"]),
        fraud_type=str(raw.get("fraud_type") or ""),
        severity=float(raw.get("severity") or 0.0),
        difficulty=str(raw.get("difficulty") or "medium"),
    )
    return HoldoutAd(
        ad=ad,
        case_study_source=str(raw.get("case_study_source") or ""),
        provenance_quarter=str(raw.get("provenance_quarter") or ""),
        ring_membership=raw.get("ring_membership"),
        shared_signals=dict(raw.get("shared_signals") or {}),
    )


def load_real_world_holdout(
    *,
    confirm_eval_only: bool = False,
    path: Path = HOLDOUT_PATH,
) -> List[HoldoutAd]:
    """Load the full Meta-CIB-modeled holdout set.

    Parameters
    ----------
    confirm_eval_only
        Must be set to ``True`` by every caller. Acts as a one-line
        opt-in declaration that the loaded data is going to the eval
        lane, not into a training rollout.
    path
        Override the JSON path (used only by tests).

    Raises
    ------
    HoldoutAccessError
        If ``confirm_eval_only`` is not explicitly ``True``.
    """
    if confirm_eval_only is not True:
        raise HoldoutAccessError(
            "real_world_test_set.json is HOLDOUT data. Pass "
            "`confirm_eval_only=True` to acknowledge that the loaded ads "
            "will not be used in training rollouts (see eval_suite.py)."
        )
    raw = _read_raw(str(path))
    return [_coerce_ad(entry) for entry in raw.get("ads", [])]


def load_for_ring(
    case_study_source: str,
    *,
    confirm_eval_only: bool = False,
) -> List[HoldoutAd]:
    """Filter the holdout to a single CIB case study.

    Useful for the demo when a judge asks "show me the China-Russia
    examples" — pass ``"China-Russia-style hub"``.
    """
    return [
        h
        for h in load_real_world_holdout(confirm_eval_only=confirm_eval_only)
        if h.case_study_source == case_study_source
    ]


# ---------------------------------------------------------------------------
# Summary helpers (no opt-in — they only report counts)
# ---------------------------------------------------------------------------


def count_by_ring(path: Path = HOLDOUT_PATH) -> Dict[str, int]:
    """Return ``{case_study_source: count}`` so the README/UI can render summaries.

    Counts are derived from the on-disk JSON without producing any
    actual ad text, so this helper is safe to call from any context
    (including training-time logging).
    """
    raw = _read_raw(str(path))
    out: Dict[str, int] = {}
    for entry in raw.get("ads", []):
        key = entry.get("case_study_source", "unknown")
        out[key] = out.get(key, 0) + 1
    return out


def list_case_studies(path: Path = HOLDOUT_PATH) -> List[str]:
    """Distinct, stably-ordered list of case study labels in the holdout."""
    raw = _read_raw(str(path))
    seen: List[str] = []
    for entry in raw.get("ads", []):
        label = entry.get("case_study_source", "")
        if label and label not in seen:
            seen.append(label)
    return seen
