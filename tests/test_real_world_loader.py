"""Tests for counterfeint.data.real_world_loader.

Validates the holdout shape AND the eval-only opt-in guard. The latter
is the single most important contract for this module: if anyone can
import the holdout into training without an explicit confirmation,
the "before / after on Meta-CIB-modeled ads" claim collapses.
"""

from __future__ import annotations

import pytest

from counterfeint.data.network_generator import RING_CASE_STUDIES
from counterfeint.data.real_world_loader import (
    HoldoutAccessError,
    HoldoutAd,
    count_by_ring,
    list_case_studies,
    load_for_ring,
    load_real_world_holdout,
)


class TestEvalOnlyGuard:
    def test_default_call_raises(self) -> None:
        with pytest.raises(HoldoutAccessError):
            load_real_world_holdout()

    def test_explicit_false_raises(self) -> None:
        with pytest.raises(HoldoutAccessError):
            load_real_world_holdout(confirm_eval_only=False)

    def test_truthy_non_true_value_still_raises(self) -> None:
        # Force callers to type the literal True; "yes", 1, etc. don't pass.
        with pytest.raises(HoldoutAccessError):
            load_real_world_holdout(confirm_eval_only=1)  # type: ignore[arg-type]

    def test_explicit_true_succeeds(self) -> None:
        ads = load_real_world_holdout(confirm_eval_only=True)
        assert len(ads) > 0


class TestHoldoutShape:
    @pytest.fixture(scope="class")
    def ads(self) -> list[HoldoutAd]:
        return load_real_world_holdout(confirm_eval_only=True)

    def test_has_15_entries(self, ads: list[HoldoutAd]) -> None:
        assert len(ads) == 15

    def test_every_entry_has_required_fields(self, ads: list[HoldoutAd]) -> None:
        for h in ads:
            assert h.ad.ad_id
            assert h.ad.ad_copy
            assert h.ad.category
            assert h.ad.ground_truth_label in {"fraud", "legit", "escalate"}
            assert 0.0 <= h.ad.severity <= 1.0
            assert h.case_study_source
            assert h.provenance_quarter

    def test_ad_ids_unique(self, ads: list[HoldoutAd]) -> None:
        ids = [h.ad.ad_id for h in ads]
        assert len(ids) == len(set(ids))

    def test_to_dict_round_trips_provenance(self, ads: list[HoldoutAd]) -> None:
        for h in ads:
            d = h.to_dict()
            assert d["case_study_source"] == h.case_study_source
            assert d["provenance_quarter"] == h.provenance_quarter
            assert d["ring_membership"] == h.ring_membership

    def test_distractor_legit_ads_have_no_ring(self, ads: list[HoldoutAd]) -> None:
        legit = [h for h in ads if h.ad.ground_truth_label == "legit"]
        assert legit, "distractor legit ads missing — eval becomes trivial"
        for h in legit:
            assert h.ring_membership is None


class TestCibAlignment:
    def test_every_case_study_aligns_with_named_topology(self) -> None:
        case_names = {cs["case_name"] for cs in RING_CASE_STUDIES}
        observed = set(list_case_studies()) - {
            "Distractor (not part of any CIB ring)",
        }
        assert observed.issubset(case_names), (
            f"Holdout references unknown CIB case names: {observed - case_names}"
        )

    def test_each_named_case_study_has_ads(self) -> None:
        counts = count_by_ring()
        for cs in RING_CASE_STUDIES:
            label = cs["case_name"]
            assert counts.get(label, 0) > 0, (
                f"No holdout ads for CIB case study {label!r}"
            )

    def test_load_for_ring_filters_correctly(self) -> None:
        ghana = load_for_ring("Ghana DigitSol-style", confirm_eval_only=True)
        assert all(h.case_study_source == "Ghana DigitSol-style" for h in ghana)
        assert len(ghana) >= 3  # at least 3 ads per ring is required by the plan

    def test_summary_helpers_do_not_require_opt_in(self) -> None:
        assert count_by_ring()
        assert list_case_studies()
