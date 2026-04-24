"""Tests for the Meta policy taxonomy metadata layer and its downstream uses."""

from __future__ import annotations

from counterfeint.data.audit_heuristics import (
    extract_evidence_tokens,
    has_meta_policy_citation,
)
from counterfeint.data.meta_policy_taxonomy import (
    LEGIT_CITATION_ID,
    META_TAXONOMY,
    MetaPolicyEntry,
    citation_blurb_for,
    citation_id_for,
    is_legit_category,
    lookup,
)


class TestTaxonomyCoverage:
    def test_every_fraud_category_has_entry(self) -> None:
        must_have = [
            "fake_giveaway",
            "counterfeit_goods",
            "miracle_cure",
            "advance_fee",
            "fake_crypto",
            "celebrity_endorsement_fraud",
            "clone_brand",
            "gray_area_supplements",
            "network_crypto",
            "network_ecommerce",
            "network_fintech",
            "network_health",
        ]
        for cat in must_have:
            entry = META_TAXONOMY[cat]
            assert isinstance(entry, MetaPolicyEntry)
            assert entry.citation_id != LEGIT_CITATION_ID, cat
            assert entry.section
            assert entry.subsection
            assert entry.url.startswith("https://transparency.meta.com/")

    def test_legit_categories_resolve_to_legit_placeholder(self) -> None:
        for cat in ["ecommerce", "saas", "local_service", "education", "fitness"]:
            entry = META_TAXONOMY[cat]
            assert entry.citation_id == LEGIT_CITATION_ID
            assert is_legit_category(cat)
            assert "No Meta policy violation" in entry.citation_blurb()

    def test_lookup_unknown_returns_legit(self) -> None:
        entry = lookup("unknown_category_zzz")
        assert entry.citation_id == LEGIT_CITATION_ID
        assert is_legit_category(None)

    def test_citation_ids_are_unique_across_non_legit_entries(self) -> None:
        ids = [
            e.citation_id
            for e in META_TAXONOMY.values()
            if e.citation_id != LEGIT_CITATION_ID
        ]
        assert len(ids) == len(set(ids)), f"duplicate citation IDs: {ids}"

    def test_shortcuts_return_consistent_values(self) -> None:
        assert citation_id_for("fake_giveaway") == "FSDP-IF-03"
        assert citation_id_for(None) == LEGIT_CITATION_ID
        blurb = citation_blurb_for("fake_giveaway")
        assert "FSDP-IF-03" in blurb
        assert "Fraud, Scams" in blurb


class TestEvidenceTokenRecognition:
    def test_has_meta_policy_citation_matches_expected_format(self) -> None:
        assert has_meta_policy_citation("Rejected under FSDP-IF-03 (fake giveaway).")
        assert has_meta_policy_citation("See AS-HC-07 for health-claims policy.")
        assert has_meta_policy_citation("Ring identified (CH-CIB-01).")
        assert not has_meta_policy_citation("No citation here.")
        assert not has_meta_policy_citation("random-text-without-ids")

    def test_meta_citation_shows_up_in_evidence_tokens(self) -> None:
        toks = extract_evidence_tokens(
            "Rejected under FSDP-IF-03; payment pmt_x99az matches."
        )
        flat = " ".join(toks).lower()
        assert "fsdp-if-03" in flat
        assert any(t.startswith("pmt_") for t in toks)
