"""Tests for the offline LLM auditor merge policy."""

from __future__ import annotations

from counterfeint.experiments.llm_auditor import (
    EXACT_FLAG_TYPES,
    SEMANTIC_FLAG_TYPES,
    merge_hybrid,
    parse_llm_flags,
)
from counterfeint.models import AuditFlag


def _f(flag_type: str, ad: str | None, track: str, sev: float, note: str = "") -> AuditFlag:
    return AuditFlag(
        track=track,
        target_ad_id=ad,
        flag_type=flag_type,
        severity=sev,
        note=note,
    )


def test_hybrid_keeps_deterministic_exact_checks():
    det = [_f("gibberish", "ad_1", "B", 1.0, "det")]
    llm = [_f("gibberish", "ad_1", "B", 0.4, "llm")]
    merged = merge_hybrid(det, llm)
    by = {(f.flag_type, f.target_ad_id): f for f in merged}
    assert by[("gibberish", "ad_1")].note == "det"


def test_hybrid_takes_llm_semantic_when_stronger():
    det = [_f("missing_citation", "ad_1", "A", 0.4, "det")]
    llm = [_f("missing_citation", "ad_1", "A", 0.9, "llm")]
    merged = merge_hybrid(det, llm)
    by = {(f.flag_type, f.target_ad_id): f for f in merged}
    assert by[("missing_citation", "ad_1")].severity == 0.9


def test_flag_type_partitions():
    assert "incoherent_rationale" in SEMANTIC_FLAG_TYPES
    assert "gibberish" in EXACT_FLAG_TYPES


def test_parse_accepts_bare_flag_list():
    flags = parse_llm_flags(
        '[{"track": "A", "flag_type": "missing_citation", "severity": 0.8}]'
    )
    assert len(flags) == 1
    assert flags[0].flag_type == "missing_citation"


def test_parse_skips_non_object_items():
    flags = parse_llm_flags({"flags": ["nope", {"track": "B", "flag_type": "gibberish", "severity": 1}]})
    assert len(flags) == 1
    assert flags[0].track == "B"
