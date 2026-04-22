"""Unit tests for Track A auditor graders (Investigator reasoning audit)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from counterfeint.data.audit_heuristics import (
    count_citations_matching_findings,
    count_phrase_hits,
    extract_evidence_tokens,
    hash_ad_pair,
)
from counterfeint.graders.auditor_track_a import (
    BIAS_ACC_GAP,
    CALIBRATION_GAP_THRESHOLD,
    bias_audit,
    calibration_audit,
    cross_ad_consistency_audit,
    investigator_audit_score,
    rationale_citation_audit,
    rationale_verdict_coherence_audit,
    run_track_a,
)
from counterfeint.graders.base_grader import EpisodeRecord, VerdictResult


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def make_record(
    verdicts: List[VerdictResult],
    ads_metadata: List[Dict[str, Any]] | None = None,
    *,
    task_id: str = "task_1",
    total_steps: int = 10,
    action_budget: int = 25,
) -> EpisodeRecord:
    return EpisodeRecord(
        task_id=task_id,
        total_steps=total_steps,
        action_budget=action_budget,
        verdicts=verdicts,
        links=[],
        ads_metadata=ads_metadata or [],
    )


def vr(
    ad_id: str,
    verdict: str,
    ground_truth: str,
    *,
    confidence: float = 0.8,
    auto_approved: bool = False,
) -> VerdictResult:
    return VerdictResult(
        ad_id=ad_id,
        verdict=verdict,
        confidence=confidence,
        ground_truth=ground_truth,
        auto_approved=auto_approved,
    )


def ad(
    ad_id: str,
    ground_truth: str,
    *,
    severity: float = 0.5,
    fraud_type: str = "",
    category: str = "",
    country: str = "",
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "ad_id": ad_id,
        "ground_truth": ground_truth,
        "severity": severity,
        "fraud_type": fraud_type,
        "category": category,
    }
    if country:
        meta["country"] = country
    return meta


# -----------------------------------------------------------------------------
# 1. Calibration audit
# -----------------------------------------------------------------------------


class TestCalibrationAudit:
    def test_well_calibrated_produces_no_flags(self) -> None:
        verdicts = [
            vr("ad_001", "reject", "fraud", confidence=0.85),
            vr("ad_002", "reject", "fraud", confidence=0.85),
            vr("ad_003", "approve", "legit", confidence=0.85),
            vr("ad_004", "approve", "legit", confidence=0.85),
            vr("ad_005", "escalate", "escalate", confidence=0.50),
            vr("ad_006", "escalate", "escalate", confidence=0.50),
        ]
        flags = calibration_audit(make_record(verdicts))
        assert flags == []

    def test_high_confidence_all_wrong_flags_miscalibration(self) -> None:
        verdicts = [
            vr("ad_001", "approve", "fraud", confidence=0.95),
            vr("ad_002", "approve", "fraud", confidence=0.95),
            vr("ad_003", "approve", "fraud", confidence=0.95),
            vr("ad_004", "approve", "fraud", confidence=0.95),
        ]
        flags = calibration_audit(make_record(verdicts))
        miscal = [f for f in flags if f.flag_type == "miscalibration"]
        assert miscal, "should flag miscalibration when high-conf is all wrong"
        assert miscal[0].severity > CALIBRATION_GAP_THRESHOLD

    def test_few_verdicts_skips_audit(self) -> None:
        verdicts = [vr("ad_001", "reject", "fraud", confidence=0.9)]
        flags = calibration_audit(make_record(verdicts))
        assert flags == []


# -----------------------------------------------------------------------------
# 2. Citation audit
# -----------------------------------------------------------------------------


class TestCitationAudit:
    def test_rationale_with_matching_evidence_passes(self) -> None:
        inv_actions = [
            {
                "action_type": "verdict",
                "ad_id": "ad_001",
                "rationale": (
                    "Domain shady-site.cn has NO SSL and uses privacy registrar "
                    "Njalla; recommend reject."
                ),
                "verdict": "reject",
            }
        ]
        findings = {
            "ad_001": {
                "landing_page": (
                    "Domain: shady-site.cn\nSSL: NO SSL / expired certificate\n"
                    "Registrar: Njalla (privacy)\n"
                ),
            }
        }
        flags = rationale_citation_audit(inv_actions, findings)
        assert flags == []

    def test_rationale_too_short_flags_missing_citation(self) -> None:
        inv_actions = [
            {
                "action_type": "verdict",
                "ad_id": "ad_001",
                "rationale": "reject",
                "verdict": "reject",
            }
        ]
        findings = {"ad_001": {"landing_page": "anything"}}
        flags = rationale_citation_audit(inv_actions, findings)
        assert any(f.flag_type == "missing_citation" for f in flags)

    def test_rationale_with_no_matching_tokens_is_flagged(self) -> None:
        inv_actions = [
            {
                "action_type": "verdict",
                "ad_id": "ad_001",
                "rationale": (
                    "I think this ad is probably bad because it feels wrong "
                    "and the vibe is off about the whole situation here."
                ),
                "verdict": "reject",
            }
        ]
        findings = {
            "ad_001": {
                "landing_page": "Domain: example.com\nSSL: valid\n",
                "payment_method": "Payment ID pmt_12345 on file for 3 years.",
            }
        }
        flags = rationale_citation_audit(inv_actions, findings)
        assert any(f.flag_type == "missing_citation" for f in flags)

    def test_non_verdict_actions_ignored(self) -> None:
        inv_actions = [
            {
                "action_type": "investigate",
                "ad_id": "ad_001",
                "rationale": "...",
                "investigation_target": "landing_page",
            }
        ]
        flags = rationale_citation_audit(inv_actions, {"ad_001": {"x": "y"}})
        assert flags == []


# -----------------------------------------------------------------------------
# 3. Coherence audit
# -----------------------------------------------------------------------------


class TestCoherenceAudit:
    def test_approve_with_many_red_flags_is_incoherent(self) -> None:
        inv_actions = [
            {
                "action_type": "verdict",
                "ad_id": "ad_001",
                "rationale": (
                    "The domain has NO SSL, the advertiser has a prior ban, "
                    "and uses a privacy registrar. Approve."
                ),
                "verdict": "approve",
            }
        ]
        flags = rationale_verdict_coherence_audit(inv_actions)
        assert len(flags) == 1
        assert flags[0].flag_type == "incoherent_rationale"
        assert flags[0].target_ad_id == "ad_001"

    def test_reject_with_many_green_flags_is_incoherent(self) -> None:
        inv_actions = [
            {
                "action_type": "verdict",
                "ad_id": "ad_001",
                "rationale": (
                    "Verified business, valid SSL, clean record, no violations, "
                    "established domain. Reject."
                ),
                "verdict": "reject",
            }
        ]
        flags = rationale_verdict_coherence_audit(inv_actions)
        assert len(flags) == 1
        assert flags[0].flag_type == "incoherent_rationale"

    def test_consistent_reject_with_red_flags_is_ok(self) -> None:
        inv_actions = [
            {
                "action_type": "verdict",
                "ad_id": "ad_001",
                "rationale": "No SSL, prior ban, privacy registrar. Reject.",
                "verdict": "reject",
            }
        ]
        assert rationale_verdict_coherence_audit(inv_actions) == []

    def test_escalate_is_exempt(self) -> None:
        inv_actions = [
            {
                "action_type": "verdict",
                "ad_id": "ad_001",
                "rationale": "No SSL, prior ban, privacy registrar. Escalate.",
                "verdict": "escalate",
            }
        ]
        assert rationale_verdict_coherence_audit(inv_actions) == []


# -----------------------------------------------------------------------------
# 4. Cross-ad consistency audit
# -----------------------------------------------------------------------------


class TestConsistencyAudit:
    def test_same_template_diverging_verdicts_flagged(self) -> None:
        ads = [
            ad("ad_001", "fraud", fraud_type="fake_crypto", severity=0.7),
            ad("ad_002", "fraud", fraud_type="fake_crypto", severity=0.7),
        ]
        verdicts = [
            vr("ad_001", "reject", "fraud", confidence=0.9),
            vr("ad_002", "approve", "fraud", confidence=0.9),
        ]
        flags = cross_ad_consistency_audit(make_record(verdicts, ads))
        assert any(f.flag_type == "inconsistency" for f in flags)

    def test_same_template_same_verdict_is_ok(self) -> None:
        ads = [
            ad("ad_001", "fraud", fraud_type="fake_crypto", severity=0.7),
            ad("ad_002", "fraud", fraud_type="fake_crypto", severity=0.7),
        ]
        verdicts = [
            vr("ad_001", "reject", "fraud", confidence=0.9),
            vr("ad_002", "reject", "fraud", confidence=0.9),
        ]
        flags = cross_ad_consistency_audit(make_record(verdicts, ads))
        assert flags == []

    def test_different_ground_truth_not_compared(self) -> None:
        ads = [
            ad("ad_001", "fraud", fraud_type="fake_crypto", severity=0.7),
            ad("ad_002", "legit", fraud_type="fake_crypto", severity=0.7),
        ]
        verdicts = [
            vr("ad_001", "reject", "fraud", confidence=0.9),
            vr("ad_002", "approve", "legit", confidence=0.9),
        ]
        flags = cross_ad_consistency_audit(make_record(verdicts, ads))
        assert flags == []


# -----------------------------------------------------------------------------
# 5. Bias audit
# -----------------------------------------------------------------------------


class TestBiasAudit:
    def test_country_bias_flagged(self) -> None:
        # Investigator gets every US ad right, every Nigerian ad wrong.
        ads = [
            ad(f"ad_us_{i}", "fraud", country="US", severity=0.6)
            for i in range(4)
        ] + [
            ad(f"ad_ng_{i}", "fraud", country="NG", severity=0.6)
            for i in range(4)
        ]
        verdicts = [
            vr(f"ad_us_{i}", "reject", "fraud", confidence=0.9) for i in range(4)
        ] + [
            vr(f"ad_ng_{i}", "approve", "fraud", confidence=0.9) for i in range(4)
        ]
        flags = bias_audit(make_record(verdicts, ads))
        assert any(
            f.flag_type == "bias" and "country" in (f.note or "")
            for f in flags
        )

    def test_balanced_accuracy_no_bias(self) -> None:
        ads = (
            [ad(f"ad_us_{i}", "fraud", country="US", severity=0.6) for i in range(3)]
            + [ad(f"ad_eu_{i}", "fraud", country="EU", severity=0.6) for i in range(3)]
        )
        verdicts = [
            vr("ad_us_0", "reject", "fraud"),
            vr("ad_us_1", "reject", "fraud"),
            vr("ad_us_2", "approve", "fraud"),
            vr("ad_eu_0", "reject", "fraud"),
            vr("ad_eu_1", "reject", "fraud"),
            vr("ad_eu_2", "approve", "fraud"),
        ]
        flags = bias_audit(make_record(verdicts, ads))
        country_flags = [f for f in flags if "country" in (f.note or "")]
        assert not country_flags

    def test_tiny_slices_not_flagged(self) -> None:
        ads = [
            ad("ad_us_1", "fraud", country="US", severity=0.6),
            ad("ad_eu_1", "fraud", country="EU", severity=0.6),
        ]
        verdicts = [
            vr("ad_us_1", "reject", "fraud"),
            vr("ad_eu_1", "approve", "fraud"),
        ]
        flags = bias_audit(make_record(verdicts, ads))
        assert flags == []


# -----------------------------------------------------------------------------
# Integration: run_track_a + investigator_audit_score
# -----------------------------------------------------------------------------


class TestTrackAIntegration:
    def test_clean_episode_zero_flags_max_score(self) -> None:
        ads = [
            ad("ad_001", "legit", category="ecommerce", severity=0.0),
            ad("ad_002", "legit", category="saas", severity=0.0),
            ad("ad_003", "fraud", fraud_type="fake_crypto", severity=0.8),
            ad("ad_004", "fraud", fraud_type="fake_crypto", severity=0.8),
        ]
        verdicts = [
            vr("ad_001", "approve", "legit", confidence=0.85),
            vr("ad_002", "approve", "legit", confidence=0.85),
            vr("ad_003", "reject", "fraud", confidence=0.85),
            vr("ad_004", "reject", "fraud", confidence=0.85),
        ]
        inv_actions = [
            {
                "action_type": "verdict",
                "ad_id": v.ad_id,
                "verdict": v.verdict,
                "rationale": (
                    "Reviewed findings including domain and advertiser history. "
                    "Domain example.com has valid SSL; advertiser has clean record."
                ) if v.verdict == "approve" else (
                    "Domain shady-site.cn has NO SSL and uses privacy registrar "
                    "Njalla; advertiser has prior ban on record."
                ),
            }
            for v in verdicts
        ]
        findings = {
            v.ad_id: {
                "landing_page": (
                    "Domain: example.com\nSSL: Valid SSL certificate\n"
                    if v.verdict == "approve"
                    else "Domain: shady-site.cn\nSSL: NO SSL / expired certificate\n"
                    "Registrar: Njalla (privacy)"
                ),
                "advertiser_history": (
                    "Clean record, no violations, verified business."
                    if v.verdict == "approve"
                    else "Prior ban on record; 2 policy violations."
                ),
            }
            for v in verdicts
        }
        flags = run_track_a(
            make_record(verdicts, ads),
            investigator_actions=inv_actions,
            investigation_data_seen=findings,
        )
        assert flags == []
        assert investigator_audit_score(flags) == pytest.approx(1.0)

    def test_investigator_audit_score_decays_with_flags(self) -> None:
        ads = [ad(f"ad_{i}", "fraud", fraud_type="fake_crypto", severity=0.7) for i in range(4)]
        verdicts = [
            vr("ad_0", "approve", "fraud", confidence=0.95),
            vr("ad_1", "approve", "fraud", confidence=0.95),
            vr("ad_2", "approve", "fraud", confidence=0.95),
            vr("ad_3", "reject", "fraud", confidence=0.95),
        ]
        flags = run_track_a(
            make_record(verdicts, ads),
            investigator_actions=[],
            investigation_data_seen={},
        )
        clean = run_track_a(
            make_record(
                [vr(f"ad_{i}", "reject", "fraud", confidence=0.85) for i in range(4)],
                ads,
            ),
            investigator_actions=[],
            investigation_data_seen={},
        )
        assert investigator_audit_score(flags) < investigator_audit_score(clean)


# -----------------------------------------------------------------------------
# audit_heuristics building blocks
# -----------------------------------------------------------------------------


class TestAuditHeuristics:
    def test_extract_evidence_tokens_finds_payment_domain_registrar(self) -> None:
        text = (
            "Suspicious payment id pmt_99999 on shady.cn registered with Njalla."
        )
        toks = extract_evidence_tokens(text)
        assert any(t.startswith("pmt_") for t in toks)
        assert any("shady.cn" in t for t in toks)
        assert any("njalla" in t.lower() for t in toks)

    def test_count_citations_needs_both_rationale_and_findings(self) -> None:
        assert count_citations_matching_findings("abc", "") == 0
        assert count_citations_matching_findings("", "abc") == 0

    def test_count_phrase_hits_case_insensitive(self) -> None:
        text = "Landing page has NO SSL and uses PRIVACY registrar with PRIOR BAN."
        assert count_phrase_hits(text, ["no ssl", "privacy registrar", "prior ban"]) == 3

    def test_hash_ad_pair_same_template_returns_key(self) -> None:
        a = ad("ad_1", "fraud", fraud_type="fake_crypto", severity=0.7)
        b = ad("ad_2", "fraud", fraud_type="fake_crypto", severity=0.7)
        key = hash_ad_pair(a, b)
        assert key is not None and "fake_crypto" in key

    def test_hash_ad_pair_diff_severity_none(self) -> None:
        a = ad("ad_1", "fraud", fraud_type="fake_crypto", severity=0.1)
        b = ad("ad_2", "fraud", fraud_type="fake_crypto", severity=0.9)
        assert hash_ad_pair(a, b) is None

    def test_hash_ad_pair_self_none(self) -> None:
        a = ad("ad_1", "fraud", fraud_type="fake_crypto", severity=0.7)
        assert hash_ad_pair(a, a) is None
