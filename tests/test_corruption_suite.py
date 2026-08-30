"""Tests for corruption suite, paired-delta validation, hybrid merge, ablations."""

from __future__ import annotations

import json

import pytest

from counterfeint.experiments.auditor_validation import (
    audit_flag_types,
    paired_delta,
    run_corruption_validation,
    validate_bundle,
)
from counterfeint.experiments.corruptions import (
    CORRUPTION_REGISTRY,
    FRAUDSTER_ATTACKS,
    INVESTIGATOR_ATTACKS,
    apply_corruption,
    corrupt_contradictory_rationale,
    corrupt_gibberish_ads,
    corrupt_grader_trigger_tokens,
    corrupt_missing_citation,
    corrupt_template_spam,
    sanitize_bundle,
)
from counterfeint.experiments.episode_bundle import run_episode_bundle
from counterfeint.experiments.llm_auditor import merge_hybrid, parse_llm_flags
from counterfeint.experiments.reward_arms import AblationArm, compute_role_rewards, run_audit
from counterfeint.models import AuditFlag


@pytest.fixture
def base_bundle():
    return run_episode_bundle(task_id="task_1", seed=42)


def test_all_corruptions_registered():
    assert len(CORRUPTION_REGISTRY) >= 16
    for name in INVESTIGATOR_ATTACKS + FRAUDSTER_ATTACKS:
        assert name in CORRUPTION_REGISTRY


def test_paired_delta_ignores_clean_flags():
    clean = {("missing_citation", "ad_1"), ("gibberish", "ad_2")}
    corr = {("missing_citation", "ad_1"), ("incoherent_rationale", "ad_3")}
    expected = {("incoherent_rationale", "ad_3")}
    tp, fp, fn = paired_delta(clean, corr, expected)
    assert ("incoherent_rationale", "ad_3") in tp
    assert ("missing_citation", "ad_1") not in fp
    assert not fn


def test_missing_citation_fires_flag(base_bundle):
    corrupted = corrupt_missing_citation(base_bundle)
    expected, fired = validate_bundle(corrupted)
    assert "missing_citation" in expected
    assert "missing_citation" in fired


def test_contradictory_rationale_fires_flag(base_bundle):
    corrupted = corrupt_contradictory_rationale(base_bundle)
    expected, fired = validate_bundle(corrupted)
    assert "incoherent_rationale" in expected
    assert "incoherent_rationale" in fired


def test_gibberish_fires_flag(base_bundle):
    corrupted = corrupt_gibberish_ads(base_bundle)
    expected, fired = validate_bundle(corrupted)
    assert "gibberish" in expected
    assert "gibberish" in fired or "template_repetition" in fired


def test_grader_trigger_fires_branding(base_bundle):
    corrupted = corrupt_grader_trigger_tokens(base_bundle)
    expected, fired = validate_bundle(corrupted)
    if expected:
        assert "branding_anomaly" in fired


def test_template_spam_on_sanitized_control(base_bundle):
    control = sanitize_bundle(base_bundle)
    corrupted = corrupt_template_spam(control)
    expected, fired = validate_bundle(corrupted)
    assert "template_repetition" in expected
    assert "template_repetition" in fired


def test_corruption_validation_runs(base_bundle):
    report = run_corruption_validation([base_bundle])
    assert report.n_corruptions == len(INVESTIGATOR_ATTACKS) + len(FRAUDSTER_ATTACKS)
    assert report.per_flag
    assert report.clean_flag_rate is not None


def test_sanitized_control_then_missing_citation(base_bundle):
    control = sanitize_bundle(base_bundle)
    assert "missing_citation" not in audit_flag_types(control)
    corrupted = corrupt_missing_citation(control)
    expected, fired = validate_bundle(corrupted)
    assert "missing_citation" in expected
    assert "missing_citation" in fired
    out = apply_corruption(base_bundle, "missing_citation")
    assert out.corruption_label == "missing_citation"
    assert out.side == "investigator"


def test_apply_corruption_by_name(base_bundle):
    out = apply_corruption(base_bundle, "missing_citation")
    assert out.corruption_label == "missing_citation"
    assert out.side == "investigator"


def test_parse_and_merge_hybrid():
    flags = parse_llm_flags(
        json.dumps(
            {
                "flags": [
                    {
                        "track": "A",
                        "flag_type": "incoherent_rationale",
                        "target_ad_id": "ad_1",
                        "severity": 0.8,
                        "note": "semantic",
                    }
                ]
            }
        )
    )
    det = [
        AuditFlag(
            track="A",
            target_ad_id="ad_2",
            flag_type="miscalibration",
            severity=0.9,
            note="det",
        )
    ]
    merged = merge_hybrid(det, flags)
    types = {f.flag_type for f in merged}
    assert "miscalibration" in types
    assert "incoherent_rationale" in types


def test_track_a_removal_raises_investigator_reward(base_bundle):
    attacked = corrupt_missing_citation(base_bundle)
    audit = run_audit(attacked)
    full = compute_role_rewards(attacked, AblationArm.FULL, audit=audit)
    no_a = compute_role_rewards(attacked, AblationArm.NO_TRACK_A, audit=audit)
    assert no_a.investigator_reward >= full.investigator_reward - 1e-9


def test_plausibility_gate_blocks_gibberish_reward(base_bundle):
    attacked = corrupt_gibberish_ads(base_bundle)
    audit = run_audit(attacked)
    full = compute_role_rewards(attacked, AblationArm.FULL, audit=audit)
    no_gate = compute_role_rewards(
        attacked, AblationArm.NO_PLAUSIBILITY_GATE, audit=audit
    )
    assert no_gate.fraudster_reward >= full.fraudster_reward - 1e-9
