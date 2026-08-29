"""Tests for corruption suite and auditor validation."""

from __future__ import annotations

import pytest

from counterfeint.experiments.corruptions import (
    CORRUPTION_REGISTRY,
    apply_corruption,
    corrupt_gibberish_ads,
    corrupt_missing_citation,
)
from counterfeint.experiments.episode_bundle import run_episode_bundle
from counterfeint.experiments.auditor_validation import (
    run_corruption_validation,
    validate_bundle,
)


@pytest.fixture
def base_bundle():
    return run_episode_bundle(task_id="task_1", seed=42)


def test_all_corruptions_registered():
    assert len(CORRUPTION_REGISTRY) >= 7


def test_missing_citation_fires_flag(base_bundle):
    corrupted = corrupt_missing_citation(base_bundle)
    expected, fired = validate_bundle(corrupted)
    assert "missing_citation" in expected
    assert "missing_citation" in fired


def test_gibberish_fires_flag(base_bundle):
    corrupted = corrupt_gibberish_ads(base_bundle)
    expected, fired = validate_bundle(corrupted)
    assert "gibberish" in expected
    assert "gibberish" in fired or "template_repetition" in fired


def test_corruption_validation_runs(base_bundle):
    report = run_corruption_validation([base_bundle])
    assert report.n_corruptions == len(CORRUPTION_REGISTRY)
    assert report.per_flag  # at least some flags measured


def test_apply_corruption_by_name(base_bundle):
    out = apply_corruption(base_bundle, "missing_citation")
    assert out.corruption_label == "missing_citation"
