"""Unit tests for the Llama Guard 3 / Purple Llama mock classifier."""

from __future__ import annotations

from counterfeint.data.policy_classifier_data import (
    LLAMA_GUARD_CATEGORIES,
    MARKER_PATTERNS,
    TS_FRAUD_MARKERS,
    classify_ad,
    detect_fraud_markers,
)


# --- Fixtures --------------------------------------------------------------

LEGIT_COPY = (
    "Shop our curated selection of artisan coffee beans. "
    "Free shipping on orders over $35. Roasted in Portland, OR."
)

FRAUD_COPY_URGENCY = (
    "ACT NOW! This deal expires in 10 minutes. Limited time offer — "
    "claim your spot before it's gone."
)

FRAUD_COPY_GET_RICH = (
    "Make $5000 daily from home with this one simple trick! "
    "Earn up to $10K per week — guaranteed results."
)

FRAUD_COPY_PHISHING = (
    "Your account has been flagged. Click here to verify your identity "
    "and confirm your account credentials before it's locked."
)


# --- Determinism ----------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self):
        """Same ad_id + same ad_copy → identical output (judge-reproducible)."""
        r1 = classify_ad(
            ad_id="ad_007",
            ad_copy=FRAUD_COPY_URGENCY,
            landing_page_text="",
            ground_truth_label="fraud",
            fraud_type="fake_giveaway",
        )
        r2 = classify_ad(
            ad_id="ad_007",
            ad_copy=FRAUD_COPY_URGENCY,
            landing_page_text="",
            ground_truth_label="fraud",
            fraud_type="fake_giveaway",
        )
        assert r1.verdict == r2.verdict
        assert r1.confidence == r2.confidence
        assert r1.triggered_lg_categories == r2.triggered_lg_categories
        assert r1.triggered_fraud_markers == r2.triggered_fraud_markers
        assert r1.explanation == r2.explanation

    def test_different_ad_ids_decorrelate(self):
        """Different ad_ids with the same copy still produce per-ad variation."""
        results = [
            classify_ad(
                ad_id=f"ad_{i:03d}",
                ad_copy="Click here to claim your prize.",
                ground_truth_label=None,
            )
            for i in range(1, 51)
        ]
        # Confidence should vary (not all ads get the exact same confidence).
        confidences = {round(r.confidence, 3) for r in results}
        assert len(confidences) > 5, (
            f"Expected per-ad confidence variation, got {len(confidences)} unique values"
        )


# --- Ground-truth correlation ---------------------------------------------


class TestGroundTruthCorrelation:
    """Fraud ads should trigger 'unsafe' more often than legit ads."""

    def test_fraud_unsafe_rate_high(self):
        """P(unsafe | fraud) ≈ 0.75 ± noise."""
        n = 80
        unsafe_count = 0
        for i in range(n):
            r = classify_ad(
                ad_id=f"fraud_{i:03d}",
                ad_copy=FRAUD_COPY_URGENCY,
                landing_page_text="",
                ground_truth_label="fraud",
                fraud_type="fake_giveaway",
            )
            if r.verdict == "unsafe":
                unsafe_count += 1
        rate = unsafe_count / n
        assert rate >= 0.5, (
            f"Fraud unsafe rate too low: {rate:.2f} (expected ≥0.5 per P_UNSAFE_GIVEN_FRAUD=0.75)"
        )

    def test_legit_unsafe_rate_low(self):
        """P(unsafe | legit) ≈ 0.05 ± noise."""
        n = 80
        unsafe_count = 0
        for i in range(n):
            r = classify_ad(
                ad_id=f"legit_{i:03d}",
                ad_copy=LEGIT_COPY,
                landing_page_text="",
                ground_truth_label="legit",
                fraud_type=None,
            )
            if r.verdict == "unsafe":
                unsafe_count += 1
        rate = unsafe_count / n
        assert rate < 0.2, (
            f"Legit unsafe rate too high: {rate:.2f} (expected <0.2 per P_UNSAFE_GIVEN_LEGIT=0.05)"
        )

    def test_fraud_rate_strictly_higher_than_legit(self):
        """The classifier must discriminate — fraud rate > legit rate."""
        n = 60
        fraud_unsafe = 0
        legit_unsafe = 0
        for i in range(n):
            fr = classify_ad(
                ad_id=f"fraud_pair_{i:03d}",
                ad_copy=FRAUD_COPY_URGENCY,
                ground_truth_label="fraud",
                fraud_type="fake_giveaway",
            )
            lr = classify_ad(
                ad_id=f"legit_pair_{i:03d}",
                ad_copy=LEGIT_COPY,
                ground_truth_label="legit",
            )
            if fr.verdict == "unsafe":
                fraud_unsafe += 1
            if lr.verdict == "unsafe":
                legit_unsafe += 1
        assert fraud_unsafe > legit_unsafe + 10, (
            f"Expected a clear fraud-vs-legit gap, got "
            f"fraud={fraud_unsafe}/{n}, legit={legit_unsafe}/{n}"
        )


# --- Surface-marker detection ---------------------------------------------


class TestSurfaceMarkers:
    def test_urgency_detected(self):
        markers = detect_fraud_markers(FRAUD_COPY_URGENCY)
        assert "high_pressure_urgency" in markers

    def test_get_rich_detected(self):
        markers = detect_fraud_markers(FRAUD_COPY_GET_RICH)
        assert "get_rich_quick" in markers
        assert "unrealistic_guarantee" in markers

    def test_phishing_detected(self):
        markers = detect_fraud_markers(FRAUD_COPY_PHISHING)
        assert "phishing_credential_request" in markers

    def test_legit_copy_no_markers(self):
        markers = detect_fraud_markers(LEGIT_COPY)
        assert markers == [] or len(markers) == 0

    def test_two_markers_without_label_triggers_unsafe(self):
        """Without a ground-truth label, 2+ markers should produce 'unsafe'."""
        r = classify_ad(
            ad_id="mk_test_01",
            ad_copy=FRAUD_COPY_GET_RICH,  # has get_rich + guarantee
            ground_truth_label=None,
        )
        assert r.verdict == "unsafe"

    def test_zero_markers_without_label_is_safe(self):
        r = classify_ad(
            ad_id="mk_test_02",
            ad_copy=LEGIT_COPY,
            ground_truth_label=None,
        )
        assert r.verdict == "safe"


# --- Fraud-type → LG category mapping --------------------------------------


class TestFraudTypeMapping:
    def test_counterfeit_fires_s8_ip(self):
        """Counterfeit fraud should trigger S8 (intellectual_property)."""
        # Try multiple seeds until we get an 'unsafe' (noise is part of the spec).
        got_s8 = False
        for i in range(10):
            r = classify_ad(
                ad_id=f"ct_{i:02d}",
                ad_copy="Authentic-looking replica Rolex at 90% off retail.",
                ground_truth_label="fraud",
                fraud_type="counterfeit",
            )
            if r.verdict == "unsafe" and "S8" in r.triggered_lg_categories:
                got_s8 = True
                break
        assert got_s8, "At least one counterfeit classification should trigger S8"

    def test_brand_impersonation_fires_s5(self):
        """Brand impersonation should trigger S5 (defamation)."""
        got_s5 = False
        for i in range(10):
            r = classify_ad(
                ad_id=f"bi_{i:02d}",
                ad_copy="Official Nike giveaway — click to enter.",
                ground_truth_label="fraud",
                fraud_type="brand_impersonation",
            )
            if r.verdict == "unsafe" and "S5" in r.triggered_lg_categories:
                got_s5 = True
                break
        assert got_s5, "At least one brand_impersonation classification should trigger S5"


# --- Output shape ---------------------------------------------------------


class TestOutputShape:
    def test_to_investigation_text_renders(self):
        r = classify_ad(
            ad_id="ad_fmt",
            ad_copy=FRAUD_COPY_URGENCY,
            ground_truth_label="fraud",
            fraud_type="fake_giveaway",
        )
        text = r.to_investigation_text()
        assert text.startswith("Llama Guard 3 Classification for ad_fmt:")
        assert "Verdict:" in text
        assert "Policy explanation:" in text

    def test_to_dict_serializable(self):
        r = classify_ad(
            ad_id="ad_json",
            ad_copy=FRAUD_COPY_PHISHING,
            ground_truth_label="fraud",
            fraud_type="advance_fee_scam",
        )
        d = r.to_dict()
        import json
        s = json.dumps(d)
        assert "verdict" in s
        assert "triggered_lg_categories" in s

    def test_all_lg_codes_valid(self):
        r = classify_ad(
            ad_id="ad_lg_valid",
            ad_copy=FRAUD_COPY_URGENCY,
            ground_truth_label="fraud",
            fraud_type="fake_giveaway",
        )
        for code in r.triggered_lg_categories:
            assert code in LLAMA_GUARD_CATEGORIES, f"Unknown LG code: {code}"

    def test_all_marker_codes_valid(self):
        r = classify_ad(
            ad_id="ad_mk_valid",
            ad_copy=FRAUD_COPY_GET_RICH,
            ground_truth_label="fraud",
        )
        for marker in r.triggered_fraud_markers:
            assert marker in TS_FRAUD_MARKERS, f"Unknown TS-Fraud marker: {marker}"

    def test_confidence_in_unit_range(self):
        r = classify_ad(
            ad_id="ad_conf",
            ad_copy=FRAUD_COPY_URGENCY,
            ground_truth_label="fraud",
        )
        assert 0.0 <= r.confidence <= 1.0


# --- Integration with ad_generator ----------------------------------------


class TestEpisodeIntegration:
    def test_episode_includes_policy_classifier_per_ad(self):
        """Every ad in a generated episode should carry a policy_classifier entry."""
        from counterfeint.data.ad_generator import generate_episode
        ep = generate_episode(seed=42, task_id="task_2")
        for ad in ep.ads:
            inv = ep.investigation_data[ad.ad_id]
            assert "policy_classifier" in inv
            text = inv["policy_classifier"]
            assert text.startswith(f"Llama Guard 3 Classification for {ad.ad_id}:")
            assert "Verdict:" in text

    def test_fraud_ads_more_often_unsafe_in_episode(self):
        """In a full episode, fraud ads should land in the unsafe bucket more often."""
        from counterfeint.data.ad_generator import generate_episode
        ep = generate_episode(seed=123, task_id="task_3")
        fraud_unsafe = 0
        legit_unsafe = 0
        fraud_n = 0
        legit_n = 0
        for ad in ep.ads:
            text = ep.investigation_data[ad.ad_id]["policy_classifier"]
            is_unsafe = "Verdict: unsafe" in text
            if ad.ground_truth_label == "fraud":
                fraud_n += 1
                if is_unsafe:
                    fraud_unsafe += 1
            elif ad.ground_truth_label == "legit":
                legit_n += 1
                if is_unsafe:
                    legit_unsafe += 1
        # With ~6-10 fraud ads per task_3 episode, we expect at least a 3:1 gap.
        fraud_rate = fraud_unsafe / fraud_n if fraud_n else 0.0
        legit_rate = legit_unsafe / legit_n if legit_n else 0.0
        assert fraud_rate > legit_rate, (
            f"Fraud unsafe rate ({fraud_rate:.2f}) should exceed "
            f"legit unsafe rate ({legit_rate:.2f})"
        )
