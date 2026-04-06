"""Tests for deterministic data generation."""

import json

from ad_fraud_env.data.ad_generator import generate_episode


class TestDeterminism:
    def test_same_seed_produces_identical_output(self):
        """Generate with seed=42 twice — output must be byte-identical."""
        ep1 = generate_episode(seed=42, task_id="task_1")
        ep2 = generate_episode(seed=42, task_id="task_1")

        assert len(ep1.ads) == len(ep2.ads)
        for a1, a2 in zip(ep1.ads, ep2.ads):
            assert a1.ad_id == a2.ad_id
            assert a1.ad_copy == a2.ad_copy
            assert a1.ground_truth_label == a2.ground_truth_label

        for ad_id in ep1.investigation_data:
            for target in ep1.investigation_data[ad_id]:
                assert (
                    ep1.investigation_data[ad_id][target]
                    == ep2.investigation_data[ad_id][target]
                )

    def test_different_seeds_produce_different_output(self):
        ep1 = generate_episode(seed=42, task_id="task_1")
        ep2 = generate_episode(seed=99, task_id="task_1")

        copies_1 = {a.ad_copy for a in ep1.ads}
        copies_2 = {a.ad_copy for a in ep2.ads}
        assert copies_1 != copies_2

    def test_task_configs_produce_correct_queue_sizes(self):
        for task_id, expected_size in [("task_1", 5), ("task_2", 12), ("task_3", 20)]:
            ep = generate_episode(seed=42, task_id=task_id)
            assert len(ep.ads) == expected_size, f"{task_id}: expected {expected_size}, got {len(ep.ads)}"

    def test_task3_has_fraud_rings(self):
        ep = generate_episode(seed=42, task_id="task_3")
        assert len(ep.fraud_rings) > 0, "Task 3 should have fraud rings"
        for ring in ep.fraud_rings:
            assert len(ring.member_ad_ids) >= 3
            assert len(ring.shared_signals) >= 2
            assert ring.topology in ("clique", "chain", "hub_spoke")

    def test_investigation_data_exists_for_all_ads(self):
        ep = generate_episode(seed=42, task_id="task_2")
        expected_targets = [
            "advertiser_history", "landing_page", "payment_method",
            "targeting_overlap", "creative_similarity", "campaign_structure",
        ]
        for ad in ep.ads:
            assert ad.ad_id in ep.investigation_data
            for target in expected_targets:
                assert target in ep.investigation_data[ad.ad_id], (
                    f"Missing {target} for {ad.ad_id}"
                )
                assert len(ep.investigation_data[ad.ad_id][target]) > 0

    def test_ground_truth_distribution(self):
        ep = generate_episode(seed=42, task_id="task_2")
        labels = [a.ground_truth_label for a in ep.ads]
        assert "fraud" in labels
        assert "legit" in labels


class TestNoExplicitCrossAdReferences:
    """Investigation text must not explicitly name other ad IDs."""

    def test_payment_investigation_no_cross_refs(self):
        ep = generate_episode(seed=42, task_id="task_3")
        for ad_id, inv in ep.investigation_data.items():
            text = inv["payment_method"]
            for other_ad in ep.investigation_data:
                if other_ad == ad_id:
                    continue
                assert other_ad not in text, (
                    f"Payment investigation for {ad_id} references {other_ad}"
                )

    def test_targeting_investigation_no_cross_refs(self):
        ep = generate_episode(seed=42, task_id="task_3")
        for ad_id, inv in ep.investigation_data.items():
            text = inv["targeting_overlap"]
            assert "HIGH OVERLAP detected with:" not in text

    def test_creative_investigation_no_cross_refs(self):
        ep = generate_episode(seed=42, task_id="task_3")
        for ad_id, inv in ep.investigation_data.items():
            text = inv["creative_similarity"]
            assert "STRONG SIMILARITY detected with:" not in text

    def test_campaign_investigation_no_cross_refs(self):
        ep = generate_episode(seed=42, task_id="task_3")
        for ad_id, inv in ep.investigation_data.items():
            text = inv["campaign_structure"]
            assert "MATCH:" not in text


class TestDecoysAndRealism:
    def test_advertiser_profiles_have_temporal_signals(self):
        ep = generate_episode(seed=42, task_id="task_2")
        for ad_id, profile in ep.advertiser_profiles.items():
            assert profile.account_created_date, f"Missing created date for {ad_id}"
            assert profile.spend_velocity, f"Missing spend velocity for {ad_id}"
            assert profile.ad_submission_pattern, f"Missing submission pattern for {ad_id}"

    def test_temporal_signals_appear_in_investigation(self):
        ep = generate_episode(seed=42, task_id="task_2")
        for ad_id, inv in ep.investigation_data.items():
            text = inv["advertiser_history"]
            assert "Account created:" in text or "Account age:" in text
            assert "Spend velocity:" in text or "spend" in text.lower()

    def test_ring_members_share_creation_week(self):
        """Ring members should have account creation dates within 7 days of each other."""
        from datetime import date
        ep = generate_episode(seed=42, task_id="task_3")
        for ring in ep.fraud_rings:
            dates = []
            for ad_id in ring.member_ad_ids:
                profile = ep.advertiser_profiles[ad_id]
                d = date.fromisoformat(profile.account_created_date)
                dates.append(d)
            if len(dates) >= 2:
                spread = (max(dates) - min(dates)).days
                assert spread <= 7, (
                    f"Ring {ring.ring_id} creation dates spread: {spread} days"
                )

    def test_investigation_has_whois_privacy_info(self):
        ep = generate_episode(seed=42, task_id="task_2")
        found_whois = False
        for ad_id, inv in ep.investigation_data.items():
            text = inv["landing_page"]
            if "WHOIS privacy:" in text:
                found_whois = True
                break
        assert found_whois, "At least one landing page should mention WHOIS privacy"
