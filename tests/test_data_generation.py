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
