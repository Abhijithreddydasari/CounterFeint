from counterfeint.experiments.paper_outputs import (
    ATTACK_DROP_ARM,
    _llm_flag_stats,
    acceptance_gate,
)


def test_llm_flag_stats_empty_judge():
    stats = _llm_flag_stats(
        {
            "n_cases": 2,
            "n_ok": 2,
            "mean_latency_sec": 1.5,
            "cost_per_episode_usd": 0.001,
            "flags": {"C001": [], "C002": []},
        }
    )
    assert stats is not None
    assert stats["n_flagged"] == 0
    assert stats["n_ok"] == 2


def test_hybrid_claim_blocked_without_human_table():
    rows = [
        {
            "corruption": "missing_citation",
            "arm": "full",
            "investigator_reward": 1.0,
            "fraudster_reward": 1.0,
        },
        {
            "corruption": "missing_citation",
            "arm": "no_track_a",
            "investigator_reward": 3.0,
            "fraudster_reward": 1.0,
        },
        {
            "corruption": "gibberish_ads",
            "arm": "full",
            "investigator_reward": 1.0,
            "fraudster_reward": 1.0,
        },
        {
            "corruption": "gibberish_ads",
            "arm": "no_plausibility_gate",
            "investigator_reward": 1.0,
            "fraudster_reward": 4.0,
        },
    ]
    gate = acceptance_gate(rows, {"clean_flag_rate": {}}, llm_stats={"n_flagged": 0})
    assert gate["anti_gaming_gate_pass"] is True
    assert gate["hybrid_claim_allowed"] is False


def test_attack_drop_arm_covers_non_gap_attacks():
    for name in ("missing_citation", "gibberish_ads", "template_spam"):
        assert name in ATTACK_DROP_ARM
    assert "investigation_farming" not in ATTACK_DROP_ARM
