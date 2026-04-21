"""Sanity tests for the scripted baseline policies."""

from __future__ import annotations

import string

from counterfeint.models import AdReviewAction, AuditorAction, FraudsterAction
from counterfeint.scripted import (
    GibberishFraudster,
    HeuristicAuditor,
    ReactiveFraudster,
    ScriptedFraudster,
    ScriptedInvestigator,
)


class TestScriptedFraudster:
    def test_proposes_ad_with_valid_category(self):
        policy = ScriptedFraudster(seed=3)
        obs = {
            "proposals_remaining": 5,
            "actions_left_this_turn": 3,
            "round_number": 1,
            "allowed_categories": ["fake_giveaway", "ecommerce", "saas"],
            "prior_verdicts": [],
            "investigation_targets_used": {},
            "current_queue": [],
        }
        action = policy.act(obs)
        assert isinstance(action, FraudsterAction)
        assert action.action_type in ("propose_ad", "commit_final")
        if action.action_type == "propose_ad":
            assert action.category in obs["allowed_categories"]
            assert action.ad_copy

    def test_commits_when_budget_exhausted(self):
        policy = ScriptedFraudster(seed=1)
        obs = {
            "proposals_remaining": 0,
            "actions_left_this_turn": 1,
            "round_number": 4,
            "allowed_categories": ["ecommerce"],
            "prior_verdicts": [],
            "investigation_targets_used": {},
            "current_queue": [],
        }
        action = policy.act(obs)
        assert action.action_type == "commit_final"


class TestReactiveFraudster:
    def test_pivots_to_camouflage_after_rejection(self):
        policy = ReactiveFraudster(seed=42)
        obs = {
            "proposals_remaining": 3,
            "actions_left_this_turn": 3,
            "round_number": 2,
            "allowed_categories": [
                "fake_giveaway", "fake_crypto", "ecommerce", "saas", "fitness",
            ],
            "prior_verdicts": [
                {
                    "ad_id": "ad_010",
                    "verdict": "reject",
                    "confidence": 0.9,
                    "rationale": "fraud",
                    "was_my_proposal": True,
                },
            ],
            "investigation_targets_used": {
                "ad_010": ["landing_page", "landing_page"],
            },
            "current_queue": [],
        }
        action = policy.act(obs)
        if action.action_type == "propose_ad":
            assert action.category in (
                "ecommerce", "saas", "fitness", "education", "local_service",
            )
            assert action.landing_page_blurb and "verified" in action.landing_page_blurb.lower()

    def test_modifies_pending_slot_when_investigator_focused(self):
        policy = ReactiveFraudster(seed=123)
        obs = {
            "proposals_remaining": 3,
            "actions_left_this_turn": 3,
            "round_number": 2,
            "allowed_categories": ["ecommerce", "saas"],
            "prior_verdicts": [],
            "investigation_targets_used": {
                "ad_001": ["landing_page", "landing_page", "landing_page"],
            },
            "current_queue": [
                {
                    "ad_id": "ad_001",
                    "status": "pending",
                    "is_my_proposal": True,
                    "slot_index": 0,
                }
            ],
        }
        actions_sampled = [policy.act(obs) for _ in range(20)]
        action_types = {a.action_type for a in actions_sampled}
        assert "modify_pending_ad" in action_types or "propose_ad" in action_types


class TestGibberishFraudster:
    def test_produces_low_entropy_nonword_copy(self):
        policy = GibberishFraudster(seed=0)
        obs = {
            "proposals_remaining": 5,
            "actions_left_this_turn": 3,
            "allowed_categories": ["ecommerce"],
            "round_number": 1,
            "prior_verdicts": [],
            "investigation_targets_used": {},
            "current_queue": [],
        }
        action = policy.act(obs)
        assert action.action_type == "propose_ad"
        assert action.ad_copy and len(action.ad_copy) >= 20
        alpha_only = "".join(c for c in (action.ad_copy or "") if c.isalpha())
        letters_set = set(alpha_only)
        assert letters_set.issubset(set(string.ascii_lowercase))


class TestScriptedInvestigator:
    def test_investigates_first_then_verdicts(self):
        policy = ScriptedInvestigator()
        base_obs = {
            "available_ads": ["ad_001", "ad_002"],
            "current_ad_info": "Ad in Focus: ad_001\ncategory: fake_giveaway\nad_copy: Free iPhone tap now",
            "investigation_findings": "",
            "queue_status": {"ad_001": {"category": "fake_giveaway"}},
        }
        first = policy.act(base_obs)
        assert isinstance(first, AdReviewAction)
        assert first.action_type == "investigate"
        assert first.ad_id == "ad_001"

        base_obs["investigation_findings"] = (
            "[ad_001 / landing_page] - verified testimonials - 30-day returns\n"
        )
        second = policy.act(base_obs)
        assert second.action_type == "verdict"
        assert second.ad_id == "ad_001"


class TestHeuristicAuditor:
    def test_flags_gibberish_and_submits_report(self):
        policy = HeuristicAuditor()
        obs = {
            "phase": "audit_phase",
            "fraudster_proposals": [
                {
                    "ad_id": "ad_001",
                    "ad_copy": "qqxxzkmqqwl xkqzqwlzzzqq zxqwm",
                },
                {
                    "ad_id": "ad_002",
                    "ad_copy": "Trusted brand with verified reviews and 30-day returns.",
                },
            ],
            "investigator_actions": [
                {
                    "action_type": "verdict",
                    "ad_id": "ad_001",
                    "rationale": "ok",
                },
                {
                    "action_type": "verdict",
                    "ad_id": "ad_002",
                    "rationale": "Rejecting ad_002 because the page exhibits multiple fraud markers.",
                },
            ],
        }
        acts = []
        while True:
            act = policy.act(obs)
            acts.append(act)
            if act.action_type == "submit_audit_report":
                break
            if len(acts) > 20:
                raise AssertionError("auditor never submitted report")

        types = [a.action_type for a in acts]
        assert "flag_fraudster" in types
        assert "flag_investigator" in types
        assert types[-1] == "submit_audit_report"

        final = acts[-1]
        assert isinstance(final, AuditorAction)
        assert final.audit_report is not None
        report = final.audit_report
        assert 0.0 <= report["fraudster_plausibility_score"] <= 1.0
        assert 0.0 <= report["investigator_audit_score"] <= 1.0
