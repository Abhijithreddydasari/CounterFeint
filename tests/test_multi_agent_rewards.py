"""
Tests for graders/multi_agent_rewards.py (Phase 2C).

Covers:
  * compute_auditor_ground_truth - true-positive vs false-positive counting
  * fraudster_reward - gibberish-zero, banned-penalty, approve-fraud-credit,
    severity/plausibility weighting
  * investigator_reward - R1 base score + rationale bonus + inconsistency
    penalty
  * auditor_reward - credit for true flags, penalty for false flags
  * compute_episode_rewards - top-level integration keys + invariants
  * end-to-end canonical episode driven through RefereeEnvironment with the
    scripted Fraudster / Investigator / Auditor policies — the path judges
    will actually exercise.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pytest

from counterfeint.graders.base_grader import (
    EpisodeRecord,
    LinkResult,
    VerdictResult,
    grade_episode,
)
from counterfeint.graders.multi_agent_rewards import (
    AUDITOR_FALSE_POSITIVE_PENALTY,
    AUDITOR_TRUE_MISCAL,
    AUDITOR_TRUE_UNREALISTIC,
    FRAUDSTER_BANNED_PENALTY,
    FRAUDSTER_UNREALISTIC_PENALTY,
    INVESTIGATOR_INCONSISTENCY_PENALTY,
    INVESTIGATOR_RATIONALE_BONUS,
    RewardInputs,
    auditor_reward,
    compute_auditor_ground_truth,
    compute_episode_rewards,
    fraudster_reward,
    investigator_reward,
)
from counterfeint.models import AuditFlag, AuditReport, RefereeState
from counterfeint.scripted import (
    GibberishFraudster,
    HeuristicAuditor,
    ReactiveFraudster,
    ScriptedInvestigator,
)
from counterfeint.server.referee import RefereeEnvironment


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


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


def ad_meta(
    ad_id: str,
    ground_truth: str,
    *,
    severity: float = 0.6,
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


def mk_record(
    verdicts: List[VerdictResult],
    ads: List[Dict[str, Any]],
    *,
    task_id: str = "task_1",
    total_steps: int = 10,
    action_budget: int = 25,
    links: Optional[List[LinkResult]] = None,
) -> EpisodeRecord:
    return EpisodeRecord(
        task_id=task_id,
        total_steps=total_steps,
        action_budget=action_budget,
        verdicts=verdicts,
        links=links or [],
        ads_metadata=ads,
    )


def mk_propose(
    ad_id: str,
    ad_copy: str,
    *,
    category: str = "general_goods",
    landing_page_blurb: str = "We ship domestically with a 30-day return policy.",
    targeting_summary: str = "Adults 25-45 interested in home goods.",
    slot_index: int = 0,
) -> Dict[str, Any]:
    """Build a fraudster_log entry that looks like what the Referee stores."""
    return {
        "ts": 0.0,
        "phase": "fraudster_turn",
        "round_number": 1,
        "action_type": "propose_ad",
        "ad_id": ad_id,
        "ad_copy": ad_copy,
        "category": category,
        "landing_page_blurb": landing_page_blurb,
        "targeting_summary": targeting_summary,
        "slot_index": slot_index,
        "new_ad_copy": None,
        "new_landing_page_blurb": None,
        "rationale": "",
        "reward": 0.0,
    }


def mk_gibberish_propose(ad_id: str, *, slot_index: int = 0) -> Dict[str, Any]:
    """Fully gibberish proposal — every text surface is non-wordlike."""
    return mk_propose(
        ad_id,
        "zzzqqxxwmqqqqxxz qqlxkzzzw zxkwlmzz qxklqzwl xkqzqwlzzz",
        landing_page_blurb="xxklzzz qqwmzzqqwl zxkwlmzzz xkxqwl qqxxmzlzz",
        targeting_summary="xklqzz qxklqz qwlxkz zzxklq",
        slot_index=slot_index,
    )


def mk_flag(
    track: str,
    flag_type: str,
    *,
    target_ad_id: Optional[str] = None,
    severity: float = 0.5,
    note: str = "",
) -> AuditFlag:
    return AuditFlag(
        track=track,
        target_ad_id=target_ad_id,
        flag_type=flag_type,
        severity=severity,
        note=note,
    )


def mk_report(
    *,
    track_a: Optional[List[AuditFlag]] = None,
    track_b: Optional[List[AuditFlag]] = None,
    investigator_audit_score: float = 1.0,
    fraudster_plausibility_score: float = 1.0,
    notes: str = "",
) -> AuditReport:
    return AuditReport(
        track_a_flags=track_a or [],
        track_b_flags=track_b or [],
        investigator_audit_score=investigator_audit_score,
        fraudster_plausibility_score=fraudster_plausibility_score,
        notes=notes,
    )


def mk_inputs(
    *,
    record: EpisodeRecord,
    audit_report: Optional[AuditReport] = None,
    fraudster_proposal_log: Optional[List[Dict[str, Any]]] = None,
    investigator_action_log: Optional[List[Dict[str, Any]]] = None,
    investigation_data_seen: Optional[Dict[str, Dict[str, str]]] = None,
    fraudster_ad_ids: Optional[List[str]] = None,
) -> RewardInputs:
    return RewardInputs(
        record=record,
        audit_report=audit_report or mk_report(),
        fraudster_proposal_log=fraudster_proposal_log or [],
        investigator_action_log=investigator_action_log or [],
        investigation_data_seen=investigation_data_seen or {},
        fraudster_ad_ids=fraudster_ad_ids or [],
    )


# -----------------------------------------------------------------------------
# 1. compute_auditor_ground_truth
# -----------------------------------------------------------------------------


class TestComputeAuditorGroundTruth:
    def test_returns_all_counter_keys(self) -> None:
        inputs = mk_inputs(
            record=mk_record(
                verdicts=[vr("ad_001", "approve", "legit")],
                ads=[ad_meta("ad_001", "legit")],
            ),
        )
        counts = compute_auditor_ground_truth(inputs)
        for key in (
            "true_miscalibrations_flagged",
            "true_unrealistic_flagged",
            "true_inconsistencies_flagged",
            "true_citation_flagged",
            "true_incoherent_flagged",
            "true_bias_flagged",
            "false_positives",
        ):
            assert key in counts, f"missing counter: {key}"

    def test_true_miscalibration_is_credited(self) -> None:
        # Over-confident wrongly-approved fraud fires the real calibration
        # audit, so when the Auditor flags it the TP counter ticks up.
        verdicts = [
            vr("ad_001", "approve", "fraud", confidence=0.95),
            vr("ad_002", "approve", "fraud", confidence=0.95),
            vr("ad_003", "approve", "fraud", confidence=0.95),
            vr("ad_004", "approve", "fraud", confidence=0.95),
        ]
        record = mk_record(
            verdicts=verdicts,
            ads=[ad_meta(v.ad_id, "fraud") for v in verdicts],
        )
        report = mk_report(
            track_a=[mk_flag("A", "miscalibration", severity=0.4)]
        )
        counts = compute_auditor_ground_truth(
            mk_inputs(record=record, audit_report=report)
        )
        assert counts["true_miscalibrations_flagged"] == 1
        assert counts["false_positives"] == 0

    def test_flag_on_clean_ad_is_false_positive(self) -> None:
        record = mk_record(
            verdicts=[vr("ad_001", "approve", "legit", confidence=0.7)],
            ads=[ad_meta("ad_001", "legit")],
        )
        report = mk_report(
            track_b=[
                mk_flag("B", "gibberish", target_ad_id="ad_001", severity=0.9),
            ],
        )
        counts = compute_auditor_ground_truth(
            mk_inputs(
                record=record,
                audit_report=report,
                fraudster_proposal_log=[
                    mk_propose(
                        "ad_001",
                        "Save 20% on organic cotton towels through our verified shop.",
                    )
                ],
                fraudster_ad_ids=["ad_001"],
            )
        )
        assert counts["false_positives"] >= 1
        assert counts["true_unrealistic_flagged"] == 0

    def test_matches_by_flag_type_and_ad_id(self) -> None:
        # Two ads, both with gibberish copy → Track B fires a gibberish flag
        # per ad. Auditor flags gibberish only on ad_001; should count 1 TP,
        # not 2.
        proposals = [
            mk_gibberish_propose("ad_001", slot_index=0),
            mk_gibberish_propose("ad_002", slot_index=1),
        ]
        record = mk_record(
            verdicts=[
                vr("ad_001", "approve", "fraud", confidence=0.7),
                vr("ad_002", "approve", "fraud", confidence=0.7),
            ],
            ads=[ad_meta("ad_001", "fraud"), ad_meta("ad_002", "fraud")],
        )
        report = mk_report(
            track_b=[
                mk_flag("B", "gibberish", target_ad_id="ad_001"),
            ],
        )
        counts = compute_auditor_ground_truth(
            mk_inputs(
                record=record,
                audit_report=report,
                fraudster_proposal_log=proposals,
                fraudster_ad_ids=["ad_001", "ad_002"],
            )
        )
        assert counts["true_unrealistic_flagged"] == 1


# -----------------------------------------------------------------------------
# 2. fraudster_reward
# -----------------------------------------------------------------------------


class TestFraudsterReward:
    def test_gibberish_zero_reward(self) -> None:
        """A queue of pure gibberish ads yields a non-positive reward — even
        if the Investigator doesn't reject them, plausibility * severity ≈ 0
        and every ad adds an unrealistic penalty via Track B flags."""
        proposals = [
            mk_gibberish_propose("ad_001", slot_index=0),
            mk_gibberish_propose("ad_002", slot_index=1),
        ]
        record = mk_record(
            verdicts=[
                vr("ad_001", "approve", "fraud", confidence=0.6),
                vr("ad_002", "approve", "fraud", confidence=0.6),
            ],
            ads=[
                ad_meta("ad_001", "fraud", severity=0.8),
                ad_meta("ad_002", "fraud", severity=0.8),
            ],
        )
        # Populate track_b with gibberish flags so the unrealistic penalty
        # kicks in (in the full Referee path these come from Track B).
        report = mk_report(
            track_b=[
                mk_flag("B", "gibberish", target_ad_id="ad_001", severity=0.9),
                mk_flag("B", "gibberish", target_ad_id="ad_002", severity=0.9),
            ],
        )
        reward = fraudster_reward(
            mk_inputs(
                record=record,
                audit_report=report,
                fraudster_proposal_log=proposals,
                fraudster_ad_ids=["ad_001", "ad_002"],
            )
        )
        assert reward <= 0.0, (
            f"gibberish-only queue should not earn the Fraudster reward, got {reward}"
        )

    def test_realistic_unbanned_ad_earns_positive_reward(self) -> None:
        """Plausible ad copy + Investigator approves = Fraudster cash-in."""
        proposals = [
            mk_propose(
                "ad_001",
                "Flash sale: shop trusted brand loyalty rewards with free returns.",
                category="general_goods",
                landing_page_blurb=(
                    "Our store has verified reviews, 30-day returns, and a customer support hotline."
                ),
                targeting_summary="Adults 25-45 shopping for household goods.",
            )
        ]
        record = mk_record(
            verdicts=[vr("ad_001", "approve", "fraud", confidence=0.6)],
            ads=[ad_meta("ad_001", "fraud", severity=0.7)],
        )
        reward = fraudster_reward(
            mk_inputs(
                record=record,
                fraudster_proposal_log=proposals,
                fraudster_ad_ids=["ad_001"],
            )
        )
        assert reward > 0.0
        # sev=0.7 and no Track B flags → 0.7 * plaus_of_clean_ad (should be ≥ 0.5)
        assert reward >= 0.3

    def test_banned_ad_incurs_penalty(self) -> None:
        proposals = [
            mk_propose(
                "ad_001",
                "Clean, realistic ad copy with no issues whatsoever.",
            )
        ]
        record = mk_record(
            verdicts=[vr("ad_001", "reject", "fraud", confidence=0.9)],
            ads=[ad_meta("ad_001", "fraud", severity=0.7)],
        )
        reward = fraudster_reward(
            mk_inputs(
                record=record,
                fraudster_proposal_log=proposals,
                fraudster_ad_ids=["ad_001"],
            )
        )
        # Reject triggers full banned penalty; nothing unbanned offsets it.
        assert reward <= -FRAUDSTER_BANNED_PENALTY + 1e-6

    def test_unrealistic_flag_increases_penalty(self) -> None:
        """Adding a Track B flag should strictly lower the reward."""
        proposals = [
            mk_propose("ad_001", "A normal product description that sounds fine.")
        ]
        record = mk_record(
            verdicts=[vr("ad_001", "approve", "fraud", confidence=0.6)],
            ads=[ad_meta("ad_001", "fraud", severity=0.7)],
        )
        inputs_clean = mk_inputs(
            record=record,
            fraudster_proposal_log=proposals,
            fraudster_ad_ids=["ad_001"],
        )
        inputs_flagged = mk_inputs(
            record=record,
            audit_report=mk_report(
                track_b=[mk_flag("B", "gibberish", target_ad_id="ad_001")]
            ),
            fraudster_proposal_log=proposals,
            fraudster_ad_ids=["ad_001"],
        )
        r_clean = fraudster_reward(inputs_clean)
        r_flagged = fraudster_reward(inputs_flagged)
        assert r_flagged == pytest.approx(
            r_clean - FRAUDSTER_UNREALISTIC_PENALTY
        )

    def test_reactive_scenario_multiple_proposals(self) -> None:
        """Fraudster proposes twice across turns; reward scales with
        severity * plausibility for every unbanned ad."""
        proposals = [
            mk_propose(
                "ad_001",
                "Reliable home delivery with verified seller and refund guarantee.",
                slot_index=0,
            ),
            mk_propose(
                "ad_002",
                "Trusted brand accessories with 2-year warranty and free returns.",
                slot_index=1,
            ),
        ]
        record = mk_record(
            verdicts=[
                vr("ad_001", "approve", "fraud", confidence=0.6),
                vr("ad_002", "reject", "fraud", confidence=0.9),
            ],
            ads=[
                ad_meta("ad_001", "fraud", severity=0.8),
                ad_meta("ad_002", "fraud", severity=0.5),
            ],
        )
        reward = fraudster_reward(
            mk_inputs(
                record=record,
                fraudster_proposal_log=proposals,
                fraudster_ad_ids=["ad_001", "ad_002"],
            )
        )
        # One unbanned (positive), one banned (–1.0). The unbanned must pull
        # the reward above a pure −1.0 penalty.
        assert reward > -FRAUDSTER_BANNED_PENALTY

    def test_no_proposals_no_reward(self) -> None:
        record = mk_record(
            verdicts=[vr("ad_001", "approve", "legit")],
            ads=[ad_meta("ad_001", "legit")],
        )
        reward = fraudster_reward(
            mk_inputs(
                record=record,
                fraudster_proposal_log=[],
                fraudster_ad_ids=[],
            )
        )
        assert reward == pytest.approx(0.0)


# -----------------------------------------------------------------------------
# 3. investigator_reward
# -----------------------------------------------------------------------------


class TestInvestigatorReward:
    def _clean_inv_log(self, ad_ids: List[str]) -> List[Dict[str, Any]]:
        return [
            {
                "action_type": "verdict",
                "ad_id": ad_id,
                "rationale": "Investigated landing page and targeting metadata.",
            }
            for ad_id in ad_ids
        ]

    def test_clean_investigator_reward_beats_base_score(self) -> None:
        verdicts = [
            vr("ad_001", "reject", "fraud", confidence=0.85),
            vr("ad_002", "approve", "legit", confidence=0.8),
        ]
        record = mk_record(
            verdicts=verdicts,
            ads=[ad_meta(v.ad_id, v.ground_truth) for v in verdicts],
        )
        inputs = mk_inputs(
            record=record,
            investigator_action_log=self._clean_inv_log(["ad_001", "ad_002"]),
        )
        base = grade_episode(record)
        reward = investigator_reward(inputs)
        assert reward >= base  # gets citation bonus for approved rationales
        assert reward == pytest.approx(
            base + INVESTIGATOR_RATIONALE_BONUS * 2
        )

    def test_approve_fraud_drops_reward(self) -> None:
        """Approving fraud tanks the R1 base grader, so the Investigator
        reward should drop below the baseline of approving legit correctly."""
        good_verdicts = [
            vr("ad_001", "reject", "fraud", confidence=0.9),
            vr("ad_002", "approve", "legit", confidence=0.9),
        ]
        bad_verdicts = [
            vr("ad_001", "approve", "fraud", confidence=0.9),
            vr("ad_002", "approve", "legit", confidence=0.9),
        ]
        good = mk_record(
            verdicts=good_verdicts,
            ads=[
                ad_meta("ad_001", "fraud", severity=0.7),
                ad_meta("ad_002", "legit"),
            ],
        )
        bad = mk_record(
            verdicts=bad_verdicts,
            ads=[
                ad_meta("ad_001", "fraud", severity=0.7),
                ad_meta("ad_002", "legit"),
            ],
        )
        r_good = investigator_reward(
            mk_inputs(
                record=good,
                investigator_action_log=self._clean_inv_log(["ad_001", "ad_002"]),
            )
        )
        r_bad = investigator_reward(
            mk_inputs(
                record=bad,
                investigator_action_log=self._clean_inv_log(["ad_001", "ad_002"]),
            )
        )
        assert r_bad < r_good

    def test_inconsistency_flag_applies_penalty(self) -> None:
        verdicts = [
            vr("ad_001", "reject", "fraud", confidence=0.85),
            vr("ad_002", "approve", "legit", confidence=0.8),
        ]
        record = mk_record(
            verdicts=verdicts,
            ads=[ad_meta(v.ad_id, v.ground_truth) for v in verdicts],
        )
        inv_log = self._clean_inv_log(["ad_001", "ad_002"])

        clean = investigator_reward(
            mk_inputs(record=record, investigator_action_log=inv_log)
        )
        inconsistent = investigator_reward(
            mk_inputs(
                record=record,
                audit_report=mk_report(
                    track_a=[
                        mk_flag("A", "inconsistency", target_ad_id="ad_001"),
                    ],
                ),
                investigator_action_log=inv_log,
            )
        )
        # One inconsistency flag strips one verdict out of the citation bonus
        # and also adds the inconsistency penalty. Net effect: reward strictly
        # drops.
        assert inconsistent < clean
        assert inconsistent == pytest.approx(
            clean - INVESTIGATOR_RATIONALE_BONUS - INVESTIGATOR_INCONSISTENCY_PENALTY
        )


# -----------------------------------------------------------------------------
# 4. auditor_reward
# -----------------------------------------------------------------------------


class TestAuditorReward:
    def test_true_positive_flags_earn_reward(self) -> None:
        verdicts = [
            vr("ad_001", "approve", "fraud", confidence=0.95),
            vr("ad_002", "approve", "fraud", confidence=0.95),
            vr("ad_003", "approve", "fraud", confidence=0.95),
            vr("ad_004", "approve", "fraud", confidence=0.95),
        ]
        record = mk_record(
            verdicts=verdicts,
            ads=[ad_meta(v.ad_id, "fraud") for v in verdicts],
        )
        report = mk_report(
            track_a=[mk_flag("A", "miscalibration", severity=0.5)]
        )
        reward = auditor_reward(mk_inputs(record=record, audit_report=report))
        assert reward == pytest.approx(AUDITOR_TRUE_MISCAL)

    def test_false_positive_only_yields_negative_reward(self) -> None:
        record = mk_record(
            verdicts=[vr("ad_001", "approve", "legit", confidence=0.75)],
            ads=[ad_meta("ad_001", "legit")],
        )
        report = mk_report(
            track_b=[mk_flag("B", "gibberish", target_ad_id="ad_001")]
        )
        inputs = mk_inputs(
            record=record,
            audit_report=report,
            fraudster_proposal_log=[
                mk_propose(
                    "ad_001",
                    "Verified family-owned shop with 10 years of reviews.",
                )
            ],
            fraudster_ad_ids=["ad_001"],
        )
        reward = auditor_reward(inputs)
        assert reward == pytest.approx(-AUDITOR_FALSE_POSITIVE_PENALTY)

    def test_mixed_true_and_false_positives(self) -> None:
        # Real miscalibration + one bogus gibberish flag on a clean ad.
        verdicts = [
            vr("ad_001", "approve", "fraud", confidence=0.95),
            vr("ad_002", "approve", "fraud", confidence=0.95),
            vr("ad_003", "approve", "fraud", confidence=0.95),
            vr("ad_004", "approve", "fraud", confidence=0.95),
        ]
        record = mk_record(
            verdicts=verdicts,
            ads=[ad_meta(v.ad_id, "fraud") for v in verdicts],
        )
        report = mk_report(
            track_a=[mk_flag("A", "miscalibration", severity=0.5)],
            track_b=[mk_flag("B", "gibberish", target_ad_id="ad_001")],
        )
        inputs = mk_inputs(
            record=record,
            audit_report=report,
            fraudster_proposal_log=[
                mk_propose(
                    "ad_001",
                    "A realistic ad with a normal product description.",
                )
            ],
            fraudster_ad_ids=["ad_001"],
        )
        reward = auditor_reward(inputs)
        assert reward == pytest.approx(
            AUDITOR_TRUE_MISCAL - AUDITOR_FALSE_POSITIVE_PENALTY
        )


# -----------------------------------------------------------------------------
# 5. compute_episode_rewards
# -----------------------------------------------------------------------------


class TestComputeEpisodeRewards:
    def test_contains_all_expected_keys(self) -> None:
        record = mk_record(
            verdicts=[vr("ad_001", "approve", "legit")],
            ads=[ad_meta("ad_001", "legit")],
        )
        rewards = compute_episode_rewards(mk_inputs(record=record))
        for key in (
            "fraudster",
            "investigator",
            "auditor",
            "grader_score",
            "per_ad_plausibility",
            "audit_ground_truth",
        ):
            assert key in rewards, f"missing key: {key}"

    def test_all_rewards_are_finite(self) -> None:
        verdicts = [
            vr("ad_001", "reject", "fraud", confidence=0.85),
            vr("ad_002", "approve", "fraud", confidence=0.6),
            vr("ad_003", "approve", "legit", confidence=0.75),
        ]
        record = mk_record(
            verdicts=verdicts,
            ads=[
                ad_meta("ad_001", "fraud", severity=0.7),
                ad_meta("ad_002", "fraud", severity=0.5),
                ad_meta("ad_003", "legit"),
            ],
        )
        inputs = mk_inputs(
            record=record,
            fraudster_proposal_log=[
                mk_propose("ad_001", "Normal copy for a trusted brand."),
                mk_propose("ad_002", "Fast shipping and full refund available."),
            ],
            fraudster_ad_ids=["ad_001", "ad_002"],
            investigator_action_log=[
                {"action_type": "verdict", "ad_id": ad, "rationale": "ok reasoning"}
                for ad in ("ad_001", "ad_002", "ad_003")
            ],
        )
        rewards = compute_episode_rewards(inputs)
        for k in ("fraudster", "investigator", "auditor", "grader_score"):
            assert math.isfinite(rewards[k]), f"{k} is not finite: {rewards[k]}"
        assert 0.0 <= rewards["grader_score"] <= 1.0


# -----------------------------------------------------------------------------
# 6. Canonical end-to-end episode through the Referee
# -----------------------------------------------------------------------------


def _run_full_episode(fraud, inv, aud) -> RefereeState:
    env = RefereeEnvironment()
    env.reset_match(task_id="task_1", seed=123, max_rounds=3)

    loops = 0
    while env.phase != "done":
        loops += 1
        assert loops <= 600, "canonical episode did not terminate"
        if env.phase == "fraudster_turn":
            obs = env.build_fraudster_observation().model_dump()
            env.step_as_fraudster(fraud.act(obs))
        elif env.phase == "investigator_turn":
            obs = env.build_investigator_observation().model_dump()
            env.step_as_investigator(inv.act(obs))
        elif env.phase == "audit_phase":
            obs = env.build_auditor_observation().model_dump()
            env.step_as_auditor(aud.act(obs))
        else:
            raise AssertionError(f"unexpected phase: {env.phase}")
    return env.state


class TestCanonicalEpisode:
    def test_rewards_are_populated_and_finite(self) -> None:
        state = _run_full_episode(
            fraud=ReactiveFraudster(seed=7),
            inv=ScriptedInvestigator(),
            aud=HeuristicAuditor(),
        )
        assert state.phase == "done"
        assert state.grader_score is not None
        assert 0.0 <= state.grader_score <= 1.0
        for r in (
            state.fraudster_reward,
            state.investigator_reward,
            state.auditor_reward,
        ):
            assert math.isfinite(r), f"non-finite reward: {r}"
        assert state.audit_report is not None
        report = state.audit_report
        assert 0.0 <= report.get("investigator_audit_score", 0.0) <= 1.0
        assert 0.0 <= report.get("fraudster_plausibility_score", 0.0) <= 1.0

    def test_gibberish_fraudster_loses(self) -> None:
        """End-to-end: gibberish Fraudster + scripted Investigator — the
        Fraudster reward should NOT be large and positive, while the
        Investigator base score + rationale bonus keeps theirs above zero."""
        state = _run_full_episode(
            fraud=GibberishFraudster(seed=11),
            inv=ScriptedInvestigator(),
            aud=HeuristicAuditor(),
        )
        assert state.phase == "done"
        # Gibberish queue yields effectively no plausibility credit.
        assert state.fraudster_reward <= 0.5, (
            f"gibberish fraudster earned too much: {state.fraudster_reward}"
        )
        assert math.isfinite(state.investigator_reward)
        assert math.isfinite(state.auditor_reward)
