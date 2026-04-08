"""
Core Ad Fraud Investigation Environment.

Implements the OpenEnv Environment interface. The agent reviews a queue of ads,
investigates them, and renders verdicts under a budget constraint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..data.ad_generator import (
        TASK_CONFIGS,
        Ad,
        GeneratedEpisode,
        generate_episode,
    )
    from ..models import AdFraudState, AdReviewAction, AdReviewObservation
    from ..graders.base_grader import EpisodeRecord, LinkResult, VerdictResult, grade_episode
except ImportError:
    from data.ad_generator import (
        TASK_CONFIGS,
        Ad,
        GeneratedEpisode,
        generate_episode,
    )
    from models import AdFraudState, AdReviewAction, AdReviewObservation
    from graders.base_grader import EpisodeRecord, LinkResult, VerdictResult, grade_episode

logger = logging.getLogger(__name__)

# Module-level store so the /grader endpoint can read the last score.
_last_grader_result: Dict[str, Any] = {}


def get_last_grader_result() -> Dict[str, Any]:
    return dict(_last_grader_result)


class AdFraudEnvironment(
    Environment[AdReviewAction, AdReviewObservation, AdFraudState]
):
    """
    Ad fraud investigation environment.

    Each episode is a review session: the agent processes a queue of N ads
    within a limited action budget, choosing what to investigate and when
    to render verdicts. Unreviewed ads auto-approve at episode end.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._state = AdFraudState(episode_id=str(uuid4()), step_count=0)
        self._episode: Optional[GeneratedEpisode] = None
        self._verdicts: Dict[str, Dict[str, Any]] = {}
        self._links: List[Dict[str, Any]] = []
        self._investigations: Dict[str, List[str]] = {}
        self._cumulative_reward: float = 0.0
        self._done = False
        self._last_feedback = ""
        self._focused_ad_id: Optional[str] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> AdReviewObservation:
        task_id = kwargs.get("task_id", "task_1")
        if task_id not in TASK_CONFIGS:
            task_id = "task_1"

        effective_seed = seed if seed is not None else hash(uuid4()) & 0xFFFFFFFF
        self._episode = generate_episode(effective_seed, task_id)

        config = self._episode.task_config
        self._state = AdFraudState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            total_ads=config.queue_size,
            reviewed_count=0,
            remaining_budget=config.action_budget,
            verdicts={},
            grader_score=None,
        )
        self._verdicts = {}
        self._links = []
        self._investigations = {}
        self._cumulative_reward = 0.0
        self._done = False
        self._last_feedback = "Episode started. Review the ad queue and begin your investigation."
        self._focused_ad_id = self._episode.ads[0].ad_id if self._episode.ads else None

        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: AdReviewAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> AdReviewObservation:
        if self._done:
            return self._build_observation(
                reward=0.0, done=True,
                feedback_override="Episode is already complete. Call reset() to start a new episode.",
            )

        if self._episode is None:
            return self._build_observation(
                reward=0.0, done=False,
                feedback_override="Environment not initialized. Call reset() first.",
            )

        self._state.step_count += 1

        ad_ids = {a.ad_id for a in self._episode.ads}
        if action.ad_id not in ad_ids:
            self._last_feedback = f"Invalid ad_id '{action.ad_id}'. Valid IDs: {', '.join(sorted(ad_ids))}"
            return self._build_observation(reward=-0.05, done=False)

        if action.action_type == "investigate":
            reward = self._handle_investigate(action)
        elif action.action_type == "verdict":
            reward = self._handle_verdict(action)
        elif action.action_type == "link_accounts":
            reward = self._handle_link(action)
        else:
            self._last_feedback = f"Unknown action_type '{action.action_type}'."
            reward = -0.05

        self._cumulative_reward += reward

        done = self._check_done()
        if done and not self._done:
            end_reward = self._handle_episode_end()
            reward += end_reward
            self._cumulative_reward += end_reward
            self._done = True

        self._state.remaining_budget = max(0, self._state.remaining_budget)
        self._state.reviewed_count = len(self._verdicts)
        self._state.verdicts = {
            ad_id: v.get("verdict", "") for ad_id, v in self._verdicts.items()
        }

        return self._build_observation(reward=reward, done=self._done)

    @property
    def state(self) -> AdFraudState:
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_investigate(self, action: AdReviewAction) -> float:
        if self._state.remaining_budget <= 0:
            self._last_feedback = "No budget remaining. You must render verdicts on remaining ads or end the episode."
            return -0.02

        if action.investigation_target is None:
            self._last_feedback = "investigation_target is required for action_type='investigate'."
            return -0.05

        ad_id = action.ad_id
        target = action.investigation_target

        prev = self._investigations.setdefault(ad_id, [])
        if target in prev:
            self._last_feedback = (
                f"You already investigated '{target}' for {ad_id}. "
                "Choose a different investigation target or render a verdict."
            )
            return -0.02

        if ad_id in self._verdicts:
            self._last_feedback = f"You already rendered a verdict on {ad_id}. Choose a different ad."
            return -0.02

        self._state.remaining_budget -= 1
        prev.append(target)
        self._focused_ad_id = ad_id

        findings = self._episode.investigation_data.get(ad_id, {}).get(
            target, "No data available for this investigation type."
        )
        self._last_feedback = (
            f"Investigation complete: {target} for {ad_id}.\n"
            f"--- Findings ---\n{findings}"
        )
        return -0.02

    def _handle_verdict(self, action: AdReviewAction) -> float:
        ad_id = action.ad_id

        if ad_id in self._verdicts:
            self._last_feedback = f"You already rendered a verdict on {ad_id}."
            return -0.02

        if action.verdict is None:
            self._last_feedback = "verdict field is required for action_type='verdict'."
            return -0.05

        confidence = action.confidence if action.confidence is not None else 0.5
        ad = self._get_ad(ad_id)
        ground_truth = ad.ground_truth_label if ad else "legit"
        severity = ad.severity if ad else 0.0

        self._verdicts[ad_id] = {
            "verdict": action.verdict,
            "confidence": confidence,
            "ground_truth": ground_truth,
        }

        reward = self._compute_verdict_reward(action.verdict, ground_truth, severity, confidence)

        pending = [a.ad_id for a in self._episode.ads if a.ad_id not in self._verdicts]
        self._last_feedback = (
            f"Verdict recorded for {ad_id}: {action.verdict} "
            f"(confidence: {confidence:.2f}). "
            f"{len(pending)} ad(s) remaining in queue."
        )

        if pending:
            self._focused_ad_id = pending[0]

        return reward

    def _handle_link(self, action: AdReviewAction) -> float:
        if action.linked_ad_id is None:
            self._last_feedback = "linked_ad_id is required for action_type='link_accounts'."
            return -0.05

        ad_ids = {a.ad_id for a in self._episode.ads}
        if action.linked_ad_id not in ad_ids:
            self._last_feedback = f"Invalid linked_ad_id '{action.linked_ad_id}'."
            return -0.05

        if action.ad_id == action.linked_ad_id:
            self._last_feedback = "Cannot link an ad to itself."
            return -0.05

        link_key = tuple(sorted([action.ad_id, action.linked_ad_id]))
        existing = {tuple(sorted([l["ad_id_1"], l["ad_id_2"]])) for l in self._links}
        if link_key in existing:
            self._last_feedback = f"Link between {action.ad_id} and {action.linked_ad_id} already recorded."
            return -0.02

        is_correct = self._check_link_correct(action.ad_id, action.linked_ad_id)

        self._links.append({
            "ad_id_1": action.ad_id,
            "ad_id_2": action.linked_ad_id,
            "reason": action.link_reason or "",
            "correct": is_correct,
        })

        self._last_feedback = (
            f"Network link recorded: {action.ad_id} <-> {action.linked_ad_id}. "
            f"Reason: {action.link_reason or 'not specified'}."
        )

        return 0.4 if is_correct else -0.25

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_verdict_reward(
        self, verdict: str, ground_truth: str, severity: float, confidence: float
    ) -> float:
        if verdict == "reject" and ground_truth == "fraud":
            return 0.3 + 0.1 * severity
        elif verdict == "approve" and ground_truth == "legit":
            return 0.1
        elif verdict == "escalate" and ground_truth == "escalate":
            return 0.15
        elif verdict == "reject" and ground_truth == "legit":
            return -0.35
        elif verdict == "approve" and ground_truth == "fraud":
            return -0.5
        elif verdict == "escalate":
            return -0.05
        elif verdict == "approve" and ground_truth == "escalate":
            return -0.15
        elif verdict == "reject" and ground_truth == "escalate":
            return -0.1
        else:
            return -0.05

    def _handle_episode_end(self) -> float:
        """Apply end-of-episode adjustments for unreviewed ads, then delegate to graders."""
        unreviewed_fraud = 0
        for ad in self._episode.ads:
            if ad.ad_id not in self._verdicts:
                self._verdicts[ad.ad_id] = {
                    "verdict": "approve",
                    "confidence": 0.0,
                    "ground_truth": ad.ground_truth_label,
                    "auto_approved": True,
                }
                if ad.ground_truth_label == "fraud":
                    unreviewed_fraud += 1

        record = self._build_episode_record()
        grader_score = grade_episode(record)
        self._state.grader_score = grader_score

        reviewed_count = len([v for v in self._verdicts.values() if not v.get("auto_approved")])
        total_ads = len(self._episode.ads)
        total_correct = sum(
            1 for v in self._verdicts.values()
            if not v.get("auto_approved")
            and (
                (v["verdict"] == "reject" and v["ground_truth"] == "fraud")
                or (v["verdict"] == "approve" and v["ground_truth"] == "legit")
                or (v["verdict"] == "escalate" and v["ground_truth"] == "escalate")
            )
        )
        false_positives = sum(
            1 for v in self._verdicts.values()
            if not v.get("auto_approved")
            and v["verdict"] == "reject" and v["ground_truth"] == "legit"
        )
        false_negatives = sum(
            1 for v in self._verdicts.values()
            if not v.get("auto_approved")
            and v["verdict"] == "approve" and v["ground_truth"] == "fraud"
        )
        correct_links = sum(1 for l in self._links if l.get("correct"))
        incorrect_links = sum(1 for l in self._links if not l.get("correct"))

        global _last_grader_result
        _last_grader_result = {
            "task_id": self._state.task_id,
            "grader_score": grader_score,
            "episode_id": self._state.episode_id,
            "total_steps": self._state.step_count,
            "verdicts_rendered": reviewed_count,
            "correct_decisions": total_correct,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "auto_approved": total_ads - reviewed_count,
            "unreviewed_fraud": unreviewed_fraud,
            "network_links_correct": correct_links,
            "network_links_incorrect": incorrect_links,
        }

        feedback_lines = [
            f"Episode complete. Grader score: {grader_score:.3f}/1.000",
            f"Verdicts rendered: {reviewed_count}/{total_ads}",
            f"Correct decisions: {total_correct}/{reviewed_count}",
            f"False positives (legit rejected): {false_positives}",
            f"False negatives (fraud approved): {false_negatives}",
            f"Unreviewed ads auto-approved: {unreviewed_fraud}",
        ]
        if self._links:
            feedback_lines.append(
                f"Network links: {correct_links} correct, {incorrect_links} incorrect"
            )
        self._last_feedback = "\n".join(feedback_lines)

        return 0.0

    def _build_episode_record(self) -> EpisodeRecord:
        """Convert internal state into an EpisodeRecord for the grader."""
        verdict_results = []
        for ad in self._episode.ads:
            v = self._verdicts.get(ad.ad_id)
            if v:
                verdict_results.append(VerdictResult(
                    ad_id=ad.ad_id,
                    verdict=v["verdict"],
                    confidence=v.get("confidence", 0.5),
                    ground_truth=v["ground_truth"],
                    auto_approved=v.get("auto_approved", False),
                ))

        link_results = [
            LinkResult(ad_id_1=l["ad_id_1"], ad_id_2=l["ad_id_2"], correct=l["correct"])
            for l in self._links
        ]

        ads_metadata = [
            {"ad_id": ad.ad_id, "ground_truth": ad.ground_truth_label, "severity": ad.severity}
            for ad in self._episode.ads
        ]

        return EpisodeRecord(
            task_id=self._state.task_id,
            total_steps=self._state.step_count,
            action_budget=self._episode.task_config.action_budget,
            verdicts=verdict_results,
            links=link_results,
            ads_metadata=ads_metadata,
            n_fraud_rings=len(self._episode.fraud_rings),
            ring_sizes=[len(r.member_ad_ids) for r in self._episode.fraud_rings],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_done(self) -> bool:
        if self._episode is None:
            return True
        all_reviewed = all(
            ad.ad_id in self._verdicts for ad in self._episode.ads
        )
        steps_exhausted = self._state.step_count >= self._episode.task_config.action_budget
        return all_reviewed or steps_exhausted

    def _check_link_correct(self, ad_id_1: str, ad_id_2: str) -> bool:
        """Check if two ads share a fraud ring."""
        for ring in self._episode.fraud_rings:
            if ad_id_1 in ring.member_ad_ids and ad_id_2 in ring.member_ad_ids:
                return True
        return False

    def _get_ad(self, ad_id: str) -> Optional[Ad]:
        if self._episode is None:
            return None
        for ad in self._episode.ads:
            if ad.ad_id == ad_id:
                return ad
        return None

    def _build_observation(
        self,
        reward: float,
        done: bool,
        feedback_override: str | None = None,
    ) -> AdReviewObservation:
        feedback = feedback_override or self._last_feedback

        if self._episode is None:
            return AdReviewObservation(
                done=done,
                reward=reward,
                queue_summary="No episode loaded.",
                current_ad_info="",
                investigation_findings="",
                verdict_history_summary="",
                feedback=feedback,
                available_ads=[],
                queue_status={},
            )

        config = self._episode.task_config
        pending = [a for a in self._episode.ads if a.ad_id not in self._verdicts]
        reviewed = [a for a in self._episode.ads if a.ad_id in self._verdicts]

        steps_remaining = max(0, config.action_budget - self._state.step_count)
        queue_summary = (
            f"Task: {config.name} ({config.difficulty})\n"
            f"Total ads: {config.queue_size} | "
            f"Reviewed: {len(reviewed)} | "
            f"Pending: {len(pending)} | "
            f"Steps remaining: {steps_remaining}/{config.action_budget} | "
            f"Investigation budget: {self._state.remaining_budget} | "
            f"Step: {self._state.step_count}"
        )

        current_ad_info = ""
        if self._focused_ad_id and not done:
            ad = self._get_ad(self._focused_ad_id)
            if ad and ad.ad_id not in self._verdicts:
                signals = ", ".join(ad.initial_risk_signals) if ad.initial_risk_signals else "None"
                investigated = self._investigations.get(ad.ad_id, [])
                inv_status = ", ".join(investigated) if investigated else "None yet"

                # Contextual metadata visible before investigation
                profile = self._episode.advertiser_profiles.get(ad.ad_id)
                meta_lines = []
                if profile:
                    meta_lines.append(f"Advertiser country: {profile.country}")
                    meta_lines.append(f"Account age: {profile.account_age_days} days")
                    if profile.account_age_days < 30:
                        meta_lines.append("Flag: New account (< 30 days)")
                context_meta = "\n".join(meta_lines)

                current_ad_info = (
                    f"=== Ad in Focus: {ad.ad_id} ===\n"
                    f"Category: {ad.category}\n"
                    f"Ad copy: \"{ad.ad_copy}\"\n"
                    f"Targeting: {ad.targeting_summary}\n"
                    f"Initial risk signals: {signals}\n"
                    f"{context_meta}\n"
                    f"Investigations completed: {inv_status}\n"
                    f"Available investigation targets: advertiser_history, landing_page, "
                    f"payment_method, targeting_overlap, creative_similarity, campaign_structure"
                )

        investigation_findings = ""
        for ad_id, targets in self._investigations.items():
            for target in targets:
                finding = self._episode.investigation_data.get(ad_id, {}).get(target, "")
                if finding:
                    investigation_findings += f"\n[{ad_id} / {target}]\n{finding}\n"

        manual_verdicts = {
            ad_id: v for ad_id, v in self._verdicts.items()
            if not v.get("auto_approved")
        }
        if manual_verdicts:
            counts = {"approve": 0, "reject": 0, "escalate": 0}
            by_decision = {"approve": [], "reject": [], "escalate": []}
            for ad_id, v in manual_verdicts.items():
                counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
                by_decision[v["verdict"]].append(ad_id)
            summary_parts = [f"{c} {k}" for k, c in counts.items() if c > 0]
            verdict_lines = [
                f"Reviewed {len(manual_verdicts)} ad(s): {', '.join(summary_parts)}."
            ]
            for decision in ("reject", "approve", "escalate"):
                if by_decision[decision]:
                    verdict_lines.append(
                        f"  {decision}: {', '.join(by_decision[decision])}"
                    )
            verdict_history_summary = "\n".join(verdict_lines)
        else:
            verdict_history_summary = "No verdicts yet."

        available_ads = [a.ad_id for a in pending]

        queue_status = {
            "total_ads": config.queue_size,
            "reviewed": len(reviewed),
            "pending": len(pending),
            "investigation_budget": self._state.remaining_budget,
            "steps_remaining": steps_remaining,
            "step": self._state.step_count,
            "task_id": config.task_id,
        }

        return AdReviewObservation(
            done=done,
            reward=reward,
            queue_summary=queue_summary,
            current_ad_info=current_ad_info,
            investigation_findings=investigation_findings.strip(),
            verdict_history_summary=verdict_history_summary,
            feedback=feedback,
            available_ads=available_ads,
            queue_status=queue_status,
        )
