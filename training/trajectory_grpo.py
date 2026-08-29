"""
Trajectory-group GRPO: roll G complete trajectories per seed, score with
environment reward, compute group-relative advantages.

This replaces snapshot-level proxy GRPO for long-horizon credit assignment.
Proxy reward remains available as an ablation via ``reward_mode``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from counterfeint.experiments.episode_bundle import run_episode_bundle
from counterfeint.experiments.episode_bundle import bundle_to_episode_record
from counterfeint.graders.base_grader import grade_episode
from counterfeint.graders.multi_agent_rewards import RewardInputs, compute_episode_rewards
from counterfeint.models import AuditReport
from counterfeint.scripted import HeuristicAuditor, ReactiveFraudster


class RewardMode(str, Enum):
    ENVIRONMENT = "environment"
    GRADER_ONLY = "grader_only"
    PROXY = "proxy"  # delegated to proxy_reward for ablation


@dataclass
class TrajectoryRecord:
    task_id: str
    seed: int
    group_id: str
    trajectory_idx: int
    grader_score: float
    investigator_reward: float
    advantage: float = 0.0
    tokens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "group_id": self.group_id,
            "trajectory_idx": self.trajectory_idx,
            "grader_score": self.grader_score,
            "investigator_reward": self.investigator_reward,
            "advantage": self.advantage,
            "n_tokens": len(self.tokens),
        }


def _score_trajectory(
    *,
    task_id: str,
    seed: int,
    investigator_factory: Callable[[], Any],
    reward_mode: RewardMode,
) -> tuple[float, float]:
    bundle = run_episode_bundle(
        task_id=task_id,
        seed=seed,
        investigator_factory=investigator_factory,
        fraudster_factory=lambda: ReactiveFraudster(seed=42),
        auditor_factory=lambda: HeuristicAuditor(),
    )
    record = bundle_to_episode_record(bundle)
    grader = grade_episode(record)

    if reward_mode == RewardMode.GRADER_ONLY:
        return grader, grader

    audit = AuditReport(
        track_a_flags=[],
        track_b_flags=[],
        investigator_audit_score=1.0,
        fraudster_plausibility_score=1.0,
        notes="",
    )
    rewards = compute_episode_rewards(
        RewardInputs(
            record=record,
            audit_report=audit,
            fraudster_proposal_log=bundle.fraudster_proposals,
            investigator_action_log=bundle.investigator_actions,
            investigation_data_seen=bundle.investigation_data_seen,
            fraudster_ad_ids=bundle.fraudster_ad_ids,
        )
    )
    return grader, float(rewards["investigator"])


def compute_group_advantages(scores: List[float]) -> List[float]:
    """Group-relative advantage: score - mean(group)."""
    if not scores:
        return []
    mean = sum(scores) / len(scores)
    return [s - mean for s in scores]


def collect_trajectory_group(
    *,
    task_id: str,
    seed: int,
    group_size: int,
    investigator_factory: Callable[[], Any],
    reward_mode: RewardMode = RewardMode.ENVIRONMENT,
) -> List[TrajectoryRecord]:
    """Roll ``group_size`` trajectories; same seed, different investigator stochasticity."""
    group_id = f"{task_id}_{seed}"
    records: List[TrajectoryRecord] = []

    for idx in range(group_size):
        # Vary per-trajectory seed offset for stochastic policies
        grader, inv_r = _score_trajectory(
            task_id=task_id,
            seed=seed + idx * 1000,
            investigator_factory=investigator_factory,
            reward_mode=reward_mode,
        )
        records.append(
            TrajectoryRecord(
                task_id=task_id,
                seed=seed,
                group_id=group_id,
                trajectory_idx=idx,
                grader_score=grader,
                investigator_reward=inv_r,
            )
        )

    advantages = compute_group_advantages([r.investigator_reward for r in records])
    for rec, adv in zip(records, advantages):
        rec.advantage = adv
    return records


def run_trajectory_grpo_collection(
    *,
    seeds_by_task: Dict[str, List[int]],
    group_size: int = 4,
    investigator_factory: Callable[[], Any],
    reward_mode: RewardMode = RewardMode.ENVIRONMENT,
    output_dir: Path,
) -> List[TrajectoryRecord]:
    """Collect trajectory groups for GRPO training dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: List[TrajectoryRecord] = []

    for task_id, seeds in seeds_by_task.items():
        for seed in seeds:
            group = collect_trajectory_group(
                task_id=task_id,
                seed=seed,
                group_size=group_size,
                investigator_factory=investigator_factory,
                reward_mode=reward_mode,
            )
            all_records.extend(group)

    payload = [r.to_dict() for r in all_records]
    (output_dir / "trajectory_groups.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return all_records


# Integration point for TRL GRPOTrainer:
# 1. collect_trajectory_group -> tokenize each trajectory's action sequence
# 2. assign advantage to every token in trajectory
# 3. pass to GRPOTrainer with custom reward_fn returning precomputed advantage
