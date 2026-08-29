"""
Anti-gaming ablation: measure reward under weakened verifier arms.

Arms:
  - full: default multi-agent rewards
  - no_track_a: strip Track A flags before investigator reward
  - no_track_b: force plausibility = 1.0
  - no_plausibility_gate: fraudster gets severity without plausibility multiply
  - outcome_only: grader_score only for investigator
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from counterfeint.graders.auditor_pipeline import run_full_audit
from counterfeint.graders.base_grader import grade_episode
from counterfeint.graders.multi_agent_rewards import (
    FRAUDSTER_BANNED_PENALTY,
    FRAUDSTER_PER_AD_SEVERITY_WEIGHT,
    INVESTIGATOR_INCONSISTENCY_CAP,
    INVESTIGATOR_INCONSISTENCY_PENALTY,
    INVESTIGATOR_RATIONALE_BONUS,
    INVESTIGATOR_RATIONALE_FLAG_TYPES,
    RewardInputs,
    build_reward_cache,
)
from counterfeint.models import AuditFlag, AuditReport

from .corruptions import apply_corruption
from .episode_bundle import EpisodeBundle, bundle_to_episode_record, run_episode_bundle


class AblationArm(str, Enum):
    FULL = "full"
    NO_TRACK_A = "no_track_a"
    NO_TRACK_B = "no_track_b"
    NO_PLAUSIBILITY_GATE = "no_plausibility_gate"
    OUTCOME_ONLY = "outcome_only"


@dataclass
class AblationResult:
    arm: str
    corruption: str
    fraudster_reward: float
    investigator_reward: float
    grader_score: float
    mean_plausibility: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "corruption": self.corruption,
            "fraudster_reward": round(self.fraudster_reward, 4),
            "investigator_reward": round(self.investigator_reward, 4),
            "grader_score": round(self.grader_score, 4),
            "mean_plausibility": round(self.mean_plausibility, 4),
        }


def _compute_rewards(
    bundle: EpisodeBundle,
    arm: AblationArm,
) -> AblationResult:
    record = bundle_to_episode_record(bundle)
    audit = run_full_audit(
        record=record,
        investigator_action_log=bundle.investigator_actions,
        investigation_data_seen=bundle.investigation_data_seen,
        fraudster_proposal_log=bundle.fraudster_proposals,
    )

    track_a = list(audit.track_a_flags)
    track_b = list(audit.track_b_flags)
    per_ad_plaus = dict(audit.per_ad_plausibility)

    if arm == AblationArm.NO_TRACK_A:
        track_a = []
    if arm in (AblationArm.NO_TRACK_B, AblationArm.NO_PLAUSIBILITY_GATE):
        per_ad_plaus = {k: 1.0 for k in per_ad_plaus}
        track_b = []

    audit_report = AuditReport(
        track_a_flags=track_a,
        track_b_flags=track_b,
        investigator_audit_score=audit.investigator_audit_score,
        fraudster_plausibility_score=audit.fraudster_plausibility_score,
        notes="ablation",
    )

    grader_score = grade_episode(record)

    if arm == AblationArm.OUTCOME_ONLY:
        return AblationResult(
            arm=arm.value,
            corruption=bundle.corruption_label or "clean",
            fraudster_reward=0.0,
            investigator_reward=grader_score,
            grader_score=grader_score,
            mean_plausibility=sum(per_ad_plaus.values()) / max(1, len(per_ad_plaus)),
        )

    # Fraudster reward
    severity_by_ad = {
        m.get("ad_id", ""): float(m.get("severity", 0.5) or 0.5)
        for m in record.ads_metadata
    }
    verdict_by_ad = {v.ad_id: v.verdict for v in record.verdicts}
    fraudster_ads = set(bundle.fraudster_ad_ids)
    unbanned = [
        ad_id
        for ad_id in fraudster_ads
        if verdict_by_ad.get(ad_id) != "reject"
    ]
    fraud_r = 0.0
    for ad_id in unbanned:
        sev = severity_by_ad.get(ad_id, 0.6)
        plaus = per_ad_plaus.get(ad_id, 0.0)
        if arm == AblationArm.NO_PLAUSIBILITY_GATE:
            fraud_r += FRAUDSTER_PER_AD_SEVERITY_WEIGHT * sev
        else:
            fraud_r += FRAUDSTER_PER_AD_SEVERITY_WEIGHT * sev * plaus
    banned = sum(1 for ad_id in fraudster_ads if verdict_by_ad.get(ad_id) == "reject")
    fraud_r -= FRAUDSTER_BANNED_PENALTY * banned

    # Investigator reward
    flagged_ids = {
        f.target_ad_id
        for f in track_a
        if f.target_ad_id and f.flag_type in INVESTIGATOR_RATIONALE_FLAG_TYPES
    }
    weighted_clean = 0.0
    for v in record.verdicts:
        if v.ad_id in flagged_ids:
            continue
        weighted_clean += per_ad_plaus.get(v.ad_id, 1.0)
    n_incons = min(
        sum(1 for f in track_a if f.flag_type == "inconsistency"),
        INVESTIGATOR_INCONSISTENCY_CAP,
    )
    inv_r = (
        grader_score
        + INVESTIGATOR_RATIONALE_BONUS * weighted_clean
        - INVESTIGATOR_INCONSISTENCY_PENALTY * n_incons
    )

    mean_plaus = (
        sum(per_ad_plaus.values()) / len(per_ad_plaus) if per_ad_plaus else 1.0
    )

    return AblationResult(
        arm=arm.value,
        corruption=bundle.corruption_label or "clean",
        fraudster_reward=fraud_r,
        investigator_reward=inv_r,
        grader_score=grader_score,
        mean_plausibility=mean_plaus,
    )


def run_ablation_suite(
    *,
    seeds_by_task: Dict[str, List[int]],
    output_dir: Path,
) -> List[AblationResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[AblationResult] = []

    corruption_names = [
        "clean",
        "gibberish_ads",
        "grader_trigger_tokens",
        "missing_citation",
        "contradictory_rationale",
    ]

    for task_id, seeds in seeds_by_task.items():
        for seed in seeds[:5]:  # cap per task for speed
            base = run_episode_bundle(task_id=task_id, seed=seed)
            for corr_name in corruption_names:
                bundle = base if corr_name == "clean" else apply_corruption(base, corr_name)
                if corr_name != "clean":
                    bundle.corruption_label = corr_name
                for arm in AblationArm:
                    results.append(_compute_rewards(bundle, arm))

    payload = [r.to_dict() for r in results]
    (output_dir / "ablation_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_ablation_md(results, output_dir / "ablation_summary.md")
    return results


def _write_ablation_md(results: List[AblationResult], path: Path) -> None:
    by_corr_arm: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        key = r.corruption
        by_corr_arm.setdefault(key, {}).setdefault(r.arm, []).append(r.fraudster_reward)

    lines = [
        "# Verifier Anti-Gaming Ablation",
        "",
        "Mean fraudster reward by corruption × arm (higher on gibberish = gaming succeeds).",
        "",
        "| Corruption | full | no_track_b | no_plausibility_gate |",
        "|------------|-----:|-----------:|---------------------:|",
    ]
    for corr in sorted(by_corr_arm.keys()):
        arms = by_corr_arm[corr]
        def _mean(k: str) -> str:
            vals = arms.get(k, [])
            return f"{sum(vals)/len(vals):.3f}" if vals else "—"
        lines.append(
            f"| {corr} | {_mean('full')} | {_mean('no_track_b')} | "
            f"{_mean('no_plausibility_gate')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


DEFAULT_ABLATION_SEEDS: Dict[str, List[int]] = {
    "task_1": [11, 13, 17, 19, 23],
    "task_2": [11, 13, 17],
    "task_3": [11, 13, 17],
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run verifier ablation suite")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/ablation"),
    )
    args = parser.parse_args()
    run_ablation_suite(seeds_by_task=DEFAULT_ABLATION_SEEDS, output_dir=args.output)
