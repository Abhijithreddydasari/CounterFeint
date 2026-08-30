"""
Anti-gaming ablations: both roles, track removal, per-check, LLM, hybrid.

Reports reward gained by an attack versus its paired clean trajectory.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from counterfeint.models import AuditFlag

from .corruptions import FRAUDSTER_ATTACKS, INVESTIGATOR_ATTACKS, apply_corruption
from .episode_bundle import EpisodeBundle, load_bundle, run_episode_bundle, save_bundle
from .reward_arms import AblationArm, compute_role_rewards, run_audit

DETERMINISTIC_ARMS = [
    AblationArm.FULL,
    AblationArm.NO_TRACK_A,
    AblationArm.NO_TRACK_B,
    AblationArm.NO_PLAUSIBILITY_GATE,
    AblationArm.OUTCOME_ONLY,
    AblationArm.NO_MISCALIBRATION,
    AblationArm.NO_MISSING_CITATION,
    AblationArm.NO_INCOHERENT,
    AblationArm.NO_INCONSISTENCY,
    AblationArm.NO_BIAS,
    AblationArm.NO_GIBBERISH,
    AblationArm.NO_PARAMETER_MISMATCH,
    AblationArm.NO_TEMPLATE_REPETITION,
    AblationArm.NO_BRANDING_ANOMALY,
]


@dataclass
class AblationRow:
    task_id: str
    seed: int
    corruption: str
    side: str
    arm: str
    fraudster_reward: float
    investigator_reward: float
    grader_score: float
    mean_plausibility: float
    delta_investigator: float
    delta_fraudster: float
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_llm_flags(path: Optional[Path], task_id: str, seed: int, corr: str) -> List[AuditFlag]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = f"{task_id}:{seed}:{corr}"
    raw = payload.get(key) or payload.get("flags", {}).get(key) or []
    flags: List[AuditFlag] = []
    for item in raw:
        flags.append(
            AuditFlag(
                track=item.get("track", "A"),
                target_ad_id=item.get("target_ad_id"),
                flag_type=item.get("flag_type", "missing_citation"),
                severity=float(item.get("severity", 0.5) or 0.5),
                note=str(item.get("note", "llm"))[:2000],
            )
        )
    return flags


def run_ablation_suite(
    *,
    seeds_by_task: Dict[str, List[int]],
    output_dir: Path,
    llm_flags_path: Optional[Path] = None,
    include_llm_arms: bool = True,
    reuse_cached: bool = True,
    cache_dir: Optional[Path] = None,
) -> List[AblationRow]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or (output_dir / "base_trajectories")
    rows: List[AblationRow] = []

    attacks = INVESTIGATOR_ATTACKS + FRAUDSTER_ATTACKS
    arms = list(DETERMINISTIC_ARMS)
    has_llm = llm_flags_path is not None and llm_flags_path.exists()
    if include_llm_arms:
        arms.append(AblationArm.HYBRID)
        if has_llm:
            arms.append(AblationArm.LLM_ONLY)

    for task_id, seeds in seeds_by_task.items():
        for seed in seeds:
            cache_path = cache_dir / f"{task_id}_seed{seed}.json"
            if reuse_cached and cache_path.exists():
                base = load_bundle(cache_path)
            else:
                base = run_episode_bundle(task_id=task_id, seed=seed)
                save_bundle(base, cache_path)

            clean_audit = run_audit(base)
            t0 = time.perf_counter()
            clean_full = compute_role_rewards(base, AblationArm.FULL, audit=clean_audit)
            clean_ms = (time.perf_counter() - t0) * 1000.0

            for corr_name in ["clean"] + attacks:
                bundle = base if corr_name == "clean" else apply_corruption(base, corr_name)
                if corr_name != "clean":
                    bundle.corruption_label = corr_name
                audit = clean_audit if corr_name == "clean" else run_audit(bundle)
                llm_flags = _load_llm_flags(llm_flags_path, task_id, seed, corr_name)

                for arm in arms:
                    t1 = time.perf_counter()
                    extra = None
                    if arm in (AblationArm.LLM_ONLY, AblationArm.HYBRID):
                        extra = llm_flags
                    rw = compute_role_rewards(
                        bundle, arm, llm_flags=extra, audit=audit
                    )
                    elapsed = (time.perf_counter() - t1) * 1000.0
                    if corr_name == "clean" and arm == AblationArm.FULL:
                        elapsed = clean_ms
                    rows.append(
                        AblationRow(
                            task_id=task_id,
                            seed=seed,
                            corruption=corr_name,
                            side=bundle.side or "clean",
                            arm=arm.value,
                            fraudster_reward=rw.fraudster_reward,
                            investigator_reward=rw.investigator_reward,
                            grader_score=rw.grader_score,
                            mean_plausibility=rw.mean_plausibility,
                            delta_investigator=rw.investigator_reward - clean_full.investigator_reward,
                            delta_fraudster=rw.fraudster_reward - clean_full.fraudster_reward,
                            elapsed_ms=elapsed,
                        )
                    )

    payload = [r.to_dict() for r in rows]
    (output_dir / "ablation_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_ablation_md(rows, output_dir / "ablation_summary.md")
    return rows


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _write_ablation_md(rows: List[AblationRow], path: Path) -> None:
    # Mean investigator/fraudster reward by corruption × key arms
    key_arms = [
        "full",
        "no_track_a",
        "no_track_b",
        "no_plausibility_gate",
        "outcome_only",
        "hybrid",
        "llm_only",
    ]
    by: Dict[str, Dict[str, List[AblationRow]]] = {}
    for r in rows:
        by.setdefault(r.corruption, {}).setdefault(r.arm, []).append(r)

    lines = [
        "# Verifier Anti-Gaming Ablation",
        "",
        "Delta = attacked reward − paired clean FULL reward. "
        "Positive Investigator delta on fabricated rationales with `no_track_a` "
        "means Track A was blocking the exploit. Positive Fraudster delta on "
        "gibberish with `no_plausibility_gate` means Track B was blocking it.",
        "",
        "## Investigator reward (mean)",
        "",
        "| Attack | " + " | ".join(key_arms) + " |",
        "|--------|" + "|".join(["-----:" for _ in key_arms]) + "|",
    ]
    for corr in sorted(by.keys()):
        cells = [corr]
        for arm in key_arms:
            vals = [x.investigator_reward for x in by[corr].get(arm, [])]
            cells.append(f"{_mean(vals):.3f}" if vals else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Fraudster reward (mean)",
            "",
            "| Attack | " + " | ".join(key_arms) + " |",
            "|--------|" + "|".join(["-----:" for _ in key_arms]) + "|",
        ]
    )
    for corr in sorted(by.keys()):
        cells = [corr]
        for arm in key_arms:
            vals = [x.fraudster_reward for x in by[corr].get(arm, [])]
            cells.append(f"{_mean(vals):.3f}" if vals else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Attack reward gain vs clean (full vs track-removed)",
            "",
            "| Attack | side | Δinv full | Δinv no_A | Δfrd full | Δfrd no_B/gate |",
            "|--------|------|----------:|----------:|----------:|---------------:|",
        ]
    )
    for corr in sorted(by.keys()):
        if corr == "clean":
            continue
        side = by[corr].get("full", [AblationRow("", 0, corr, "?", "", 0, 0, 0, 0, 0, 0, 0)])[0].side
        inv_full = _mean([x.delta_investigator for x in by[corr].get("full", [])])
        inv_na = _mean([x.delta_investigator for x in by[corr].get("no_track_a", [])])
        frd_full = _mean([x.delta_fraudster for x in by[corr].get("full", [])])
        frd_nb = _mean(
            [x.delta_fraudster for x in by[corr].get("no_plausibility_gate", [])]
        )
        lines.append(
            f"| {corr} | {side} | {inv_full:.3f} | {inv_na:.3f} | {frd_full:.3f} | {frd_nb:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# Ablation uses 10+10+10 = 30 episodes (fast enough; validation uses 100)
DEFAULT_ABLATION_SEEDS: Dict[str, List[int]] = {
    "task_1": list(range(11, 21)),
    "task_2": list(range(11, 21)),
    "task_3": list(range(11, 21)),
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run verifier ablation suite")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/ablation"),
    )
    parser.add_argument("--llm-flags", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    run_ablation_suite(
        seeds_by_task=DEFAULT_ABLATION_SEEDS,
        output_dir=args.output,
        llm_flags_path=args.llm_flags,
        cache_dir=args.cache_dir,
    )
