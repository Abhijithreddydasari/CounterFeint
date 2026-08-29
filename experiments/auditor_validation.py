"""
Measure Track A/B auditor flag precision, recall, and F1 against corruption labels.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set

from counterfeint.graders.auditor_pipeline import run_full_audit

from .corruptions import CORRUPTION_REGISTRY, apply_corruption
from .episode_bundle import (
    EpisodeBundle,
    bundle_to_episode_record,
    load_bundle,
    run_episode_bundle,
    save_bundle,
)


@dataclass
class FlagMetrics:
    flag_type: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flag_type": self.flag_type,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class ValidationReport:
    n_base_episodes: int = 0
    n_corruptions: int = 0
    per_flag: Dict[str, FlagMetrics] = field(default_factory=dict)
    corruption_hits: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_base_episodes": self.n_base_episodes,
            "n_corruptions": self.n_corruptions,
            "per_flag": {k: v.to_dict() for k, v in self.per_flag.items()},
            "corruption_hits": dict(self.corruption_hits),
        }


def _audit_flags(bundle: EpisodeBundle) -> Set[str]:
    record = bundle_to_episode_record(bundle)
    result = run_full_audit(
        record=record,
        investigator_action_log=bundle.investigator_actions,
        investigation_data_seen=bundle.investigation_data_seen,
        fraudster_proposal_log=bundle.fraudster_proposals,
    )
    fired = {f.flag_type for f in result.track_a_flags + result.track_b_flags}
    return fired


def _update_metrics(
    metrics: Dict[str, FlagMetrics],
    expected: Set[str],
    fired: Set[str],
) -> None:
    all_types = expected | fired
    for ft in all_types:
        m = metrics.setdefault(ft, FlagMetrics(flag_type=ft))
        if ft in expected and ft in fired:
            m.tp += 1
        elif ft in fired and ft not in expected:
            m.fp += 1
        elif ft in expected and ft not in fired:
            m.fn += 1


def validate_bundle(bundle: EpisodeBundle) -> tuple[Set[str], Set[str]]:
    expected = set(bundle.expected_flags)
    fired = _audit_flags(bundle)
    return expected, fired


def collect_base_trajectories(
    *,
    seeds_by_task: Dict[str, List[int]],
    out_dir: Path,
) -> List[EpisodeBundle]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles: List[EpisodeBundle] = []
    for task_id, seeds in seeds_by_task.items():
        for seed in seeds:
            bundle = run_episode_bundle(task_id=task_id, seed=seed)
            path = out_dir / f"{task_id}_seed{seed}.json"
            save_bundle(bundle, path)
            bundles.append(bundle)
    return bundles


def run_corruption_validation(
    bundles: List[EpisodeBundle],
) -> ValidationReport:
    report = ValidationReport(n_base_episodes=len(bundles))
    metrics: Dict[str, FlagMetrics] = {}

    for base in bundles:
        for corr_name in CORRUPTION_REGISTRY:
            corrupted = apply_corruption(base, corr_name)
            report.n_corruptions += 1
            expected, fired = validate_bundle(corrupted)
            _update_metrics(metrics, expected, fired)
            if expected & fired:
                report.corruption_hits[corr_name] = (
                    report.corruption_hits.get(corr_name, 0) + 1
                )

    report.per_flag = metrics
    return report


def run_full_validation(
    *,
    seeds_by_task: Dict[str, List[int]],
    output_dir: Path,
    reuse_cached: bool = True,
) -> ValidationReport:
    base_dir = output_dir / "base_trajectories"
    bundles: List[EpisodeBundle] = []

    if reuse_cached and base_dir.exists():
        for path in sorted(base_dir.glob("*.json")):
            bundles.append(load_bundle(path))

    if not bundles:
        bundles = collect_base_trajectories(
            seeds_by_task=seeds_by_task, out_dir=base_dir
        )

    report = run_corruption_validation(bundles)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "corruption_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )
    _write_summary_md(report, output_dir / "corruption_summary.md")
    return report


def _write_summary_md(report: ValidationReport, path: Path) -> None:
    lines = [
        "# Auditor Corruption Validation",
        "",
        f"- Base episodes: {report.n_base_episodes}",
        f"- Corruption trials: {report.n_corruptions}",
        "",
        "## Per-flag metrics",
        "",
        "| Flag | TP | FP | FN | Precision | Recall | F1 |",
        "|------|---:|---:|---:|----------:|-------:|---:|",
    ]
    for ft in sorted(report.per_flag.keys()):
        m = report.per_flag[ft]
        lines.append(
            f"| {ft} | {m.tp} | {m.fp} | {m.fn} | "
            f"{m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} |"
        )
    lines.extend(["", "## Corruption hit rate", ""])
    for name, hits in sorted(report.corruption_hits.items()):
        lines.append(f"- `{name}`: {hits} episodes with ≥1 expected flag fired")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


DEFAULT_VALIDATION_SEEDS: Dict[str, List[int]] = {
    "task_1": [11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
    "task_2": [11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
    "task_3": [11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run auditor corruption validation")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/corruption"),
    )
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    report = run_full_validation(
        seeds_by_task=DEFAULT_VALIDATION_SEEDS,
        output_dir=args.output,
        reuse_cached=not args.no_cache,
    )
    print(json.dumps(report.to_dict(), indent=2))
