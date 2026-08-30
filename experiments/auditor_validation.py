"""
Paired clean-vs-corrupted Track A/B evaluation.

Naturally occurring flags on the unmodified trajectory are NOT counted as
corruption false positives. Matching is at (flag_type, target_ad_id).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from counterfeint.graders.auditor_pipeline import run_full_audit

from .corruptions import FRAUDSTER_ATTACKS, INVESTIGATOR_ATTACKS, apply_corruption, sanitize_bundle
from .episode_bundle import (
    EpisodeBundle,
    bundle_to_episode_record,
    load_bundle,
    run_episode_bundle,
    save_bundle,
)

FlagKey = Tuple[str, Optional[str]]


@dataclass
class FlagMetrics:
    flag_type: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision_ci: Tuple[float, float] = (0.0, 0.0)
    recall_ci: Tuple[float, float] = (0.0, 0.0)
    f1_ci: Tuple[float, float] = (0.0, 0.0)

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
            "precision_ci95": [round(self.precision_ci[0], 4), round(self.precision_ci[1], 4)],
            "recall_ci95": [round(self.recall_ci[0], 4), round(self.recall_ci[1], 4)],
            "f1_ci95": [round(self.f1_ci[0], 4), round(self.f1_ci[1], 4)],
        }


@dataclass
class ValidationReport:
    n_base_episodes: int = 0
    n_corruptions: int = 0
    n_applicable: int = 0
    per_flag: Dict[str, FlagMetrics] = field(default_factory=dict)
    per_corruption: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    clean_flag_rate: Dict[str, float] = field(default_factory=dict)
    trial_rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_base_episodes": self.n_base_episodes,
            "n_corruptions": self.n_corruptions,
            "n_applicable": self.n_applicable,
            "per_flag": {k: v.to_dict() for k, v in self.per_flag.items()},
            "per_corruption": self.per_corruption,
            "clean_flag_rate": self.clean_flag_rate,
        }


def audit_flag_keys(bundle: EpisodeBundle) -> Set[FlagKey]:
    record = bundle_to_episode_record(bundle)
    result = run_full_audit(
        record=record,
        investigator_action_log=bundle.investigator_actions,
        investigation_data_seen=bundle.investigation_data_seen,
        fraudster_proposal_log=bundle.fraudster_proposals,
    )
    keys: Set[FlagKey] = set()
    for f in list(result.track_a_flags) + list(result.track_b_flags):
        keys.add((f.flag_type, f.target_ad_id))
    return keys


def audit_flag_types(bundle: EpisodeBundle) -> Set[str]:
    return {k[0] for k in audit_flag_keys(bundle)}


def _expected_keys(bundle: EpisodeBundle) -> Set[FlagKey]:
    keys: Set[FlagKey] = set()
    for item in bundle.expected_flag_keys or []:
        if not item:
            continue
        ft = str(item[0])
        ad = item[1] if len(item) > 1 else None
        keys.add((ft, ad))
    if not keys and bundle.expected_flags:
        for ft in bundle.expected_flags:
            keys.add((ft, None))
    return keys


def paired_delta(
    clean_keys: Set[FlagKey],
    corr_keys: Set[FlagKey],
    expected: Set[FlagKey],
) -> Tuple[Set[FlagKey], Set[FlagKey], Set[FlagKey]]:
    """Return (tp, fp, fn) using NEW flags vs expected, dropping pre-existing expected."""
    new_keys = corr_keys - clean_keys
    applicable = set()
    for ft, ad in expected:
        # If expected is type-only (ad is None), any new flag of that type counts.
        if ad is None:
            if not any(ck[0] == ft for ck in clean_keys):
                applicable.add((ft, None))
            elif any(ck[0] == ft and ck not in clean_keys for ck in corr_keys):
                applicable.add((ft, None))
        else:
            if (ft, ad) not in clean_keys:
                applicable.add((ft, ad))
    tp: Set[FlagKey] = set()
    for ft, ad in applicable:
        if ad is None:
            if any(nk[0] == ft for nk in new_keys) or any(
                ck[0] == ft for ck in corr_keys
            ):
                tp.add((ft, None))
        elif (ft, ad) in new_keys or (ft, ad) in corr_keys:
            tp.add((ft, ad))
    fn = applicable - tp
    # FP: new flags whose type is not in the expected set
    expected_types = {ft for ft, _ in expected}
    fp = {k for k in new_keys if k[0] not in expected_types}
    return tp, fp, fn


def validate_bundle(bundle: EpisodeBundle) -> tuple[Set[str], Set[str]]:
    """Absolute type-level check (used by unit tests)."""
    expected = set(bundle.expected_flags)
    fired = audit_flag_types(bundle)
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


def _bootstrap_ci(
    values: List[float],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[random.Random] = None,
) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = rng or random.Random(0)
    n = len(values)
    means: List[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def run_corruption_validation(
    bundles: List[EpisodeBundle],
    *,
    corruptions: Optional[Iterable[str]] = None,
) -> ValidationReport:
    names = list(corruptions) if corruptions is not None else list(
        INVESTIGATOR_ATTACKS + FRAUDSTER_ATTACKS
    )
    report = ValidationReport(n_base_episodes=len(bundles))
    metrics: Dict[str, FlagMetrics] = {}
    per_corr_hits: Dict[str, List[int]] = {n: [] for n in names}
    clean_type_counts: Dict[str, int] = {}
    trials: List[Dict[str, Any]] = []

    # per-flag lists of precision/recall contributions for bootstrap
    flag_trials: Dict[str, List[Tuple[int, int, int]]] = {}

    for base in bundles:
        natural_keys = audit_flag_keys(base)
        for ft in {k[0] for k in natural_keys}:
            clean_type_counts[ft] = clean_type_counts.get(ft, 0) + 1
        control = sanitize_bundle(base)
        clean_keys = audit_flag_keys(control)
        for corr_name in names:
            corrupted = apply_corruption(control, corr_name)
            report.n_corruptions += 1
            expected = _expected_keys(corrupted)
            if not expected and not corrupted.expected_flags:
                per_corr_hits[corr_name].append(0)
                continue
            report.n_applicable += 1
            corr_keys = audit_flag_keys(corrupted)
            tp, fp, fn = paired_delta(clean_keys, corr_keys, expected)
            hit = 1 if tp else 0
            per_corr_hits[corr_name].append(hit)

            types = {ft for ft, _ in expected} | {k[0] for k in tp | fp | fn}
            for ft in types:
                m = metrics.setdefault(ft, FlagMetrics(flag_type=ft))
                t_tp = sum(1 for k in tp if k[0] == ft)
                t_fp = sum(1 for k in fp if k[0] == ft)
                t_fn = sum(1 for k in fn if k[0] == ft)
                m.tp += t_tp
                m.fp += t_fp
                m.fn += t_fn
                flag_trials.setdefault(ft, []).append((t_tp, t_fp, t_fn))

            trials.append(
                {
                    "task_id": base.task_id,
                    "seed": base.seed,
                    "corruption": corr_name,
                    "side": corrupted.side,
                    "hit": hit,
                    "tp": [list(k) for k in tp],
                    "fp": [list(k) for k in fp],
                    "fn": [list(k) for k in fn],
                }
            )

    rng = random.Random(0)
    for ft, m in metrics.items():
        rows = flag_trials.get(ft, [])
        precs, recs, f1s = [], [], []
        for _ in range(1000):
            if not rows:
                break
            sample = [rows[rng.randrange(len(rows))] for _ in rows]
            tp = sum(r[0] for r in sample)
            fp = sum(r[1] for r in sample)
            fn = sum(r[2] for r in sample)
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            precs.append(p)
            recs.append(r)
            f1s.append(f)
        m.precision_ci = _bootstrap_ci(precs, rng=rng) if precs else (0.0, 0.0)
        m.recall_ci = _bootstrap_ci(recs, rng=rng) if recs else (0.0, 0.0)
        m.f1_ci = _bootstrap_ci(f1s, rng=rng) if f1s else (0.0, 0.0)

    n_base = max(1, len(bundles))
    report.clean_flag_rate = {
        ft: round(c / n_base, 4) for ft, c in sorted(clean_type_counts.items())
    }
    report.per_flag = metrics
    report.trial_rows = trials
    for name, hits in per_corr_hits.items():
        n = len(hits) or 1
        report.per_corruption[name] = {
            "n": len(hits),
            "hit_rate": round(sum(hits) / n, 4),
        }
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
    (output_dir / "corruption_trials.json").write_text(
        json.dumps(report.trial_rows, indent=2), encoding="utf-8"
    )
    _write_summary_md(report, output_dir / "corruption_summary.md")
    return report


def _write_summary_md(report: ValidationReport, path: Path) -> None:
    lines = [
        "# Auditor Corruption Validation (paired delta)",
        "",
        f"- Base episodes: {report.n_base_episodes}",
        f"- Corruption trials: {report.n_corruptions}",
        f"- Applicable (non-empty expected flags): {report.n_applicable}",
        "",
        "## Per-flag metrics (new flags vs expected; 95% bootstrap CI)",
        "",
        "| Flag | TP | FP | FN | Precision | Recall | F1 |",
        "|------|---:|---:|---:|----------:|-------:|---:|",
    ]
    for ft in sorted(report.per_flag.keys()):
        m = report.per_flag[ft]
        lines.append(
            f"| {ft} | {m.tp} | {m.fp} | {m.fn} | "
            f"{m.precision:.3f} [{m.precision_ci[0]:.2f},{m.precision_ci[1]:.2f}] | "
            f"{m.recall:.3f} [{m.recall_ci[0]:.2f},{m.recall_ci[1]:.2f}] | "
            f"{m.f1:.3f} [{m.f1_ci[0]:.2f},{m.f1_ci[1]:.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Clean-control flag rate (share of unmodified episodes with flag)",
            "",
        ]
    )
    for ft, rate in report.clean_flag_rate.items():
        lines.append(f"- `{ft}`: {rate:.3f}")
    lines.extend(["", "## Corruption hit rate (new expected flag fired)", ""])
    for name, stats in sorted(report.per_corruption.items()):
        lines.append(
            f"- `{name}`: {stats['hit_rate']:.3f} ({stats['n']} trials)"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# 40 + 35 + 25 = 100 stratified trajectories
DEFAULT_VALIDATION_SEEDS: Dict[str, List[int]] = {
    "task_1": list(range(11, 51)),
    "task_2": list(range(11, 46)),
    "task_3": list(range(11, 36)),
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
