"""
Extended integrity metrics: fraud leaks, ring links, policy citations, holdout.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from counterfeint.data.real_world_loader import load_real_world_holdout
from counterfeint.experiments.episode_bundle import (
    EpisodeBundle,
    bundle_to_episode_record,
    run_episode_bundle,
)
from counterfeint.graders.base_grader import EpisodeRecord, LinkResult, VerdictResult


POLICY_CODE_RE = re.compile(r"FSDP-[A-Z]{2}-\d{2}")


@dataclass
class IntegrityMetrics:
    task_id: str
    seed: int
    grader_score: float
    n_fraud_leaks: int
    n_ground_truth_fraud: int
    link_precision: float
    link_recall: float
    citation_precision: float
    citation_recall: float
    holdout_eval: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "grader_score": round(self.grader_score, 4),
            "n_fraud_leaks": self.n_fraud_leaks,
            "n_ground_truth_fraud": self.n_ground_truth_fraud,
            "link_precision": round(self.link_precision, 4),
            "link_recall": round(self.link_recall, 4),
            "citation_precision": round(self.citation_precision, 4),
            "citation_recall": round(self.citation_recall, 4),
            "holdout_eval": self.holdout_eval,
        }


def _fraud_leaks(record: EpisodeRecord) -> int:
    leaks = 0
    for v in record.verdicts:
        if v.ground_truth == "fraud" and v.verdict in ("approve", "escalate"):
            leaks += 1
        if v.ground_truth == "fraud" and v.auto_approved:
            leaks += 1
    return leaks


def _link_metrics(record: EpisodeRecord) -> tuple[float, float]:
    if not record.links:
        return 0.0, 0.0
    tp = sum(1 for l in record.links if l.correct)
    fp = sum(1 for l in record.links if not l.correct)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    # Recall: fraction of ring pairs that should be linked (approx via correct links)
    recall = tp / len(record.links) if record.links else 0.0
    return precision, recall


def _citation_metrics(
    bundle: EpisodeBundle,
    record: EpisodeRecord,
) -> tuple[float, float]:
    """Precision/recall of FSDP-* codes in rationales vs ad fraud types."""
    cited: List[str] = []
    for act in bundle.investigator_actions:
        if act.get("action_type") != "verdict":
            continue
        rat = act.get("rationale") or ""
        cited.extend(POLICY_CODE_RE.findall(rat))

    expected: List[str] = []
    for meta in record.ads_metadata:
        if meta.get("ground_truth") == "fraud":
            # Placeholder: any citation counts as attempt
            expected.append("FSDP")

    if not cited:
        return 0.0, 0.0
    if not expected:
        return 1.0, 0.0
    # Loose metric: fraction of verdicts with any policy citation
    n_verdicts = sum(
        1 for a in bundle.investigator_actions if a.get("action_type") == "verdict"
    )
    n_with_cite = sum(
        1
        for a in bundle.investigator_actions
        if a.get("action_type") == "verdict"
        and POLICY_CODE_RE.search(a.get("rationale") or "")
    )
    precision = n_with_cite / len(cited) if cited else 0.0
    recall = n_with_cite / n_verdicts if n_verdicts else 0.0
    return precision, recall


def compute_integrity_metrics(bundle: EpisodeBundle) -> IntegrityMetrics:
    from counterfeint.graders.base_grader import grade_episode

    record = bundle_to_episode_record(bundle)
    grader = grade_episode(record)
    leaks = _fraud_leaks(record)
    n_fraud = sum(1 for m in record.ads_metadata if m.get("ground_truth") == "fraud")
    lp, lr = _link_metrics(record)
    cp, cr = _citation_metrics(bundle, record)
    return IntegrityMetrics(
        task_id=bundle.task_id,
        seed=bundle.seed,
        grader_score=grader,
        n_fraud_leaks=leaks,
        n_ground_truth_fraud=n_fraud,
        link_precision=lp,
        link_recall=lr,
        citation_precision=cp,
        citation_recall=cr,
    )


def run_integrity_suite(
    *,
    seeds_by_task: Dict[str, List[int]],
    output_dir: Path,
) -> List[IntegrityMetrics]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[IntegrityMetrics] = []

    for task_id, seeds in seeds_by_task.items():
        for seed in seeds:
            bundle = run_episode_bundle(task_id=task_id, seed=seed)
            rows.append(compute_integrity_metrics(bundle))

    # Holdout vignette count (no full episode — loader smoke)
    holdout = load_real_world_holdout(confirm_eval_only=True)
    holdout_meta = {
        "n_holdout_ads": len(holdout),
        "case_studies": list({h.case_study_source for h in holdout}),
    }

    payload = {
        "episodes": [r.to_dict() for r in rows],
        "holdout_meta": holdout_meta,
    }
    (output_dir / "integrity_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return rows


DEFAULT_INTEGRITY_SEEDS = {
    "task_1": [1001, 1002, 1003],
    "task_2": [2001, 2002],
    "task_3": [3001, 3002, 3003],
    "task_3_unseen": [4001, 4002],
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/integrity"),
    )
    args = parser.parse_args()
    run_integrity_suite(seeds_by_task=DEFAULT_INTEGRITY_SEEDS, output_dir=args.output)
