"""
Dataset split registry and coverage reporting.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def load_manifest(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_table(manifest: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, int]]:
    """Count axis values per split for paper tables."""
    out: Dict[str, Dict[str, int]] = {}
    for split_name, specs in manifest.items():
        counter: Counter = Counter()
        for spec in specs:
            for axis in (
                "violation_type",
                "industry",
                "geography",
                "ring_topology",
                "budget_regime",
            ):
                counter[f"{axis}:{spec.get(axis, '')}"] += 1
        out[split_name] = dict(counter)
    return out


def write_coverage_report(manifest_path: Path, output_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    coverage = coverage_table(manifest)
    lines = ["# Dataset Axis Coverage", ""]
    for split_name, counts in coverage.items():
        lines.append(f"## {split_name} ({len(manifest.get(split_name, []))} scenarios)")
        lines.append("")
        for key in sorted(counts.keys()):
            lines.append(f"- {key}: {counts[key]}")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/outputs/data/dataset_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/data/coverage_report.md"),
    )
    args = parser.parse_args()
    write_coverage_report(args.manifest, args.output)
