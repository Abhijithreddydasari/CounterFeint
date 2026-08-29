"""
Model bake-off: compare investigator candidates on a small held-out slice.

Run on Modal with vLLM after server is up. Falls back to in-process
ScriptedInvestigator for smoke tests without GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from counterfeint.experiments.run_eval import run_in_process_eval
from counterfeint.scripted import ScriptedInvestigator


BAKEOFF_MODELS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3.5-4B",
]

BAKEOFF_SEEDS = {
    "task_1": [1001, 1002, 1003, 1004, 1005],
    "task_2": [2001, 2002, 2003, 2004, 2005],
    "task_3": [3001, 3002, 3003, 3004, 3005],
}


def run_bakeoff_smoke(output_dir: Path) -> Dict[str, Any]:
    """CPU smoke: scripted investigator baseline on 15 episodes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_in_process_eval(
        output_dir=output_dir / "scripted_smoke",
        seeds=BAKEOFF_SEEDS,
        tag="scripted_smoke",
        investigator_factory=lambda: ScriptedInvestigator(),
    )
    manifest = {
        "models": BAKEOFF_MODELS,
        "note": (
            "Full LLM bake-off requires vLLM on Modal. "
            "Run modal/run_bakeoff.py after vLLM server is healthy."
        ),
        "scripted_smoke": result.get("aggregates", {}),
    }
    (output_dir / "bakeoff_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/bakeoff"),
    )
    args = parser.parse_args()
    print(json.dumps(run_bakeoff_smoke(args.output), indent=2))
