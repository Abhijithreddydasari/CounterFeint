"""CLI entrypoints for CounterFeint experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="CounterFeint experiment runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_corr = sub.add_parser("corruption", help="Auditor corruption validation")
    p_corr.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/corruption"),
    )
    p_corr.add_argument("--no-cache", action="store_true")

    p_abl = sub.add_parser("ablation", help="Verifier anti-gaming ablation")
    p_abl.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/ablation"),
    )

    p_eval = sub.add_parser("eval", help="Held-out baseline eval (in-process)")
    p_eval.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/eval"),
    )

    args = parser.parse_args()

    if args.command == "corruption":
        from counterfeint.experiments.auditor_validation import (
            DEFAULT_VALIDATION_SEEDS,
            run_full_validation,
        )

        run_full_validation(
            seeds_by_task=DEFAULT_VALIDATION_SEEDS,
            output_dir=args.output,
            reuse_cached=not args.no_cache,
        )
    elif args.command == "ablation":
        from counterfeint.experiments.verifier_ablation import (
            DEFAULT_ABLATION_SEEDS,
            run_ablation_suite,
        )

        run_ablation_suite(seeds_by_task=DEFAULT_ABLATION_SEEDS, output_dir=args.output)
    elif args.command == "eval":
        from counterfeint.experiments.run_eval import run_in_process_eval

        run_in_process_eval(output_dir=args.output)


if __name__ == "__main__":
    main()
