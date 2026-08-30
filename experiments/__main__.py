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
        default=Path("experiments/outputs/auditor_expanded/corruption"),
    )
    p_corr.add_argument("--no-cache", action="store_true")

    p_abl = sub.add_parser("ablation", help="Verifier anti-gaming ablation")
    p_abl.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded/ablation"),
    )
    p_abl.add_argument("--llm-flags", type=Path, default=None)
    p_abl.add_argument("--cache-dir", type=Path, default=None)

    p_eval = sub.add_parser("eval", help="Held-out baseline eval (in-process)")
    p_eval.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/eval"),
    )

    p_ann = sub.add_parser("annotation", help="Build blinded 50-case sheet")
    p_ann.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded/annotation"),
    )
    p_ann.add_argument("--cache-dir", type=Path, default=None)

    p_paper = sub.add_parser("paper-outputs", help="Write tables/figures")
    p_paper.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded"),
    )

    p_all = sub.add_parser("paper1", help="Run Paper 1 CPU suite + paper outputs")
    p_all.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded"),
    )
    p_all.add_argument("--no-cache", action="store_true")
    p_all.add_argument("--llm-flags", type=Path, default=None)

    args = parser.parse_args()
    root = getattr(args, "output", Path("experiments/outputs/auditor_expanded"))

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

        run_ablation_suite(
            seeds_by_task=DEFAULT_ABLATION_SEEDS,
            output_dir=args.output,
            llm_flags_path=args.llm_flags,
            cache_dir=args.cache_dir,
        )
    elif args.command == "eval":
        from counterfeint.experiments.run_eval import run_in_process_eval

        run_in_process_eval(output_dir=args.output)
    elif args.command == "annotation":
        from counterfeint.experiments.annotation_sheet import build_annotation_sheet

        build_annotation_sheet(output_dir=args.output, cache_dir=args.cache_dir)
    elif args.command == "paper-outputs":
        from counterfeint.experiments.paper_outputs import write_paper_outputs

        write_paper_outputs(
            output_dir=args.output,
            corruption_report=args.output / "corruption" / "corruption_report.json",
            ablation_results=args.output / "ablation" / "ablation_results.json",
            annotation_scores=(
                args.output / "annotation" / "annotation_scores.json"
                if (args.output / "annotation" / "annotation_scores.json").exists()
                else None
            ),
            llm_flags=(
                args.output / "llm_flags.json"
                if (args.output / "llm_flags.json").exists()
                else None
            ),
        )
    elif args.command == "paper1":
        from counterfeint.experiments.annotation_sheet import build_annotation_sheet
        from counterfeint.experiments.auditor_validation import (
            DEFAULT_VALIDATION_SEEDS,
            run_full_validation,
        )
        from counterfeint.experiments.paper_outputs import write_paper_outputs
        from counterfeint.experiments.verifier_ablation import (
            DEFAULT_ABLATION_SEEDS,
            run_ablation_suite,
        )

        corr_dir = root / "corruption"
        abl_dir = root / "ablation"
        run_full_validation(
            seeds_by_task=DEFAULT_VALIDATION_SEEDS,
            output_dir=corr_dir,
            reuse_cached=not args.no_cache,
        )
        run_ablation_suite(
            seeds_by_task=DEFAULT_ABLATION_SEEDS,
            output_dir=abl_dir,
            llm_flags_path=args.llm_flags,
            cache_dir=corr_dir / "base_trajectories",
        )
        build_annotation_sheet(
            output_dir=root / "annotation",
            cache_dir=corr_dir / "base_trajectories",
        )
        write_paper_outputs(
            output_dir=root,
            corruption_report=corr_dir / "corruption_report.json",
            ablation_results=abl_dir / "ablation_results.json",
            annotation_scores=(
                root / "annotation" / "annotation_scores.json"
                if (root / "annotation" / "annotation_scores.json").exists()
                else None
            ),
            llm_flags=args.llm_flags,
        )


if __name__ == "__main__":
    main()
