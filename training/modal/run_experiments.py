"""
Modal deployment for vLLM + CounterFeint experiments.

Usage (from repo root):
  modal run counterfeint/training/modal/run_experiments.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[3]

if modal is not None:
    app = modal.App("counterfeint-experiments")

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "vllm>=0.8.0",
            "openai>=1.0.0",
            "torch",
            "transformers",
            "accelerate",
            "datasets",
            "trl",
            "peft",
            "bitsandbytes",
            "faker==33.1.0",
            "networkx>=3.2",
            "pydantic>=2.0",
            "fastapi",
        )
        .pip_install("-e", str(REPO_ROOT / "counterfeint"))
    )

    @app.function(gpu="A100-80GB", timeout=3600, image=image)
    def run_corruption_and_ablation() -> str:
        """Paper 1 experiments (CPU portion, no vLLM needed)."""
        cmds = [
            [sys.executable, "-m", "counterfeint.experiments", "corruption"],
            [sys.executable, "-m", "counterfeint.experiments", "ablation"],
            [
                sys.executable,
                "-m",
                "counterfeint.experiments.integrity_metrics",
            ],
        ]
        for cmd in cmds:
            subprocess.check_call(cmd, cwd=str(REPO_ROOT))
        return "done"

    @app.function(gpu="A100-80GB", timeout=7200, image=image)
    def run_vllm_benchmark(
        model: str = "Qwen/Qwen3-4B-Instruct-2507",
        n_prompts: int = 20,
    ) -> dict:
        """Start vLLM and benchmark throughput."""
        import time

        proc = subprocess.Popen(
            [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model,
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "8192",
                "--port",
                "8000",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(120)
            from counterfeint.training.vllm_client import benchmark_throughput

            return benchmark_throughput(n_prompts=n_prompts)
        finally:
            proc.terminate()

    @app.local_entrypoint()
    def main() -> None:
        print(run_corruption_and_ablation.remote())
        print(run_vllm_benchmark.remote())
