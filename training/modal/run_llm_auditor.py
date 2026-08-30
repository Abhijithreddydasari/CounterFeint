"""
Modal job: serve Qwen3.5-9B with vLLM and judge saved episode bundles.

Usage (from counterfeint/, Modal secrets `huggingface-secret` or `huggingface` must contain HF_TOKEN):
  python -m modal run training/modal/run_llm_auditor.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import modal
except ImportError:
    modal = None  # type: ignore

def _counterfeint_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *list(here.parents)]:
        if (p / "experiments" / "llm_auditor.py").exists():
            return p
        if (p / "counterfeint" / "experiments" / "llm_auditor.py").exists():
            return p / "counterfeint"
    return Path("/root/counterfeint")


COUNTERFEINT = _counterfeint_root()
REPO_ROOT = COUNTERFEINT.parent

if modal is not None:
    app = modal.App("counterfeint-llm-auditor")
    # devel image provides nvcc; debian_slim + pip vLLM has CUDA runtime but
    # flashinfer JIT-compiles sampling kernels and dies without /usr/local/cuda.
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            add_python="3.11",
        )
        .entrypoint([])
        .pip_install(
            "vllm>=0.8.0",
            "openai>=1.0.0",
            "torch",
            "transformers",
            "pydantic>=2.0",
            "faker==33.1.0",
            "networkx>=3.2",
            "fastapi",
            "openenv-core>=0.2.3",
            "uvicorn",
            "requests",
            "python-dotenv",
        )
        .env(
            {
                "PYTHONPATH": "/root",
                "CUDA_HOME": "/usr/local/cuda",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                "TRANSFORMERS_VERBOSITY": "error",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        .add_local_dir(
            str(COUNTERFEINT),
            remote_path="/root/counterfeint",
            ignore=[
                "**/outputs/**",
                "**/__pycache__/**",
                "**/.pytest_cache/**",
                "**/*.pyc",
            ],
        )
    )

    @app.function(
        gpu="A100-80GB",
        timeout=3600,
        image=image,
        memory=65536,
        secrets=[
            modal.Secret.from_name("huggingface-secret"),
            modal.Secret.from_name("huggingface"),
        ],
    )
    def judge_cached_bundles(
        bundles_payload: List[Dict[str, Any]],
        model: str = "Qwen/Qwen3.5-9B",
    ) -> Dict[str, Any]:
        """Start vLLM, judge each bundle, return typed flags + latency."""
        os.environ["CUDA_HOME"] = "/usr/local/cuda"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
        if not token:
            return {
                "error": "HF_TOKEN missing. Put it in Modal secret huggingface-secret (or huggingface).",
                "flags": {},
            }
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        child_env = os.environ.copy()
        child_env["CUDA_HOME"] = "/usr/local/cuda"
        child_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        child_env["HF_TOKEN"] = token
        child_env["HUGGING_FACE_HUB_TOKEN"] = token
        child_env["TRANSFORMERS_VERBOSITY"] = "error"
        child_env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        cuda_bin = "/usr/local/cuda/bin"
        child_env["PATH"] = cuda_bin + os.pathsep + child_env.get("PATH", "")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model,
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "16384",
                "--port",
                "8000",
                "--disable-log-requests",
                "--default-chat-template-kwargs",
                '{"enable_thinking": false}',
            ],
            env=child_env,
        )
        os.environ["VLLM_JUDGE_MODEL"] = model
        os.environ["VLLM_BASE_URL"] = "http://127.0.0.1:8000/v1"
        os.environ["VLLM_API_KEY"] = "EMPTY"
        deadline = time.time() + 900
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                return {
                    "error": f"vLLM exited {proc.returncode} before serving",
                    "flags": {},
                }
            try:
                import urllib.request

                urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2)
                ready = True
                break
            except Exception:
                time.sleep(5)
        if not ready:
            proc.kill()
            return {"error": "vLLM failed to start", "flags": {}}

        sys.path.insert(0, "/root")
        from counterfeint.experiments.episode_bundle import EpisodeBundle
        from counterfeint.experiments.llm_auditor import judge_bundle_async

        import asyncio

        results: Dict[str, List[Dict[str, Any]]] = {}
        latencies: List[float] = []
        raw_previews: Dict[str, str] = {}

        async def _run() -> None:
            for item in bundles_payload:
                bundle = EpisodeBundle.from_dict(item["bundle"])
                key = item["key"]
                try:
                    flags, elapsed, raw = await judge_bundle_async(bundle)
                except Exception as exc:  # noqa: BLE001
                    results[key] = [{"error": str(exc)}]
                    continue
                latencies.append(elapsed)
                if len(raw_previews) < 8:
                    raw_previews[key] = raw
                results[key] = [
                    {
                        "track": f.track,
                        "flag_type": f.flag_type,
                        "target_ad_id": f.target_ad_id,
                        "severity": f.severity,
                        "note": f.note,
                    }
                    for f in flags
                ]

        asyncio.run(_run())
        proc.terminate()
        mean_lat = sum(latencies) / len(latencies) if latencies else None
        # A100-80GB ~ $2.10/hr on Modal; cost ≈ mean_lat * hourly / 3600
        cost = None if mean_lat is None else mean_lat * (2.10 / 3600.0)
        n_flagged = sum(1 for v in results.values() if v and not v[0].get("error"))
        return {
            "model": model,
            "n_cases": len(bundles_payload),
            "n_ok": len(latencies),
            "n_flagged": n_flagged,
            "mean_latency_sec": mean_lat,
            "cost_per_episode_usd": cost,
            "raw_previews": raw_previews,
            "flags": results,
        }


def _load_local_targets(cache_dir: Path, max_bases: int = 30) -> List[Dict[str, Any]]:
    sys.path.insert(0, str(COUNTERFEINT.parent))
    from counterfeint.experiments.corruptions import (
        FRAUDSTER_ATTACKS,
        INVESTIGATOR_ATTACKS,
        apply_corruption,
    )
    from counterfeint.experiments.episode_bundle import load_bundle

    attacks = INVESTIGATOR_ATTACKS + FRAUDSTER_ATTACKS
    payload: List[Dict[str, Any]] = []
    paths = sorted(cache_dir.glob("*.json"))[:max_bases]
    for path in paths:
        base = load_bundle(path)
        payload.append(
            {
                "key": f"{base.task_id}:{base.seed}:clean",
                "bundle": base.to_dict(),
            }
        )
        for name in attacks:
            corrupted = apply_corruption(base, name)
            payload.append(
                {
                    "key": f"{base.task_id}:{base.seed}:{name}",
                    "bundle": corrupted.to_dict(),
                }
            )
    return payload


def main() -> None:
    ann = COUNTERFEINT / "experiments" / "outputs" / "auditor_expanded" / "annotation" / "bundles"
    payload: List[Dict[str, Any]]
    if ann.exists():
        sys.path.insert(0, str(COUNTERFEINT.parent))
        from counterfeint.experiments.episode_bundle import load_bundle

        payload = []
        for path in sorted(ann.glob("C*.json")):
            bundle = load_bundle(path)
            payload.append({"key": path.stem, "bundle": bundle.to_dict()})
    else:
        cache = (
            COUNTERFEINT
            / "experiments"
            / "outputs"
            / "auditor_expanded"
            / "corruption"
            / "base_trajectories"
        )
        payload = _load_local_targets(cache, max_bases=10)
    if modal is None:
        raise SystemExit("modal is not installed")
    result = judge_cached_bundles.remote(payload)  # type: ignore[name-defined]
    out = COUNTERFEINT / "experiments" / "outputs" / "auditor_expanded" / "llm_flags.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out} n={result.get('n_ok')} mean_lat={result.get('mean_latency_sec')}")


if modal is not None:
    main = app.local_entrypoint()(main)

