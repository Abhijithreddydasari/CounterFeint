"""
Async vLLM client for batched investigator generation.

Requires vLLM server with OpenAI-compatible API. Use guided JSON for
schema-valid actions.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore


INVESTIGATOR_ACTION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["investigate", "verdict", "link_accounts"],
        },
        "ad_id": {"type": "string"},
        "investigation_target": {"type": "string"},
        "verdict": {"type": "string", "enum": ["approve", "reject", "escalate"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
        "linked_ad_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["action_type"],
}


@dataclass
class VLLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    api_key: str = "EMPTY"
    max_tokens: int = 256
    temperature: float = 0.7
    concurrency: int = 64


class VLLMInvestigatorClient:
    """Thin async wrapper around vLLM OpenAI-compatible completions."""

    def __init__(self, config: Optional[VLLMConfig] = None) -> None:
        if AsyncOpenAI is None:
            raise ImportError("openai package required for VLLMInvestigatorClient")
        self.config = config or VLLMConfig(
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            model=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        )
        self._client = AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
        )
        self._sem = asyncio.Semaphore(self.config.concurrency)

    async def generate_one(self, prompt: str) -> str:
        async with self._sem:
            resp = await self._client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                extra_body={
                    "guided_json": INVESTIGATOR_ACTION_SCHEMA,
                },
            )
            return resp.choices[0].message.content or ""

    async def generate_batch(self, prompts: List[str]) -> List[str]:
        return await asyncio.gather(*(self.generate_one(p) for p in prompts))

    def parse_action(self, completion: str) -> Dict[str, Any]:
        return json.loads(completion)


def benchmark_throughput(
    *,
    n_prompts: int = 20,
    config: Optional[VLLMConfig] = None,
) -> Dict[str, float]:
    """Quick throughput check; returns prompts/sec."""
    import time

    client = VLLMInvestigatorClient(config)
    dummy = "You are an ad fraud investigator. Respond with JSON action."

    async def _run() -> float:
        t0 = time.perf_counter()
        await client.generate_batch([dummy] * n_prompts)
        return time.perf_counter() - t0

    elapsed = asyncio.run(_run())
    return {
        "n_prompts": n_prompts,
        "elapsed_sec": elapsed,
        "prompts_per_sec": n_prompts / elapsed if elapsed else 0.0,
    }
