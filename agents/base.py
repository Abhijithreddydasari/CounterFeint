"""
Shared infrastructure for LLM-backed role policies.

The :class:`LLMPolicyBase` class gives every LLM policy:

  * A bounded chat-completion call (retries + hard timeout).
  * JSON parsing + Pydantic validation of the assistant message.
  * A deterministic ``fallback_policy`` that is called on every type of
    failure — API error, timeout, JSON decode error, validation error — so
    RL training rollouts never see the policy crash mid-episode.
  * A per-episode ``fallback_count`` integer so harnesses can surface the
    LLM health of each run (:func:`counterfeint.inference.log_end_r2`
    prints it in the ``[END]`` line).

The concrete LLM policies (:class:`.llm_fraudster.LLMFraudster`,
:class:`.llm_investigator.LLMInvestigator`) subclass this and only implement:

  * :attr:`system_prompt` (class attribute string).
  * :attr:`action_model` — the Pydantic ``BaseModel`` subclass to validate
    the raw JSON response against (``FraudsterAction`` / ``AdReviewAction``).
  * :meth:`_build_user_prompt` to assemble the per-turn user message from the
    current observation.
  * :meth:`_log_name` — role name for debug logging (``"fraudster"`` /
    ``"investigator"``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from ..scripted._base import PolicyBase


logger = logging.getLogger(__name__)


# Matches an optional ```json fenced block OR just raw JSON; group(1) is the
# payload without fences. Reused from inference._extract_json's regex style.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def _extract_json_text(raw: str) -> str:
    """Pull JSON out of a possibly-markdown-fenced LLM response."""
    text = (raw or "").strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        return "\n".join(lines).strip()
    return text


class LLMCallError(Exception):
    """Raised when :meth:`LLMPolicyBase._call_llm_with_retries` exhausts retries."""


class LLMPolicyBase(PolicyBase):
    """Abstract LLM-backed role policy.

    Subclasses MUST set the following class attributes:

    - ``system_prompt``  (str)
    - ``action_model``   (Type[BaseModel])

    Subclasses MUST implement:

    - ``_build_user_prompt(observation) -> str``
    - ``_log_name`` (class attribute str, e.g. ``"fraudster"``)

    The constructor accepts the same API envs inference.py already uses
    (``API_BASE_URL``, ``MODEL_NAME``, ``HF_TOKEN``), so an Ollama deployment
    at ``http://localhost:11434/v1`` works out of the box.
    """

    # ------------------------------------------------------------------
    # Subclass-provided
    # ------------------------------------------------------------------
    system_prompt: str = ""
    action_model: Optional[Type[BaseModel]] = None
    _log_name: str = "llm_policy"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        fallback_policy: PolicyBase,
        model_name: Optional[str] = None,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 384,
        timeout_s: float = 5.0,
        retries: int = 2,
        client: Optional[Any] = None,
    ) -> None:
        if self.system_prompt == "" or self.action_model is None:
            raise TypeError(
                f"{type(self).__name__} must set `system_prompt` and `action_model`"
            )

        self.fallback_policy = fallback_policy
        self.model_name = model_name or os.getenv(
            "MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct"
        )
        self.api_base_url = api_base_url or os.getenv(
            "API_BASE_URL", "https://router.huggingface.co/v1"
        )
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)

        self.fallback_count: int = 0
        self.call_count: int = 0
        self.last_error: Optional[str] = None

        # Accept a pre-built client (test hook) or lazily build one.
        self._client = client

    # ------------------------------------------------------------------
    # Public API (Policy protocol)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear per-episode counters and forward to the fallback."""
        self.fallback_count = 0
        self.call_count = 0
        self.last_error = None
        if self.fallback_policy is not None:
            self.fallback_policy.reset()

    def act(self, observation: Dict[str, Any]) -> Any:
        """Single LLM step, with full error surface delegated to fallback."""
        self.call_count += 1
        try:
            user_prompt = self._build_user_prompt(observation)
            raw = self._call_llm_with_retries(user_prompt)
            data = self._parse_and_validate(raw)
            return data
        except Exception as exc:  # noqa: BLE001 — intentional: any error -> fallback
            self.fallback_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "[LLM-%s] step %d failed (%s); delegating to %s",
                self._log_name,
                self.call_count,
                self.last_error,
                type(self.fallback_policy).__name__,
            )
            return self.fallback_policy.act(observation)

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------
    def _build_user_prompt(self, observation: Dict[str, Any]) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------
    @property
    def client(self) -> Any:
        """Lazily-instantiated OpenAI-compatible client."""
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # imported lazily to keep tests light
        except ImportError as exc:  # pragma: no cover - the pkg is a hard dep
            raise RuntimeError(
                "openai>=1.0.0 is required to use LLM policies"
            ) from exc

        kwargs: Dict[str, Any] = {"base_url": self.api_base_url}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self._client = OpenAI(**kwargs)
        return self._client

    def _call_llm_once(self, user_prompt: str) -> str:
        """Single blocking call to the chat completions endpoint."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout_s,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def _call_llm_with_retries(self, user_prompt: str) -> str:
        """Call with up to ``retries`` additional attempts after the first."""
        last_exc: Optional[BaseException] = None
        attempts = max(1, self.retries + 1)
        for attempt in range(attempts):
            try:
                return self._call_llm_once(user_prompt)
            except Exception as exc:  # noqa: BLE001 — downstream classifier handles
                last_exc = exc
                if not self._is_retryable(exc) or attempt == attempts - 1:
                    break
                # Very small backoff; the outer caller has a hard fallback anyway.
                time.sleep(0.1 * (attempt + 1))
        assert last_exc is not None
        raise LLMCallError(str(last_exc)) from last_exc

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """Retry on timeouts / transient API errors; fail fast on JSON/schema."""
        name = type(exc).__name__
        # openai-python transient/HTTP-shape errors; matched by class name to
        # keep the import surface small (and test doubles friendly).
        return name in {
            "APITimeoutError",
            "APIConnectionError",
            "APIConnectionTimeoutError",
            "InternalServerError",
            "RateLimitError",
            "ServiceUnavailableError",
            "TimeoutError",
        }

    def _parse_and_validate(self, raw: str) -> Any:
        """Strip markdown fences, ``json.loads``, then Pydantic-validate."""
        text = _extract_json_text(raw)
        if not text:
            raise ValueError("empty LLM response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM produced invalid JSON: {exc}") from exc
        assert self.action_model is not None  # enforced in __init__
        try:
            return self.action_model.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"LLM JSON failed {self.action_model.__name__} schema: {exc}") from exc


__all__ = ["LLMPolicyBase", "LLMCallError"]
