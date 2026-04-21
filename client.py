"""
CounterFeint — multi-agent FraudArena client library.

Exports:

    FraudsterClient      WebSocket client for `/ws/fraudster`
    InvestigatorClient   WebSocket client for `/ws/investigator`
    AuditorClient        WebSocket client for `/ws/auditor`
    MatchClient          Convenience coordinator that owns all three
                         WS connections, shares a single `match_id`,
                         and exposes a flat async API.

    AdFraudEnv           Legacy single-agent client (R1 compatibility).
                         Speaks to `/ws` (Investigator-only). Kept so
                         existing R1 inference / baseline scripts run
                         without change.

Example (three-agent):

    async with MatchClient("ws://localhost:8000") as match:
        await match.reset(seed=42, task_id="task_1")
        # match.fraudster.step(...), match.investigator.step(...), etc.
        state = await match.state()

Example (R1 single-agent):

    env = AdFraudEnv(base_url="http://localhost:8000").sync()
    env.connect()
    result = env.reset(seed=42, task_id="task_1")
    result = env.step(AdReviewAction(
        action_type="verdict", ad_id="ad_001",
        verdict="approve", confidence=0.8,
    ))
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import TracebackType
from typing import Any, Dict, Optional, Type

import websockets
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import (
    AdFraudState,
    AdReviewAction,
    AdReviewObservation,
    AuditorAction,
    AuditorObservation,
    FraudsterAction,
    FraudsterObservation,
    RefereeState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy R1 single-agent client (unchanged)
# ---------------------------------------------------------------------------


class AdFraudEnv(EnvClient[AdReviewAction, AdReviewObservation, AdFraudState]):
    """
    R1 single-agent WebSocket client (Investigator-only).

    Kept for backwards compatibility with the Round-1 baseline script
    and for OpenEnv's standard `/ws` route. For Round-2 multi-agent
    workflows use `MatchClient` or the per-role clients.
    """

    def _step_payload(self, action: AdReviewAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True, exclude={"metadata"})

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[AdReviewObservation]:
        obs_data = payload.get("observation", {})
        reward = payload.get("reward", 0.0) or 0.0
        done = payload.get("done", False)

        observation = AdReviewObservation(
            done=done,
            reward=reward,
            queue_summary=obs_data.get("queue_summary", ""),
            current_ad_info=obs_data.get("current_ad_info", ""),
            investigation_findings=obs_data.get("investigation_findings", ""),
            verdict_history_summary=obs_data.get("verdict_history_summary", ""),
            feedback=obs_data.get("feedback", ""),
            available_ads=obs_data.get("available_ads", []),
            queue_status=obs_data.get("queue_status", {}),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(observation=observation, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> AdFraudState:
        return AdFraudState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            total_ads=payload.get("total_ads", 0),
            reviewed_count=payload.get("reviewed_count", 0),
            remaining_budget=payload.get("remaining_budget", 0),
            verdicts=payload.get("verdicts", {}),
            grader_score=payload.get("grader_score"),
        )


# ---------------------------------------------------------------------------
# Shared multi-agent client infrastructure
# ---------------------------------------------------------------------------


class MultiAgentProtocolError(RuntimeError):
    """Raised when the server returns a protocol-level error (validation, phase violation, ...)."""

    def __init__(self, message: str, code: str = "execution_error") -> None:
        super().__init__(message)
        self.code = code


def _http_to_ws(base_url: str) -> str:
    """Normalize an http(s)://... base URL to ws(s)://... (idempotent)."""
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://"):]
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://"):]
    return base_url


class _RoleClient:
    """
    Async WebSocket client for a single role.

    All three role-specific subclasses share this logic. The only variation
    is the WS path (`/ws/fraudster`, `/ws/investigator`, `/ws/auditor`) and
    the action/observation Pydantic types (enforced via generics in the
    subclasses).
    """

    ws_path: str = ""  # override in subclasses
    action_cls: Type[Any] = Any  # type: ignore[assignment]
    observation_cls: Type[Any] = Any  # type: ignore[assignment]

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self._ws_base = _http_to_ws(base_url.rstrip("/"))
        self._url = f"{self._ws_base}{self.ws_path}"
        self._timeout = timeout
        self._ws: Optional[Any] = None
        self._match_id: Optional[str] = None

    @property
    def match_id(self) -> Optional[str]:
        return self._match_id

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def connect(self) -> None:
        if self._ws is not None:
            return
        self._ws = await websockets.connect(
            self._url,
            open_timeout=self._timeout,
            ping_interval=None,
        )

    async def close(self) -> None:
        if self._ws is None:
            return
        try:
            await self._send("close", {})
        except Exception:
            pass
        try:
            await self._ws.close()
        except Exception:
            pass
        self._ws = None

    async def __aenter__(self) -> "_RoleClient":
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self.close()

    async def reset(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Create a new match. Only one role needs to call `reset` per match
        (typically the Fraudster). Returns the role's initial observation.
        """
        await self.connect()
        data = {k: v for k, v in kwargs.items() if v is not None}
        await self._send("reset", data)
        payload = await self._recv_observation()
        self._match_id = payload.get("match_id")
        return payload

    async def join(self, match_id: str) -> Dict[str, Any]:
        """Attach to an existing match (another role already called `reset`)."""
        await self.connect()
        await self._send("join", {"match_id": match_id})
        payload = await self._recv_observation()
        self._match_id = match_id
        return payload

    async def step(self, action: Any) -> Dict[str, Any]:
        """Execute an action. Raises MultiAgentProtocolError on validation / phase errors."""
        await self._require_connected()
        payload = (
            action.model_dump(exclude_none=True)
            if hasattr(action, "model_dump")
            else dict(action)
        )
        await self._send("step", payload)
        return await self._recv_observation()

    async def obs(self) -> Dict[str, Any]:
        """Return the current observation without stepping."""
        await self._require_connected()
        await self._send("obs", {})
        return await self._recv_observation()

    async def state(self) -> Dict[str, Any]:
        """Return the full shared `RefereeState` for this match."""
        await self._require_connected()
        await self._send("state", {})
        msg = await self._recv_any()
        if msg.get("type") != "state":
            raise MultiAgentProtocolError(
                f"expected state response, got {msg!r}", code="protocol_error"
            )
        return msg["data"]

    async def _send(self, msg_type: str, data: Dict[str, Any]) -> None:
        assert self._ws is not None
        await self._ws.send(json.dumps({"type": msg_type, "data": data}))

    async def _recv_any(self) -> Dict[str, Any]:
        assert self._ws is not None
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self._timeout)
        msg = json.loads(raw)
        if msg.get("type") == "error":
            err = msg.get("data", {})
            raise MultiAgentProtocolError(
                err.get("message", "unknown error"),
                code=err.get("code", "execution_error"),
            )
        return msg

    async def _recv_observation(self) -> Dict[str, Any]:
        msg = await self._recv_any()
        if msg.get("type") != "observation":
            raise MultiAgentProtocolError(
                f"expected observation, got {msg!r}", code="protocol_error"
            )
        return msg["data"]

    async def _require_connected(self) -> None:
        if self._ws is None:
            raise RuntimeError(
                f"{type(self).__name__} is not connected; call connect()/reset()/join() first"
            )


class FraudsterClient(_RoleClient):
    """Fraudster agent (proposes / modifies ads). Connects to `/ws/fraudster`."""

    ws_path = "/ws/fraudster"
    action_cls = FraudsterAction
    observation_cls = FraudsterObservation


class InvestigatorClient(_RoleClient):
    """Investigator agent (investigates + verdicts). Connects to `/ws/investigator`."""

    ws_path = "/ws/investigator"
    action_cls = AdReviewAction
    observation_cls = AdReviewObservation


class AuditorClient(_RoleClient):
    """Auditor agent (audits both peers post-hoc). Connects to `/ws/auditor`."""

    ws_path = "/ws/auditor"
    action_cls = AuditorAction
    observation_cls = AuditorObservation


# ---------------------------------------------------------------------------
# MatchClient — convenience coordinator
# ---------------------------------------------------------------------------


class MatchClient:
    """
    Convenience wrapper owning three role-specific WS clients plus a
    shared `match_id`. Handles the dance of:

        1. Fraudster connects + resets a match
        2. Investigator and Auditor join using the returned `match_id`

    Use as an async context manager, or call `connect()`/`close()` manually.
    """

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url
        self.fraudster = FraudsterClient(base_url, timeout=timeout)
        self.investigator = InvestigatorClient(base_url, timeout=timeout)
        self.auditor = AuditorClient(base_url, timeout=timeout)
        self._match_id: Optional[str] = None

    @property
    def match_id(self) -> Optional[str]:
        return self._match_id

    async def __aenter__(self) -> "MatchClient":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self.close()

    async def reset(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Open the three WS connections, create a match via the Fraudster, and
        have the Investigator + Auditor join. Returns the Fraudster's initial
        observation (includes `match_id`).
        """
        fraud_obs = await self.fraudster.reset(**kwargs)
        match_id = fraud_obs.get("match_id")
        if not match_id:
            raise MultiAgentProtocolError(
                "server did not return match_id on reset", code="protocol_error"
            )
        self._match_id = match_id

        await self.investigator.join(match_id)
        await self.auditor.join(match_id)
        return fraud_obs

    async def state(self) -> RefereeState:
        """Return the shared RefereeState as a typed Pydantic object."""
        if self._match_id is None:
            raise RuntimeError("no active match; call reset() first")
        data = await self.fraudster.state()
        return RefereeState.model_validate(data)

    async def close(self) -> None:
        """Close all three WS connections. Safe to call multiple times."""
        for client in (self.fraudster, self.investigator, self.auditor):
            try:
                await client.close()
            except Exception:
                logger.debug("client.close() raised", exc_info=True)


__all__ = [
    "AdFraudEnv",
    "AuditorClient",
    "FraudsterClient",
    "InvestigatorClient",
    "MatchClient",
    "MultiAgentProtocolError",
]
