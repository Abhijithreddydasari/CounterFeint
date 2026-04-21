"""
Role-specific WebSocket routes for the multi-agent FraudArena.

Registers three WebSocket endpoints on the shared FastAPI app:

    /ws/fraudster
    /ws/investigator
    /ws/auditor

All three routes are backed by a per-match `RefereeEnvironment` instance
stored in a module-level pool keyed by `match_id`. This lets three separate
WebSocket connections (one per role) mutate the same underlying environment
state — which is exactly what the "Multi-Agent Interactions" theme asks for:

    > Shared-environment-state + observation-of-opponent-actions is valid
    > (organizer confirmation received during Round 2).

The original `/ws` endpoint (registered by OpenEnv's `create_app`) is left
untouched for Round 1 backwards compatibility.

Protocol (client -> server, JSON strings):

    {"type": "reset",  "data": {"seed": 42, "task_id": "task_1",
                                  "max_rounds": 4, "max_proposals": 5,
                                  ...}}
        Creates a fresh match. Server returns the role's observation plus
        a `match_id` in the response data. The Fraudster typically calls
        this first since they go first.

    {"type": "join",  "data": {"match_id": "<uuid>"}}
        Attaches to an existing match without re-initializing it. Server
        returns the role's *current* observation.

    {"type": "step",  "data": {<role-specific action JSON>}}
        Executes the action against the shared Referee. Server returns
        the role's updated observation.

    {"type": "obs",   "data": {}}
        Returns the role's *current* observation without taking an action.
        Useful for polling after another role has acted.

    {"type": "state", "data": {}}
        Returns the full RefereeState (shared across all roles).

    {"type": "close", "data": {}}
        Gracefully detaches this role from the match.

Protocol (server -> client, JSON strings):

    {"type": "observation", "data": {<role observation>, "match_id": "..."}}
    {"type": "state",       "data": {<RefereeState>}}
    {"type": "error",       "data": {"message": "...", "code": "..."}}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Type

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

try:
    from ..models import (
        AdReviewAction,
        AdReviewObservation,
        AuditorAction,
        AuditorObservation,
        FraudsterAction,
        FraudsterObservation,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        AdReviewAction,
        AdReviewObservation,
        AuditorAction,
        AuditorObservation,
        FraudsterAction,
        FraudsterObservation,
    )

from openenv.core.env_server.types import Action, Observation

from .referee import RefereeEnvironment

logger = logging.getLogger(__name__)

Role = Literal["fraudster", "investigator", "auditor"]


# ---------------------------------------------------------------------------
# Match pool
# ---------------------------------------------------------------------------


@dataclass
class MatchEntry:
    """One shared Referee instance with an asyncio lock and connected-role set."""

    env: RefereeEnvironment
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    connected_roles: Set[Role] = field(default_factory=set)


_matches: Dict[str, MatchEntry] = {}
_match_history: Dict[str, Dict[str, Any]] = {}
_MAX_HISTORY = 200
_pool_lock = asyncio.Lock()


def _summarize_match(match_id: str, entry: MatchEntry) -> Dict[str, Any]:
    env = entry.env
    state = env.state
    return {
        "match_id": match_id,
        "task_id": state.task_id,
        "phase": state.phase,
        "round_number": state.round_number,
        "max_rounds": state.max_rounds,
        "proposals_used": state.proposals_used,
        "max_proposals": state.max_proposals,
        "fraudster_committed": state.fraudster_committed,
        "grader_score": state.grader_score,
        "end_reason": state.end_reason,
        "rewards": {
            "fraudster": state.fraudster_reward,
            "investigator": state.investigator_reward,
            "auditor": state.auditor_reward,
        },
        "connected_roles": sorted(entry.connected_roles),
        "done": env.done,
    }


def _archive_match(match_id: str, entry: MatchEntry) -> None:
    """Snapshot a match's final state into the history ring buffer."""
    env = entry.env
    state = env.state
    snapshot = {
        **_summarize_match(match_id, entry),
        "audit_report": state.audit_report,
        "fraudster_proposals": list(state.fraudster_proposals),
        "investigator_action_log": list(state.investigator_action_log),
        "archived_at": time.time(),
    }
    _match_history[match_id] = snapshot
    if len(_match_history) > _MAX_HISTORY:
        oldest = min(
            _match_history, key=lambda k: _match_history[k].get("archived_at", 0)
        )
        _match_history.pop(oldest, None)


async def _create_match(**reset_kwargs: Any) -> MatchEntry:
    env = RefereeEnvironment()
    env.reset_match(**reset_kwargs)
    entry = MatchEntry(env=env)
    async with _pool_lock:
        _matches[env.match_id] = entry
    logger.info("[multi-agent] match %s created", env.match_id)
    return entry


async def _get_match(match_id: str) -> Optional[MatchEntry]:
    async with _pool_lock:
        return _matches.get(match_id)


async def _drop_match_if_stale(match_id: str) -> None:
    """Archive + remove a match from the pool when it's done and idle."""
    async with _pool_lock:
        entry = _matches.get(match_id)
        if entry is None:
            return
        if entry.env.done and not entry.connected_roles:
            _archive_match(match_id, entry)
            _matches.pop(match_id, None)
            logger.info("[multi-agent] match %s archived and evicted", match_id)


# ---------------------------------------------------------------------------
# Public accessors — used by /api/v1/* in public_api.py
# ---------------------------------------------------------------------------


def list_active_matches() -> List[Dict[str, Any]]:
    """Return summaries of all currently-live matches."""
    return [_summarize_match(mid, entry) for mid, entry in _matches.items()]


def list_archived_matches(limit: int = 50) -> List[Dict[str, Any]]:
    """Return up to `limit` most-recent completed match snapshots."""
    items = sorted(
        _match_history.values(),
        key=lambda s: s.get("archived_at", 0),
        reverse=True,
    )
    return items[: max(0, limit)]


def get_match_summary(match_id: str) -> Optional[Dict[str, Any]]:
    """Return the live summary if the match is active, else the archived snapshot."""
    entry = _matches.get(match_id)
    if entry is not None:
        return _summarize_match(match_id, entry)
    return _match_history.get(match_id)


def get_match_entry(match_id: str) -> Optional[MatchEntry]:
    """Return the live MatchEntry (for the public_api events/report lookups)."""
    return _matches.get(match_id)


def get_match_archive(match_id: str) -> Optional[Dict[str, Any]]:
    """Return the archived snapshot for a done match, if any."""
    return _match_history.get(match_id)


async def create_match_async(**reset_kwargs: Any) -> MatchEntry:
    """Public wrapper around `_create_match` (used by POST /api/v1/matches)."""
    return await _create_match(**reset_kwargs)


async def end_match_async(match_id: str) -> bool:
    """Force-archive+evict a match. Returns True if it existed."""
    async with _pool_lock:
        entry = _matches.pop(match_id, None)
        if entry is None:
            return False
        _archive_match(match_id, entry)
        logger.info("[multi-agent] match %s force-ended", match_id)
        return True


def get_active_match_ids() -> Dict[str, Dict[str, Any]]:
    """Diagnostic helper for debugging; used by /matches endpoint."""
    out: Dict[str, Dict[str, Any]] = {}
    for mid, entry in _matches.items():
        out[mid] = {
            "phase": entry.env.phase,
            "round_number": entry.env.state.round_number,
            "done": entry.env.done,
            "connected_roles": sorted(entry.connected_roles),
        }
    return out


# ---------------------------------------------------------------------------
# Role -> (Action, Observation) lookup
# ---------------------------------------------------------------------------


_ROLE_ACTIONS: Dict[Role, Type[Action]] = {
    "fraudster": FraudsterAction,
    "investigator": AdReviewAction,
    "auditor": AuditorAction,
}

_ROLE_OBSERVATIONS: Dict[Role, Type[Observation]] = {
    "fraudster": FraudsterObservation,
    "investigator": AdReviewObservation,
    "auditor": AuditorObservation,
}


def _build_role_observation(role: Role, entry: MatchEntry) -> Observation:
    env = entry.env
    if role == "fraudster":
        return env.build_fraudster_observation()
    if role == "investigator":
        return env.build_investigator_observation()
    if role == "auditor":
        return env.build_auditor_observation()
    raise ValueError(f"Unknown role: {role}")


def _step_role(role: Role, entry: MatchEntry, action: Action) -> Observation:
    env = entry.env
    if role == "fraudster":
        return env.step_as_fraudster(action)  # type: ignore[arg-type]
    if role == "investigator":
        return env.step_as_investigator(action)  # type: ignore[arg-type]
    if role == "auditor":
        return env.step_as_auditor(action)  # type: ignore[arg-type]
    raise ValueError(f"Unknown role: {role}")


def _observation_to_payload(
    role: Role, obs: Observation, entry: MatchEntry
) -> Dict[str, Any]:
    payload = obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)
    env = entry.env
    st = env.state
    payload["match_id"] = env.match_id
    payload["role"] = role
    payload.setdefault("phase", env.phase)
    payload.setdefault("round_number", st.round_number)
    payload.setdefault("max_rounds", st.max_rounds)
    payload.setdefault(
        "rounds_remaining", max(0, st.max_rounds - st.round_number + 1)
    )
    return payload


def _error(message: str, code: str = "execution_error") -> str:
    return json.dumps({"type": "error", "data": {"message": message, "code": code}})


# ---------------------------------------------------------------------------
# Core WS dispatch
# ---------------------------------------------------------------------------


async def _run_role_ws(websocket: WebSocket, role: Role) -> None:
    """
    One WS connection, one role. Lives until the client closes or the
    server hits a terminal error.
    """
    await websocket.accept()

    match_id: Optional[str] = None
    entry: Optional[MatchEntry] = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                await websocket.send_text(_error(f"invalid JSON: {exc}", "invalid_json"))
                continue

            msg_type = msg.get("type", "")
            data: Dict[str, Any] = msg.get("data", {}) or {}

            if msg_type == "reset":
                reset_kwargs = {
                    k: v
                    for k, v in data.items()
                    if k
                    in {
                        "seed",
                        "episode_id",
                        "task_id",
                        "max_rounds",
                        "max_proposals",
                        "max_fraudster_actions_per_turn",
                        "max_investigator_actions_per_turn",
                        "allowed_categories",
                    }
                    and v is not None
                }
                try:
                    entry = await _create_match(**reset_kwargs)
                except Exception as exc:
                    logger.exception("[multi-agent] reset failed")
                    await websocket.send_text(
                        _error(f"reset failed: {exc}", "reset_error")
                    )
                    continue

                match_id = entry.env.match_id
                entry.connected_roles.add(role)
                obs = _build_role_observation(role, entry)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "observation",
                            "data": _observation_to_payload(role, obs, entry),
                        }
                    )
                )

            elif msg_type == "join":
                requested = data.get("match_id")
                if not requested:
                    await websocket.send_text(
                        _error("join requires match_id", "missing_match_id")
                    )
                    continue
                found = await _get_match(requested)
                if found is None:
                    await websocket.send_text(
                        _error(
                            f"unknown match_id {requested}; create via reset first",
                            "unknown_match",
                        )
                    )
                    continue
                entry = found
                match_id = requested
                entry.connected_roles.add(role)
                obs = _build_role_observation(role, entry)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "observation",
                            "data": _observation_to_payload(role, obs, entry),
                        }
                    )
                )

            elif msg_type == "step":
                if entry is None or match_id is None:
                    await websocket.send_text(
                        _error("no active match — call reset or join first", "no_match")
                    )
                    continue
                action_cls = _ROLE_ACTIONS[role]
                try:
                    action = action_cls.model_validate(data)
                except ValidationError as exc:
                    await websocket.send_text(
                        _error(
                            f"action validation failed: {exc.errors()}",
                            "validation_error",
                        )
                    )
                    continue
                async with entry.lock:
                    try:
                        obs = _step_role(role, entry, action)
                    except PermissionError as exc:
                        await websocket.send_text(
                            _error(str(exc), "phase_violation")
                        )
                        continue
                    except Exception as exc:
                        logger.exception("[multi-agent] step failed")
                        await websocket.send_text(
                            _error(f"step failed: {exc}", "execution_error")
                        )
                        continue
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "observation",
                            "data": _observation_to_payload(role, obs, entry),
                        }
                    )
                )

            elif msg_type == "obs":
                if entry is None or match_id is None:
                    await websocket.send_text(
                        _error(
                            "no active match — call reset or join first",
                            "no_match",
                        )
                    )
                    continue
                obs = _build_role_observation(role, entry)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "observation",
                            "data": _observation_to_payload(role, obs, entry),
                        }
                    )
                )

            elif msg_type == "state":
                if entry is None or match_id is None:
                    await websocket.send_text(
                        _error(
                            "no active match — call reset or join first",
                            "no_match",
                        )
                    )
                    continue
                state = entry.env.state
                payload = (
                    state.model_dump() if hasattr(state, "model_dump") else dict(state)
                )
                payload["match_id"] = match_id
                await websocket.send_text(
                    json.dumps({"type": "state", "data": payload})
                )

            elif msg_type == "close":
                break

            else:
                await websocket.send_text(
                    _error(f"unknown message type: {msg_type}", "unknown_type")
                )

    except WebSocketDisconnect:
        logger.info("[multi-agent] %s disconnected (match=%s)", role, match_id)
    except Exception as exc:
        logger.exception("[multi-agent] fatal error for role=%s match=%s", role, match_id)
        try:
            await websocket.send_text(_error(str(exc), "fatal_error"))
        except Exception:
            pass
    finally:
        if entry is not None:
            entry.connected_roles.discard(role)
        if match_id is not None:
            await _drop_match_if_stale(match_id)
        try:
            await websocket.close()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def register_multi_agent_routes(app: FastAPI) -> None:
    """
    Register the three role-specific WebSocket routes plus a `/matches`
    diagnostic endpoint on the given FastAPI app. Safe to call exactly once.
    """

    @app.websocket("/ws/fraudster")
    async def _ws_fraudster(ws: WebSocket) -> None:
        await _run_role_ws(ws, "fraudster")

    @app.websocket("/ws/investigator")
    async def _ws_investigator(ws: WebSocket) -> None:
        await _run_role_ws(ws, "investigator")

    @app.websocket("/ws/auditor")
    async def _ws_auditor(ws: WebSocket) -> None:
        await _run_role_ws(ws, "auditor")

    @app.get("/matches", tags=["Multi-Agent"])
    async def _matches_diag() -> Dict[str, Any]:
        """Return the set of currently active matches (for debugging)."""
        return {"active_matches": get_active_match_ids()}
