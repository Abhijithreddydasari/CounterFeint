"""Shared helpers for scripted role policies."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class Policy(Protocol):
    """Common callable-style interface for scripted agents.

    Implementations may hold light per-episode state (e.g. a round counter)
    but must expose `reset()` so the driver can reuse them across episodes.
    """

    def act(self, observation: Dict[str, Any]) -> Any: ...
    def reset(self) -> None: ...


class PolicyBase:
    """Mixin that gives every policy a no-op `reset()` by default."""

    def reset(self) -> None:  # pragma: no cover - trivial
        return None
