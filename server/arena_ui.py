"""
Multi-agent arena HTTP API for the interactive demo UI.

Provides stateful HTTP endpoints that drive a shared RefereeEnvironment,
plus an ``auto-match`` endpoint that runs a complete scripted match and
returns the full replay trace for animated playback in the frontend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from ..models import AdReviewAction, AuditorAction, FraudsterAction
    from ..scripted.auditor import HeuristicAuditor
    from ..scripted.fraudster import ReactiveFraudster
    from ..scripted.investigator import ScriptedInvestigator
    from .referee import RefereeEnvironment
except ImportError:
    from models import AdReviewAction, AuditorAction, FraudsterAction  # type: ignore[no-redef]
    from scripted.auditor import HeuristicAuditor  # type: ignore[no-redef]
    from scripted.fraudster import ReactiveFraudster  # type: ignore[no-redef]
    from scripted.investigator import ScriptedInvestigator  # type: ignore[no-redef]
    from server.referee import RefereeEnvironment  # type: ignore[no-redef]

_arena_env: Optional[RefereeEnvironment] = None


def _get_arena_env() -> RefereeEnvironment:
    global _arena_env
    if _arena_env is None:
        _arena_env = RefereeEnvironment()
    return _arena_env


class ArenaResetBody(BaseModel):
    task_id: str = Field(default="task_1")
    seed: int = Field(default=42, ge=0)


def _obs_to_dict(obs: Any) -> Dict[str, Any]:
    return obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)


def register_arena_ui(app: FastAPI) -> None:
    """Register multi-agent arena HTTP endpoints on the given FastAPI app."""

    @app.post("/arena/api/reset", tags=["Arena Demo"])
    async def arena_reset(body: ArenaResetBody) -> Dict[str, Any]:
        env = _get_arena_env()
        env.reset_match(seed=body.seed, task_id=body.task_id)
        return {
            "match_id": env.match_id,
            "phase": env.phase,
            "state": env.state.model_dump(),
            "fraudster_obs": _obs_to_dict(env.build_fraudster_observation()),
        }

    @app.post("/arena/api/step/fraudster", tags=["Arena Demo"])
    async def arena_step_fraudster(
        body: Dict[str, Any] = Body(...)
    ) -> Dict[str, Any]:
        env = _get_arena_env()
        try:
            action = FraudsterAction(**body)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        try:
            obs = env.step_as_fraudster(action)
        except PermissionError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return {
            "observation": _obs_to_dict(obs),
            "phase": env.phase,
            "done": env.done,
            "state": env.state.model_dump(),
        }

    @app.post("/arena/api/step/investigator", tags=["Arena Demo"])
    async def arena_step_investigator(
        body: Dict[str, Any] = Body(...)
    ) -> Dict[str, Any]:
        env = _get_arena_env()
        try:
            action = AdReviewAction(**body)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        try:
            obs = env.step_as_investigator(action)
        except PermissionError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return {
            "observation": _obs_to_dict(obs),
            "phase": env.phase,
            "done": env.done,
            "state": env.state.model_dump(),
        }

    @app.post("/arena/api/step/auditor", tags=["Arena Demo"])
    async def arena_step_auditor(
        body: Dict[str, Any] = Body(...)
    ) -> Dict[str, Any]:
        env = _get_arena_env()
        try:
            action = AuditorAction(**body)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        try:
            obs = env.step_as_auditor(action)
        except PermissionError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return {
            "observation": _obs_to_dict(obs),
            "phase": env.phase,
            "done": env.done,
            "state": env.state.model_dump(),
        }

    @app.get("/arena/api/state", tags=["Arena Demo"])
    async def arena_state() -> Dict[str, Any]:
        env = _get_arena_env()
        return {
            "match_id": env.match_id,
            "phase": env.phase,
            "done": env.done,
            "state": env.state.model_dump(),
        }

    @app.post("/arena/api/auto", tags=["Arena Demo"])
    async def arena_auto_demo(body: ArenaResetBody) -> Dict[str, Any]:
        """Run a complete scripted match and return the full replay trace."""
        env = RefereeEnvironment()
        env.reset_match(seed=body.seed, task_id=body.task_id)

        fraudster = ReactiveFraudster(seed=body.seed)
        investigator = ScriptedInvestigator()
        auditor = HeuristicAuditor()

        trace: List[Dict[str, Any]] = []
        cum = {"fraudster": 0.0, "investigator": 0.0, "auditor": 0.0}
        trajectories: Dict[str, List[float]] = {
            "fraudster": [],
            "investigator": [],
            "auditor": [],
        }

        max_steps = 120
        step = 0

        while not env.done and step < max_steps:
            phase = env.phase

            if phase == "fraudster_turn":
                obs = env.build_fraudster_observation()
                action = fraudster.act(_obs_to_dict(obs))
                result = env.step_as_fraudster(action)
                r = float(result.reward or 0)
                cum["fraudster"] += r
                trajectories["fraudster"].append(cum["fraudster"])
                trace.append({
                    "step": step,
                    "role": "fraudster",
                    "action_type": action.action_type,
                    "detail": _summarize_action("fraudster", action),
                    "reward": round(r, 4),
                    "cum_reward": round(cum["fraudster"], 4),
                    "feedback": (result.feedback or "")[:250],
                    "phase_after": env.phase,
                })

            elif phase == "investigator_turn":
                obs = env.build_investigator_observation()
                action = investigator.act(_obs_to_dict(obs))
                result = env.step_as_investigator(action)
                r = float(result.reward or 0)
                cum["investigator"] += r
                trajectories["investigator"].append(cum["investigator"])
                trace.append({
                    "step": step,
                    "role": "investigator",
                    "action_type": action.action_type,
                    "detail": _summarize_action("investigator", action),
                    "reward": round(r, 4),
                    "cum_reward": round(cum["investigator"], 4),
                    "feedback": (result.feedback or "")[:250],
                    "phase_after": env.phase,
                })

            elif phase == "audit_phase":
                obs = env.build_auditor_observation()
                action = auditor.act(_obs_to_dict(obs))
                result = env.step_as_auditor(action)
                r = float(result.reward or 0)
                cum["auditor"] += r
                trajectories["auditor"].append(cum["auditor"])
                trace.append({
                    "step": step,
                    "role": "auditor",
                    "action_type": action.action_type,
                    "detail": _summarize_action("auditor", action),
                    "reward": round(r, 4),
                    "cum_reward": round(cum["auditor"], 4),
                    "feedback": (result.feedback or "")[:250],
                    "phase_after": env.phase,
                })
            else:
                break

            step += 1

        state = env.state
        return {
            "match_id": env.match_id,
            "task_id": body.task_id,
            "total_steps": step,
            "trace": trace,
            "final_rewards": {k: round(v, 4) for k, v in cum.items()},
            "reward_trajectories": {
                k: [round(v, 4) for v in vs]
                for k, vs in trajectories.items()
            },
            "final_state": {
                "grader_score": state.grader_score,
                "fraudster_reward": state.fraudster_reward,
                "investigator_reward": state.investigator_reward,
                "auditor_reward": state.auditor_reward,
                "end_reason": state.end_reason,
                "proposals_used": state.proposals_used,
                "round_number": state.round_number,
                "audit_report": state.audit_report,
            },
        }


def _summarize_action(role: str, action: Any) -> str:
    """One-liner summary of an action for the trace timeline."""
    if role == "fraudster":
        if action.action_type == "propose_ad":
            copy = (action.ad_copy or "")[:60]
            return f"Proposed ad ({action.category}): \"{copy}...\""
        if action.action_type == "modify_pending_ad":
            return f"Modified slot {action.slot_index}"
        if action.action_type == "end_turn":
            return "Ended turn"
        if action.action_type == "commit_final":
            return "Committed final — no more proposals"
    elif role == "investigator":
        if action.action_type == "investigate":
            return f"Investigated {action.ad_id} → {action.investigation_target}"
        if action.action_type == "verdict":
            return f"Verdict on {action.ad_id}: {action.verdict} ({action.confidence:.0%})"
        if action.action_type == "link_accounts":
            return f"Linked {action.ad_id} ↔ {action.linked_ad_id}"
    elif role == "auditor":
        if action.action_type == "flag_investigator":
            return f"Track A flag: {action.flag_type} on {action.target_ad_id}"
        if action.action_type == "flag_fraudster":
            return f"Track B flag: {action.flag_type} on {action.target_ad_id}"
        if action.action_type == "submit_audit_report":
            return "Submitted final audit report"
    return action.action_type
