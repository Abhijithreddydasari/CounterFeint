"""
Pure HTML investigation dashboard: persistent env + JSON API.

OpenEnv's HTTP POST /reset and /step spin up a new environment each call,
so this UI uses a singleton AdFraudEnvironment for multi-step episodes.
Does not replace /ws or competition endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server import serialize_observation
from pydantic import BaseModel, Field

try:
    from ..models import AdReviewAction
    from .environment import AdFraudEnvironment
except ImportError:
    from models import AdReviewAction
    from server.environment import AdFraudEnvironment

_ui_env: Optional[AdFraudEnvironment] = None


def _get_ui_env() -> AdFraudEnvironment:
    global _ui_env
    if _ui_env is None:
        _ui_env = AdFraudEnvironment()
    return _ui_env


class UIResetBody(BaseModel):
    task_id: str = Field(default="task_1")
    seed: int = Field(default=42, ge=0)


def register_investigate_ui(app: FastAPI) -> None:
    static_dir = Path(__file__).resolve().parent / "static"

    @app.get("/", include_in_schema=False)
    async def root_to_investigate() -> RedirectResponse:
        return RedirectResponse(url="/investigate", status_code=302)

    @app.get("/web", include_in_schema=False)
    async def web_to_investigate() -> RedirectResponse:
        return RedirectResponse(url="/investigate", status_code=302)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/investigate", include_in_schema=False)
    async def investigate_page() -> FileResponse:
        path = static_dir / "investigate_hq.html"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="investigate_hq.html missing")
        return FileResponse(path, media_type="text/html; charset=utf-8")

    @app.post("/investigate/api/reset", tags=["Investigation UI"])
    async def investigate_api_reset(body: UIResetBody) -> Dict[str, Any]:
        env = _get_ui_env()
        obs = env.reset(task_id=body.task_id, seed=body.seed)
        return serialize_observation(obs)

    @app.post("/investigate/api/step", tags=["Investigation UI"])
    async def investigate_api_step(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        try:
            action = AdReviewAction(**body)
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        env = _get_ui_env()
        obs = env.step(action)
        return serialize_observation(obs)

    @app.get("/investigate/api/state", tags=["Investigation UI"])
    async def investigate_api_state() -> Dict[str, Any]:
        return _get_ui_env().state.model_dump()
