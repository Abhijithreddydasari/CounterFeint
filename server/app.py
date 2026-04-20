"""
FastAPI application for the Ad Fraud Investigation Environment.

Creates the OpenEnv server via create_app() and registers custom HTTP
endpoints required by the hackathon: /tasks, /baseline, /grader.

Gradio is disabled (ENABLE_WEB_INTERFACE=false); the HTML UI lives at /investigate.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from openenv.core.env_server import create_app

try:
    from ..models import AdReviewAction, AdReviewObservation
except ImportError:
    from models import AdReviewAction, AdReviewObservation

from .environment import AdFraudEnvironment, get_last_grader_result
from .investigate_ui import register_investigate_ui

logger = logging.getLogger(__name__)

# Do not mount OpenEnv's Gradio stack (single FastAPI process on port 8000).
os.environ.setdefault("ENABLE_WEB_INTERFACE", "false")

app = create_app(
    AdFraudEnvironment,
    AdReviewAction,
    AdReviewObservation,
    env_name="counterfeint",
)

register_investigate_ui(app)


# ------------------------------------------------------------------
# Custom endpoints required by the competition
# ------------------------------------------------------------------


@app.get("/tasks", tags=["Competition"])
async def tasks() -> Dict[str, Any]:
    """Return the list of tasks and the action schema."""
    try:
        from ..data.ad_generator import TASK_CONFIGS
    except ImportError:
        from data.ad_generator import TASK_CONFIGS

    task_list = []
    for tid, cfg in TASK_CONFIGS.items():
        task_list.append({
            "id": cfg.task_id,
            "name": cfg.name,
            "difficulty": cfg.difficulty,
            "queue_size": cfg.queue_size,
            "action_budget": cfg.action_budget,
            "description": cfg.description,
        })

    return {
        "tasks": task_list,
        "action_schema": AdReviewAction.model_json_schema(),
    }


@app.get("/baseline", tags=["Competition"])
async def baseline() -> Dict[str, Any]:
    """Return baseline scores, running live inference if credentials are available."""
    baseline_path = Path(__file__).resolve().parent.parent / "baseline_scores.json"

    has_creds = all(os.getenv(v) for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"))
    if has_creds:
        try:
            try:
                from ..inference import run_baseline
            except ImportError:
                from inference import run_baseline
            scores = run_baseline()
            with open(baseline_path, "w") as f:
                json.dump(scores, f, indent=2)
            return scores
        except Exception as e:
            logger.warning("Live baseline failed, falling back to cached: %s", e)

    if baseline_path.exists():
        with open(baseline_path) as f:
            return json.load(f)

    return {
        "error": "No baseline scores available. Set API_BASE_URL, MODEL_NAME, and HF_TOKEN to run live inference.",
        "tasks": {},
    }


@app.get("/grader", tags=["Competition"])
async def grader() -> Dict[str, Any]:
    """Return grader score from the most recently completed episode."""
    result = get_last_grader_result()
    if not result:
        return {
            "error": "No completed episode. Run an episode via WebSocket or the /investigate UI.",
            "grader_score": None,
        }
    return result


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
