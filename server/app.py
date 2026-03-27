"""
FastAPI application for the Ad Fraud Investigation Environment.

Creates the OpenEnv server via create_app() and registers custom HTTP
endpoints required by the hackathon: /tasks, /baseline, /grader.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from openenv.core.env_server import create_app

from ..models import AdReviewAction, AdReviewObservation
from .environment import AdFraudEnvironment, get_last_grader_result

logger = logging.getLogger(__name__)

app = create_app(
    AdFraudEnvironment,
    AdReviewAction,
    AdReviewObservation,
    env_name="ad_fraud_env",
)


# ------------------------------------------------------------------
# Custom endpoints required by the competition
# ------------------------------------------------------------------


@app.get("/tasks", tags=["Competition"])
async def tasks() -> Dict[str, Any]:
    """Return the list of tasks and the action schema."""
    from ..data.ad_generator import TASK_CONFIGS

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
    """Return baseline scores, running live inference if API key is available."""
    baseline_path = Path(__file__).resolve().parent.parent / "baseline_scores.json"

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from ..inference import run_baseline
            scores = run_baseline(api_key)
            with open(baseline_path, "w") as f:
                json.dump(scores, f, indent=2)
            return scores
        except Exception as e:
            logger.warning("Live baseline failed, falling back to cached: %s", e)

    if baseline_path.exists():
        with open(baseline_path) as f:
            return json.load(f)

    return {
        "error": "No baseline scores available. Set OPENAI_API_KEY to run live inference.",
        "tasks": {},
    }


@app.get("/grader", tags=["Competition"])
async def grader() -> Dict[str, Any]:
    """Return grader score from the most recently completed episode."""
    result = get_last_grader_result()
    if not result:
        return {
            "error": "No completed episode. Run an episode via WebSocket first.",
            "grader_score": None,
        }
    return result


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
