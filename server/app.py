"""
FastAPI application for CounterFeint — a multi-agent ad-fraud FraudArena.

Creates the OpenEnv server via create_app() (Round-1-compatible `/ws`)
and layers on Round-2 role-specific routes:

    /ws/fraudster     Fraudster agent (proposes / modifies ads)
    /ws/investigator  Investigator agent (reviews ads, renders verdicts)
    /ws/auditor       Auditor agent (audits Investigator + Fraudster traces)

Custom HTTP endpoints: /tasks, /baseline, /grader, /matches.

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
    from ..models import (
        AdReviewAction,
        AdReviewObservation,
        AuditorAction,
        AuditorObservation,
        FraudsterAction,
        FraudsterObservation,
    )
except ImportError:
    from models import (
        AdReviewAction,
        AdReviewObservation,
        AuditorAction,
        AuditorObservation,
        FraudsterAction,
        FraudsterObservation,
    )

from .environment import AdFraudEnvironment, get_last_grader_result
from .investigate_ui import register_investigate_ui
from .multi_agent_ws import register_multi_agent_routes
from .public_api import register_public_api
from .referee import get_last_grader_result as get_last_multi_agent_grader_result

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
register_multi_agent_routes(app)
register_public_api(app)


# ------------------------------------------------------------------
# Custom endpoints required by the competition
# ------------------------------------------------------------------


@app.get("/tasks", tags=["Competition"])
async def tasks() -> Dict[str, Any]:
    """Return the list of tasks, the action schema, and the R2 role catalog."""
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

    roles = {
        "fraudster": {
            "description": (
                "Adversarial agent. Proposes and mutates ads into the shared "
                "queue during its turn, reacting to Investigator feedback."
            ),
            "ws": "/ws/fraudster",
            "action_schema": FraudsterAction.model_json_schema(),
            "observation_schema": FraudsterObservation.model_json_schema(),
        },
        "investigator": {
            "description": (
                "Review agent. Investigates ads via sub-tools and renders "
                "verdicts (approve/reject/escalate). Cannot see Fraudster "
                "intent — only the growing queue."
            ),
            "ws": "/ws/investigator",
            "action_schema": AdReviewAction.model_json_schema(),
            "observation_schema": AdReviewObservation.model_json_schema(),
        },
        "auditor": {
            "description": (
                "Third-agent arbiter. After the match ends, audits the "
                "Investigator's reasoning (Track A) and the Fraudster's "
                "ad plausibility (Track B). Emits flags + a final audit report."
            ),
            "ws": "/ws/auditor",
            "action_schema": AuditorAction.model_json_schema(),
            "observation_schema": AuditorObservation.model_json_schema(),
        },
    }

    return {
        "tasks": task_list,
        "action_schema": AdReviewAction.model_json_schema(),
        "roles": roles,
        "multi_agent_endpoints": {
            "fraudster_ws": "/ws/fraudster",
            "investigator_ws": "/ws/investigator",
            "auditor_ws": "/ws/auditor",
            "matches": "/matches",
            "grader": "/grader",
        },
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
    """
    Return grader score from the most recently completed episode. Prefers
    a multi-agent (Referee) result if one exists; falls back to the R1
    Investigator result otherwise.
    """
    multi_agent_result = get_last_multi_agent_grader_result()
    if multi_agent_result and multi_agent_result.get("grader_score") is not None:
        multi_agent_result.setdefault("mode", "multi_agent")
        return multi_agent_result

    result = get_last_grader_result()
    if not result:
        return {
            "error": "No completed episode. Run an episode via WebSocket or the /investigate UI.",
            "grader_score": None,
        }
    result.setdefault("mode", "single_agent")
    return result


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
