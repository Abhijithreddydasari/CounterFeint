"""
CounterFeint - Ad Fraud Investigation Environment Client.

Provides programmatic access via WebSocket. Users and evaluators interact
with the environment using:

    env = AdFraudEnv.from_hub("your-org/counterfeint")
    # or
    env = AdFraudEnv(base_url="http://localhost:8000")
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import AdFraudState, AdReviewAction, AdReviewObservation


class AdFraudEnv(EnvClient[AdReviewAction, AdReviewObservation, AdFraudState]):
    """
    WebSocket client for the Ad Fraud Investigation Environment.

    Example:
        >>> with AdFraudEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(seed=42, task_id="task_1")
        ...     print(result.observation.queue_summary)
        ...     result = env.step(AdReviewAction(
        ...         action_type="investigate",
        ...         ad_id="ad_001",
        ...         investigation_target="advertiser_history",
        ...     ))
        ...     print(result.observation.feedback)
    """

    def _step_payload(self, action: AdReviewAction) -> Dict[str, Any]:
        """Convert action to JSON payload for the WebSocket step message."""
        return action.model_dump(exclude_none=True, exclude={"metadata"})

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[AdReviewObservation]:
        """Parse server response into a typed StepResult.

        OpenEnv serializes as {"observation": {...}, "reward": float, "done": bool}
        with reward/done at the top level, excluded from the observation dict.
        """
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

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AdFraudState:
        """Parse server state response into AdFraudState."""
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
