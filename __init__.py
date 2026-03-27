"""
Ad Fraud Investigation Environment for OpenEnv.

A real-world simulation where an AI agent investigates a queue of ads
for fraud, making sequential investigation and verdict decisions under
a limited action budget.

Example:
    >>> from ad_fraud_env import AdFraudEnv, AdReviewAction
    >>>
    >>> with AdFraudEnv(base_url="http://localhost:8000").sync() as env:
    ...     result = env.reset(seed=42, task_id="task_1")
    ...     result = env.step(AdReviewAction(
    ...         action_type="investigate",
    ...         ad_id="ad_001",
    ...         investigation_target="landing_page",
    ...     ))
    ...     print(result.observation.feedback)
"""

from .client import AdFraudEnv
from .models import AdFraudState, AdReviewAction, AdReviewObservation

__all__ = [
    "AdFraudEnv",
    "AdReviewAction",
    "AdReviewObservation",
    "AdFraudState",
]
