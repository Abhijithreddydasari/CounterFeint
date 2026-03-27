"""
Data models for the Ad Fraud Investigation Environment.

Defines the Action, Observation, and State types used by the OpenEnv interface.
The agent interacts with a queue of ads, investigating them and rendering verdicts.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class AdReviewAction(Action):
    """
    Action space for the ad fraud investigation agent.

    Three action types:
    - investigate: Spend budget to reveal information about an ad
    - verdict: Approve, reject, or escalate an ad
    - link_accounts: Flag two ads as part of the same fraud network
    """

    action_type: Literal["investigate", "verdict", "link_accounts"]
    ad_id: str = Field(..., description="Target ad identifier (e.g. 'ad_001')")

    investigation_target: Optional[
        Literal[
            "advertiser_history",
            "landing_page",
            "payment_method",
            "targeting_overlap",
            "creative_similarity",
        ]
    ] = Field(None, description="What to investigate (required for action_type='investigate')")

    verdict: Optional[Literal["approve", "reject", "escalate"]] = Field(
        None, description="Verdict decision (required for action_type='verdict')"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Agent's confidence in verdict (0.0-1.0)"
    )

    linked_ad_id: Optional[str] = Field(
        None, description="Other ad in suspected fraud ring (required for action_type='link_accounts')"
    )
    link_reason: Optional[str] = Field(
        None, description="Why the agent believes these ads are connected"
    )


class AdReviewObservation(Observation):
    """
    Observation returned after each step.

    Text-heavy by design so LLM agents can reason about the content naturally.
    Structured data is in queue_status for programmatic access.
    """

    queue_summary: str = Field(
        default="", description="Natural language overview of the ad queue"
    )
    current_ad_info: str = Field(
        default="", description="Details of the ad currently in focus"
    )
    investigation_findings: str = Field(
        default="", description="Accumulated investigation results"
    )
    verdict_history_summary: str = Field(
        default="", description="Summary of verdicts rendered so far"
    )
    feedback: str = Field(
        default="", description="Natural language feedback on the last action taken"
    )
    available_ads: List[str] = Field(
        default_factory=list, description="Ad IDs still pending review"
    )
    queue_status: Dict = Field(
        default_factory=dict,
        description="Structured status: total_ads, reviewed, pending, budget, step",
    )


class AdFraudState(State):
    """
    Internal environment state exposed via the state() property.

    Inherits episode_id and step_count from State.
    Uses extra='allow' so custom fields are permitted.
    """

    task_id: str = ""
    total_ads: int = 0
    reviewed_count: int = 0
    remaining_budget: int = 0
    verdicts: Dict = Field(default_factory=dict)
    grader_score: Optional[float] = None
