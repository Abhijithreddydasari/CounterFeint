"""
CounterFeint — Multi-Agent Ad Fraud Investigation Environment for OpenEnv.

Round 2 introduces a three-agent FraudArena:

    Fraudster     proposes / mutates ads to sneak past review
    Investigator  investigates ads and renders verdicts
    Auditor       audits the trace post-hoc for miscalibration or gibberish

All three agents share one environment instance per match and interact via
WebSockets on role-specific routes (/ws/fraudster, /ws/investigator, /ws/auditor).

Example (three-agent):

    >>> from counterfeint import MatchClient, FraudsterAction
    >>> import asyncio
    >>> async def demo():
    ...     async with MatchClient("http://localhost:8000") as match:
    ...         await match.reset(seed=42, task_id="task_1")
    ...         await match.fraudster.step(FraudsterAction(
    ...             action_type="propose_ad",
    ...             ad_copy="Free iPhone - tap here!",
    ...             category="fake_giveaway",
    ...         ))
    ...         state = await match.state()
    ...         print(state.phase, state.grader_score)
    >>> asyncio.run(demo())

Example (single-agent legacy):

    >>> from counterfeint import AdFraudEnv, AdReviewAction
    >>> with AdFraudEnv(base_url="http://localhost:8000").sync() as env:
    ...     env.reset(seed=42, task_id="task_1")
"""

from .client import (
    AdFraudEnv,
    AuditorClient,
    FraudsterClient,
    InvestigatorClient,
    MatchClient,
    MultiAgentProtocolError,
)
from .models import (
    AdFraudState,
    AdReviewAction,
    AdReviewObservation,
    AuditFlag,
    AuditorAction,
    AuditorObservation,
    AuditReport,
    FraudsterAction,
    FraudsterObservation,
    InvestigatorAction,
    InvestigatorObservation,
    InvestigatorState,
    RefereeState,
)

__all__ = [
    "AdFraudEnv",
    "AdFraudState",
    "AdReviewAction",
    "AdReviewObservation",
    "AuditFlag",
    "AuditorAction",
    "AuditorClient",
    "AuditorObservation",
    "AuditReport",
    "FraudsterAction",
    "FraudsterClient",
    "FraudsterObservation",
    "InvestigatorAction",
    "InvestigatorClient",
    "InvestigatorObservation",
    "InvestigatorState",
    "MatchClient",
    "MultiAgentProtocolError",
    "RefereeState",
]
