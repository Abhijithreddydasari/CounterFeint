"""
CounterFeint Public REST API — versioned under `/api/v1/*`.

This layer wraps the WebSocket environment with the kind of HTTP surface
you would expect from a production ad-fraud platform:

    GET    /api/v1/info              platform metadata (marketing blurb)
    GET    /api/v1/version           semver + build metadata
    GET    /api/v1/health            deep health (match pool, disk, env)
    GET    /api/v1/tasks             task catalog (wraps existing /tasks)
    GET    /api/v1/policies          scripted / LLM / trained policy catalog
    GET    /api/v1/schema/{role}     action + observation JSON schema
    POST   /api/v1/matches           spawn a new match, return WS URLs
    GET    /api/v1/matches           list active + recent matches
    GET    /api/v1/matches/{id}      match details
    GET    /api/v1/matches/{id}/events    per-turn event timeline
    GET    /api/v1/matches/{id}/report    final audit report
    DELETE /api/v1/matches/{id}      force-end a match
    GET    /api/v1/leaderboard       (MOCK) agent rating league
    GET    /api/v1/metrics           Prometheus-style text metrics
    POST   /api/v1/tools/policy_classifier  (MOCK) Llama Guard 3 / Purple Llama classification

Mock endpoints always include `"mock": true` in the response so judges can
see what is real state vs. what is a demo-shaped stub.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

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
    from models import (  # type: ignore[no-redef]
        AdReviewAction,
        AdReviewObservation,
        AuditorAction,
        AuditorObservation,
        FraudsterAction,
        FraudsterObservation,
    )

from .multi_agent_ws import (
    create_match_async,
    end_match_async,
    get_match_archive,
    get_match_entry,
    get_match_summary,
    list_active_matches,
    list_archived_matches,
)

BUILD_SHA = os.getenv("COUNTERFEINT_BUILD_SHA", "dev")
BUILD_TIME = os.getenv("COUNTERFEINT_BUILD_TIME", "unknown")
VERSION = "0.2.0"
SERVICE_STARTED_AT = time.time()


# ---------------------------------------------------------------------------
# Response models (feed FastAPI's OpenAPI / Swagger)
# ---------------------------------------------------------------------------


class InfoResponse(BaseModel):
    name: str = "CounterFeint FraudArena"
    tagline: str = (
        "Multi-agent adversarial ad-fraud detection platform built on OpenEnv."
    )
    description: str = (
        "Three agents (Fraudster, Investigator, Auditor) share a single "
        "environment per match. The Fraudster proposes ads in reaction to "
        "the Investigator's verdicts; the Investigator renders verdicts; "
        "the Auditor grades both sides post-hoc. Supports scripted baselines, "
        "LLM policies, and trained RL agents."
    )
    themes: List[str] = Field(
        default_factory=lambda: [
            "Multi-Agent Interactions",
            "Long-Horizon Planning & Instruction Following",
            "World Modeling",
            "Self-Improving Agents",
        ]
    )
    capabilities: List[str] = Field(
        default_factory=lambda: [
            "three-role turn-based FraudArena",
            "reactive fraudster can propose/modify ads mid-episode",
            "dual-track auditor (investigator reasoning + fraudster plausibility)",
            "scripted baseline policies included",
            "versioned REST + WebSocket APIs",
            "OpenEnv container-ready",
        ]
    )
    documentation: Dict[str, str] = Field(
        default_factory=lambda: {
            "openapi": "/docs",
            "tasks": "/api/v1/tasks",
            "policies": "/api/v1/policies",
        }
    )


class VersionResponse(BaseModel):
    version: str
    build_sha: str
    build_time: str
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    active_matches: int
    archived_matches: int
    env_name: str = "counterfeint"


class PolicyDescriptor(BaseModel):
    name: str
    role: str
    kind: str
    description: str
    source: str


class PoliciesResponse(BaseModel):
    scripted: List[PolicyDescriptor]
    llm: List[PolicyDescriptor]
    trained: List[PolicyDescriptor]


class CreateMatchRequest(BaseModel):
    seed: Optional[int] = Field(None, ge=0)
    task_id: Optional[str] = Field(None, description="Must match /api/v1/tasks id")
    max_rounds: Optional[int] = Field(None, ge=1, le=20)
    max_proposals: Optional[int] = Field(None, ge=0, le=50)
    max_fraudster_actions_per_turn: Optional[int] = Field(None, ge=1, le=10)
    max_investigator_actions_per_turn: Optional[int] = Field(None, ge=1, le=30)
    allowed_categories: Optional[List[str]] = None


class CreateMatchResponse(BaseModel):
    match_id: str
    task_id: str
    phase: str
    round_number: int
    max_rounds: int
    websocket: Dict[str, str]
    rest: Dict[str, str]


class MatchSummaryResponse(BaseModel):
    match_id: str
    task_id: str
    phase: str
    round_number: int
    max_rounds: int
    proposals_used: int
    max_proposals: int
    fraudster_committed: bool
    grader_score: Optional[float]
    end_reason: Optional[str]
    rewards: Dict[str, float]
    connected_roles: List[str]
    done: bool


class MatchListResponse(BaseModel):
    active: List[Dict[str, Any]]
    history: List[Dict[str, Any]]


class EventsResponse(BaseModel):
    match_id: str
    count: int
    events: List[Dict[str, Any]]


class ReportResponse(BaseModel):
    match_id: str
    task_id: str
    grader_score: Optional[float]
    end_reason: Optional[str]
    audit_report: Optional[Dict[str, Any]]
    rewards: Dict[str, float]
    generated_at: float


class PolicyClassifierRequest(BaseModel):
    ad_id: str = Field(
        ...,
        description=(
            "Deterministic RNG seed. Same ad_id + same ad_copy → same output "
            "(so judges can reproduce findings by re-curling the endpoint)."
        ),
    )
    ad_copy: str = Field(..., description="Ad creative body text.")
    landing_page_blurb: Optional[str] = Field(
        None, description="Optional landing-page summary text — also scanned for fraud markers."
    )
    ground_truth_label: Optional[str] = Field(
        None,
        description=(
            "Optional label for in-pipeline use ('fraud' | 'legit' | 'escalate'). "
            "External callers should leave this unset — the classifier will "
            "fall back to surface-marker heuristics."
        ),
    )
    fraud_type: Optional[str] = Field(
        None,
        description=(
            "Optional fraud_type hint (e.g. 'fake_crypto', 'counterfeit'). "
            "Only used when ground_truth_label='fraud'."
        ),
    )


class PolicyClassifierLGCategory(BaseModel):
    code: str
    name: str


class PolicyClassifierFraudMarker(BaseModel):
    code: str
    description: str


class PolicyClassifierResponse(BaseModel):
    mock: bool = True
    model: str = "llama-guard-3-8b-mock"
    ad_id: str
    verdict: str
    confidence: float
    triggered_lg_categories: List[PolicyClassifierLGCategory]
    triggered_fraud_markers: List[PolicyClassifierFraudMarker]
    explanation: str
    notes: str = (
        "Deterministic mock of Meta's Llama Guard 3 (Purple Llama) output. "
        "Weights are not loaded — see /docs and counterfeint/data/policy_classifier_data.py "
        "for the category taxonomy and marker heuristics."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fqdn_ws_url(request: Request, path: str) -> str:
    """
    Build an absolute `ws://host:port/path` URL using the request's own
    scheme + host. Falls back to the relative `path` if scheme/host are
    missing (shouldn't happen with normal ASGI servers, but keeps the
    helper defensive).
    """
    url = request.url
    host = url.netloc  # includes port, e.g. "localhost:8091"
    if not host:
        return path
    scheme = "wss" if (url.scheme or "").endswith("s") else "ws"
    return f"{scheme}://{host}{path}"


def _fqdn_http_url(request: Request, path: str) -> str:
    """Absolute `http(s)://host:port/path` URL for mock REST links."""
    url = request.url
    host = url.netloc
    if not host:
        return path
    scheme = url.scheme or "http"
    return f"{scheme}://{host}{path}"


def _build_events(match_id: str) -> List[Dict[str, Any]]:
    """Interleave fraudster + investigator logs into a flat timeline."""
    entry = get_match_entry(match_id)
    summary = entry and entry.env.state
    archive = get_match_archive(match_id)

    fraud_log: List[Dict[str, Any]] = []
    inv_log: List[Dict[str, Any]] = []
    audit_report: Optional[Dict[str, Any]] = None

    if summary is not None:
        fraud_log = list(summary.fraudster_proposals)
        inv_log = list(summary.investigator_action_log)
        audit_report = summary.audit_report
    elif archive is not None:
        fraud_log = archive.get("fraudster_proposals", [])
        inv_log = archive.get("investigator_action_log", [])
        audit_report = archive.get("audit_report")

    timeline: List[Dict[str, Any]] = []
    for entry_log in fraud_log:
        timeline.append({"type": "fraudster_action", **entry_log})
    for entry_log in inv_log:
        timeline.append({"type": "investigator_action", **entry_log})

    timeline.sort(key=lambda e: e.get("timestamp", 0.0))

    if audit_report is not None:
        timeline.append({"type": "audit_report", "report": audit_report})

    return timeline


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["Public API (v1)"])

    @router.get("/info", response_model=InfoResponse)
    async def info() -> InfoResponse:
        return InfoResponse()

    @router.get("/version", response_model=VersionResponse)
    async def version() -> VersionResponse:
        return VersionResponse(
            version=VERSION,
            build_sha=BUILD_SHA,
            build_time=BUILD_TIME,
            uptime_seconds=round(time.time() - SERVICE_STARTED_AT, 2),
        )

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            uptime_seconds=round(time.time() - SERVICE_STARTED_AT, 2),
            active_matches=len(list_active_matches()),
            archived_matches=len(list_archived_matches(limit=1_000)),
        )

    @router.get("/tasks")
    async def tasks() -> Dict[str, Any]:
        try:
            from ..data.ad_generator import TASK_CONFIGS
        except ImportError:
            from data.ad_generator import TASK_CONFIGS  # type: ignore[no-redef]
        return {
            "tasks": [
                {
                    "id": cfg.task_id,
                    "name": cfg.name,
                    "difficulty": cfg.difficulty,
                    "queue_size": cfg.queue_size,
                    "action_budget": cfg.action_budget,
                    "description": cfg.description,
                }
                for cfg in TASK_CONFIGS.values()
            ],
        }

    @router.get("/policies", response_model=PoliciesResponse)
    async def policies() -> PoliciesResponse:
        return PoliciesResponse(
            scripted=[
                PolicyDescriptor(
                    name="ScriptedFraudster",
                    role="fraudster",
                    kind="scripted",
                    description="Deterministic fraudster: canonical propose/commit sequence.",
                    source="counterfeint.scripted.ScriptedFraudster",
                ),
                PolicyDescriptor(
                    name="ReactiveFraudster",
                    role="fraudster",
                    kind="scripted",
                    description="Adapts to investigator verdicts and investigation targets.",
                    source="counterfeint.scripted.ReactiveFraudster",
                ),
                PolicyDescriptor(
                    name="GibberishFraudster",
                    role="fraudster",
                    kind="scripted",
                    description="Low-plausibility baseline emitting random gibberish copy.",
                    source="counterfeint.scripted.GibberishFraudster",
                ),
                PolicyDescriptor(
                    name="ScriptedInvestigator",
                    role="investigator",
                    kind="scripted",
                    description="Heuristic investigate-then-verdict rule-based policy.",
                    source="counterfeint.scripted.ScriptedInvestigator",
                ),
                PolicyDescriptor(
                    name="HeuristicAuditor",
                    role="auditor",
                    kind="scripted",
                    description="Rule-based dual-track auditor (Track A + Track B).",
                    source="counterfeint.scripted.HeuristicAuditor",
                ),
            ],
            llm=[
                PolicyDescriptor(
                    name="llm-investigator",
                    role="investigator",
                    kind="llm",
                    description=(
                        "OpenAI-compatible LLM investigator (see counterfeint/inference.py)."
                    ),
                    source="counterfeint.inference.run_single_task",
                ),
            ],
            trained=[
                PolicyDescriptor(
                    name="counterfeint-rl-investigator-v0",
                    role="investigator",
                    kind="trained",
                    description=(
                        "Self-play RL investigator (placeholder descriptor; "
                        "checkpoints land here in Phase 3)."
                    ),
                    source="planned",
                ),
                PolicyDescriptor(
                    name="counterfeint-rl-fraudster-v0",
                    role="fraudster",
                    kind="trained",
                    description="Self-play RL fraudster (planned).",
                    source="planned",
                ),
            ],
        )

    @router.get("/schema/{role}")
    async def schema(role: str) -> Dict[str, Any]:
        role = role.lower()
        if role == "fraudster":
            return {
                "action": FraudsterAction.model_json_schema(),
                "observation": FraudsterObservation.model_json_schema(),
            }
        if role == "investigator":
            return {
                "action": AdReviewAction.model_json_schema(),
                "observation": AdReviewObservation.model_json_schema(),
            }
        if role == "auditor":
            return {
                "action": AuditorAction.model_json_schema(),
                "observation": AuditorObservation.model_json_schema(),
            }
        raise HTTPException(
            status_code=404,
            detail=f"unknown role '{role}'; expected one of fraudster, investigator, auditor",
        )

    @router.post(
        "/matches", response_model=CreateMatchResponse, status_code=201
    )
    async def create_match(
        body: CreateMatchRequest, request: Request
    ) -> CreateMatchResponse:
        kwargs = {k: v for k, v in body.model_dump().items() if v is not None}
        try:
            entry = await create_match_async(**kwargs)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"reset failed: {exc}")

        env = entry.env
        state = env.state
        match_id = env.match_id
        return CreateMatchResponse(
            match_id=match_id,
            task_id=state.task_id,
            phase=state.phase,
            round_number=state.round_number,
            max_rounds=state.max_rounds,
            websocket={
                "fraudster": _fqdn_ws_url(
                    request, f"/ws/fraudster?match_id={match_id}"
                ),
                "investigator": _fqdn_ws_url(
                    request, f"/ws/investigator?match_id={match_id}"
                ),
                "auditor": _fqdn_ws_url(
                    request, f"/ws/auditor?match_id={match_id}"
                ),
            },
            rest={
                "summary": _fqdn_http_url(
                    request, f"/api/v1/matches/{match_id}"
                ),
                "events": _fqdn_http_url(
                    request, f"/api/v1/matches/{match_id}/events"
                ),
                "report": _fqdn_http_url(
                    request, f"/api/v1/matches/{match_id}/report"
                ),
            },
        )

    @router.get("/matches", response_model=MatchListResponse)
    async def list_matches(
        history_limit: int = Query(25, ge=0, le=200),
    ) -> MatchListResponse:
        return MatchListResponse(
            active=list_active_matches(),
            history=list_archived_matches(limit=history_limit),
        )

    @router.get("/matches/{match_id}", response_model=MatchSummaryResponse)
    async def match_detail(match_id: str) -> MatchSummaryResponse:
        summary = get_match_summary(match_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"unknown match {match_id}")
        summary = dict(summary)
        summary.setdefault("rewards", {})
        summary.pop("archived_at", None)
        summary.pop("fraudster_proposals", None)
        summary.pop("investigator_action_log", None)
        summary.pop("audit_report", None)
        return MatchSummaryResponse.model_validate(summary)

    @router.get("/matches/{match_id}/events", response_model=EventsResponse)
    async def match_events(match_id: str) -> EventsResponse:
        summary = get_match_summary(match_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"unknown match {match_id}")
        events = _build_events(match_id)
        return EventsResponse(match_id=match_id, count=len(events), events=events)

    @router.get("/matches/{match_id}/report", response_model=ReportResponse)
    async def match_report(match_id: str) -> ReportResponse:
        archive = get_match_archive(match_id)
        entry = get_match_entry(match_id)
        if archive is None and entry is None:
            raise HTTPException(status_code=404, detail=f"unknown match {match_id}")

        if entry is not None:
            state = entry.env.state
            payload = ReportResponse(
                match_id=match_id,
                task_id=state.task_id,
                grader_score=state.grader_score,
                end_reason=state.end_reason,
                audit_report=state.audit_report,
                rewards={
                    "fraudster": state.fraudster_reward,
                    "investigator": state.investigator_reward,
                    "auditor": state.auditor_reward,
                },
                generated_at=time.time(),
            )
            return payload

        assert archive is not None
        rewards = archive.get("rewards") or {}
        return ReportResponse(
            match_id=match_id,
            task_id=archive.get("task_id", ""),
            grader_score=archive.get("grader_score"),
            end_reason=archive.get("end_reason"),
            audit_report=archive.get("audit_report"),
            rewards={
                "fraudster": rewards.get("fraudster", 0.0),
                "investigator": rewards.get("investigator", 0.0),
                "auditor": rewards.get("auditor", 0.0),
            },
            generated_at=archive.get("archived_at", time.time()),
        )

    @router.delete("/matches/{match_id}", status_code=204)
    async def delete_match(match_id: str) -> Response:
        existed = await end_match_async(match_id)
        if not existed:
            raise HTTPException(status_code=404, detail=f"unknown match {match_id}")
        return Response(status_code=204)

    @router.get("/leaderboard")
    async def leaderboard() -> Dict[str, Any]:
        return {
            "mock": True,
            "note": (
                "Demonstration leaderboard — ratings are illustrative. "
                "Populated by the self-play training loop in Phase 3."
            ),
            "entries": [
                {
                    "agent": "counterfeint-rl-fraudster-v0",
                    "role": "fraudster",
                    "rating": 1624,
                    "games": 482,
                    "win_rate": 0.541,
                },
                {
                    "agent": "counterfeint-rl-investigator-v0",
                    "role": "investigator",
                    "rating": 1598,
                    "games": 482,
                    "win_rate": 0.459,
                },
                {
                    "agent": "ReactiveFraudster",
                    "role": "fraudster",
                    "rating": 1500,
                    "games": 120,
                    "win_rate": 0.483,
                },
                {
                    "agent": "ScriptedInvestigator",
                    "role": "investigator",
                    "rating": 1485,
                    "games": 120,
                    "win_rate": 0.517,
                },
                {
                    "agent": "GibberishFraudster",
                    "role": "fraudster",
                    "rating": 1183,
                    "games": 50,
                    "win_rate": 0.06,
                },
            ],
        }

    @router.post(
        "/tools/policy_classifier",
        response_model=PolicyClassifierResponse,
    )
    async def policy_classifier(
        body: PolicyClassifierRequest,
    ) -> PolicyClassifierResponse:
        """Mock Llama Guard 3 / Purple Llama classification endpoint.

        Wraps the same ``classify_ad`` helper the InvestigatorEnvironment uses
        when an Investigator calls ``investigate(policy_classifier, ad_id)``.
        Deterministic per ``ad_id`` — judges can curl this endpoint with any
        ad text to see the classifier fire live.
        """
        try:
            from ..data.policy_classifier_data import classify_ad
        except ImportError:
            from data.policy_classifier_data import classify_ad  # type: ignore[no-redef]

        result = classify_ad(
            ad_id=body.ad_id,
            ad_copy=body.ad_copy,
            landing_page_text=body.landing_page_blurb or "",
            ground_truth_label=body.ground_truth_label,
            fraud_type=body.fraud_type,
        )
        payload = result.to_dict()
        return PolicyClassifierResponse(
            ad_id=payload["ad_id"],  # type: ignore[arg-type]
            verdict=payload["verdict"],  # type: ignore[arg-type]
            confidence=float(payload["confidence"]),  # type: ignore[arg-type]
            triggered_lg_categories=[
                PolicyClassifierLGCategory(**c)  # type: ignore[arg-type]
                for c in payload["triggered_lg_categories"]  # type: ignore[index]
            ],
            triggered_fraud_markers=[
                PolicyClassifierFraudMarker(**m)  # type: ignore[arg-type]
                for m in payload["triggered_fraud_markers"]  # type: ignore[index]
            ],
            explanation=payload["explanation"],  # type: ignore[arg-type]
        )

    @router.get("/metrics", response_class=Response)
    async def metrics() -> Response:
        active = list_active_matches()
        history = list_archived_matches(limit=1_000)
        finished = [m for m in history if m.get("grader_score") is not None]
        avg_score = (
            sum(float(m.get("grader_score") or 0.0) for m in finished) / len(finished)
            if finished
            else 0.0
        )
        uptime = time.time() - SERVICE_STARTED_AT

        lines = [
            "# HELP counterfeint_uptime_seconds Service uptime in seconds.",
            "# TYPE counterfeint_uptime_seconds gauge",
            f"counterfeint_uptime_seconds {uptime:.2f}",
            "",
            "# HELP counterfeint_active_matches Number of currently-running matches.",
            "# TYPE counterfeint_active_matches gauge",
            f"counterfeint_active_matches {len(active)}",
            "",
            "# HELP counterfeint_archived_matches Number of completed matches held in history.",
            "# TYPE counterfeint_archived_matches gauge",
            f"counterfeint_archived_matches {len(history)}",
            "",
            "# HELP counterfeint_grader_score_avg Average grader score across archived matches.",
            "# TYPE counterfeint_grader_score_avg gauge",
            f"counterfeint_grader_score_avg {avg_score:.4f}",
            "",
        ]
        return Response(content="\n".join(lines), media_type="text/plain; version=0.0.4")

    return router


def register_public_api(app: FastAPI) -> None:
    """Attach the versioned public REST API to the given FastAPI app."""
    app.include_router(build_router())
