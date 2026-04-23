"""
Inference driver — CounterFeint FraudArena.

Supports two modes:

* single-agent (R1 compatibility): LLM-driven Investigator plays alone against
  `/ws`; preserved verbatim for judges scoring R1 regressions.
* three-agent (R2 default):        scripted Fraudster / Investigator / Auditor
  policies drive a match through `/ws/fraudster`, `/ws/investigator`, and
  `/ws/auditor`, producing a full trace with role+round-prefixed [STEP] logs.

Mode is selected via `COUNTERFEINT_MODE=single-agent|three-agent` (default
`three-agent`). The R1 mandatory STDOUT format is preserved:

    [START] task=<task> env=counterfeint mode=<mode> model=<name>
    [STEP]  ...role+round-annotated line per agent action...
    [END]   success=<bool> steps=<n> score=<float> rewards=<...>

Environment variables (R1 mode):

    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    COUNTERFEINT_ENV_URL  Base URL of the CounterFeint server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from .client import AdFraudEnv, MatchClient
    from .data.ad_generator import TASK_CONFIGS
    from .models import AdReviewAction, AuditorAction, FraudsterAction
    from .scripted import (
        HeuristicAuditor,
        ReactiveFraudster,
        ScriptedFraudster,
        ScriptedInvestigator,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from counterfeint.client import AdFraudEnv, MatchClient
    from counterfeint.data.ad_generator import TASK_CONFIGS
    from counterfeint.models import AdReviewAction, AuditorAction, FraudsterAction
    from counterfeint.scripted import (
        HeuristicAuditor,
        ReactiveFraudster,
        ScriptedFraudster,
        ScriptedInvestigator,
    )

from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("COUNTERFEINT_ENV_URL", "http://localhost:8000")
MODE = os.getenv("COUNTERFEINT_MODE", "three-agent").strip().lower()
BENCHMARK = "counterfeint"
TEMPERATURE = 0.1
MAX_TOKENS = 256
FALLBACK_VERDICT = "escalate"

logger = logging.getLogger(__name__)
LOG_DIR = Path(__file__).resolve().parent / "convo_logging"


# ---------------------------------------------------------------------------
# Mandatory structured stdout logging
# ---------------------------------------------------------------------------


def log_start(task: str, mode: str, model: str) -> None:
    print(
        f"[START] task={task} env={BENCHMARK} mode={mode} model={model}",
        flush=True,
    )


def log_step_r1(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:+.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_step_r2(
    step: int,
    role: str,
    round_number: int,
    action: str,
    reward: float,
    phase: str,
    done: bool,
    error: Optional[str] = None,
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} role={role} round={round_number} "
        f"action={action} reward={reward:+.2f} phase={phase} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end_r1(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:+.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def log_end_r2(
    success: bool,
    steps: int,
    rounds_played: int,
    grader_score: float,
    rewards_by_role: Dict[str, float],
    end_reason: Optional[str],
) -> None:
    role_rewards = " ".join(
        f"{role}_reward={val:+.2f}" for role, val in rewards_by_role.items()
    )
    print(
        f"[END] mode=three-agent success={str(success).lower()} steps={steps} "
        f"rounds={rounds_played} score={grader_score:.2f} "
        f"{role_rewards} end_reason={end_reason or 'unknown'}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# R1: action formatter (LLM-driven investigator)
# ---------------------------------------------------------------------------


def _format_investigator_action(action: AdReviewAction) -> str:
    if action.action_type == "investigate":
        return f"investigate({action.ad_id},{action.investigation_target})"
    if action.action_type == "verdict":
        conf = action.confidence if action.confidence is not None else 0.5
        return f"verdict({action.ad_id},{action.verdict},{conf:.2f})"
    if action.action_type == "link_accounts":
        return f"link_accounts({action.ad_id},{action.linked_ad_id})"
    return f"unknown({action.ad_id})"


def _format_fraudster_action(action: FraudsterAction) -> str:
    if action.action_type == "propose_ad":
        return f"propose_ad({action.category or '-'})"
    if action.action_type == "modify_pending_ad":
        return f"modify_pending_ad(slot={action.slot_index})"
    if action.action_type == "end_turn":
        return "end_turn"
    if action.action_type == "commit_final":
        return "commit_final"
    return action.action_type


def _format_auditor_action(action: AuditorAction) -> str:
    if action.action_type == "flag_investigator":
        return f"flag_investigator({action.flag_type or '-'},sev={action.severity or 0:.2f})"
    if action.action_type == "flag_fraudster":
        return f"flag_fraudster({action.flag_type or '-'},sev={action.severity or 0:.2f})"
    if action.action_type == "submit_audit_report":
        return "submit_audit_report"
    return action.action_type


def _format_action(action: Any) -> str:
    if isinstance(action, AdReviewAction):
        return _format_investigator_action(action)
    if isinstance(action, FraudsterAction):
        return _format_fraudster_action(action)
    if isinstance(action, AuditorAction):
        return _format_auditor_action(action)
    return str(action)


# ---------------------------------------------------------------------------
# R1: LLM-driven investigator (kept for baseline regression tests)
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
You are an ad fraud investigator reviewing a queue of advertisements.
Your job is to investigate suspicious ads and render verdicts (approve, reject, or escalate).

For each step, you must output a single JSON action. The action schema is:

{
  "action_type": "investigate" | "verdict" | "link_accounts",
  "ad_id": "<ad ID, e.g. ad_001>",
  "investigation_target": "advertiser_history" | "landing_page" | "payment_method" | "targeting_overlap" | "campaign_structure" | "policy_classifier",
  "verdict": "approve" | "reject" | "escalate",
  "confidence": <float 0.0-1.0>,
  "linked_ad_id": "<other ad ID>",
  "link_reason": "<reason>"
}

Strategy:
1. Start by reading the queue summary and the first ad's information.
2. For obviously suspicious ads, investigate 1-2 signals then reject.
3. For clearly legitimate ads, approve quickly with high confidence.
4. For ambiguous ads, investigate more deeply before deciding.
5. Manage your budget — you cannot investigate everything.
6. For link_accounts, only flag connections when you see shared signals across ads.

Output ONLY the JSON action, no other text.
"""

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
_FINDING_BLOCK_RE = re.compile(r"\[(ad_\d+)\s*/\s*([a-z_]+)\]")
_ANALYSIS_HEADER_RE = re.compile(
    r"^(?:Payment Method|Targeting|Campaign Structure) Analysis for ad_\d+:$"
    r"|^Llama Guard 3 Classification for ad_\d+:$"
)


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    m = JSON_BLOCK_RE.search(text)
    if m:
        text = m.group(1).strip()
    elif text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _compact_finding(text: str) -> str:
    parts: List[str] = []
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _ANALYSIS_HEADER_RE.match(stripped):
            continue
        if stripped in ("Key claims on page:", "Suspicious elements:"):
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:]
        parts.append(stripped)
    return " | ".join(parts)


def _build_compact_findings(raw: str, focused_ad: Optional[str]) -> str:
    blocks: List[tuple] = []
    current_ad: Optional[str] = None
    current_header: Optional[str] = None
    current_lines: List[str] = []

    for line in raw.split("\n"):
        m = _FINDING_BLOCK_RE.match(line.strip())
        if m:
            if current_ad is not None:
                blocks.append((current_ad, current_header, "\n".join(current_lines)))
            current_ad = m.group(1)
            current_header = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_ad is not None:
        blocks.append((current_ad, current_header, "\n".join(current_lines)))

    result: List[str] = []
    for ad_id, header, text in blocks:
        if focused_ad and ad_id == focused_ad:
            result.append(f"\n{header}\n{text}")
        else:
            compact = _compact_finding(text)
            if compact:
                result.append(f"{header} {compact}")
    return "\n".join(result)


def build_obs_prompt(obs: Any) -> str:
    focused_ad: Optional[str] = None
    if obs.current_ad_info:
        m = re.search(r"Ad in Focus:\s*(ad_\d+)", obs.current_ad_info)
        if m:
            focused_ad = m.group(1)

    parts = [
        f"Queue: {obs.queue_summary}",
        f"Current Ad: {obs.current_ad_info}",
        f"Feedback: {obs.feedback}",
        f"Available ads: {', '.join(obs.available_ads)}",
    ]
    if obs.verdict_history_summary and obs.verdict_history_summary != "No verdicts yet.":
        parts.append(f"Verdicts: {obs.verdict_history_summary}")
    if obs.investigation_findings:
        findings = _build_compact_findings(obs.investigation_findings, focused_ad)
        if findings:
            parts.append(f"Findings:\n{findings}")
    return "\n\n".join(parts)


class EpisodeLogger:
    """Logs the full agent-environment conversation to a markdown file."""

    def __init__(self, task_id: str, log_dir: Path, mode: str = "single-agent") -> None:
        self.task_id = task_id
        self.lines: List[str] = []
        log_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_three_agent" if mode == "three-agent" else ""
        self.path = log_dir / f"{task_id}{suffix}_conversation.md"
        self._md(f"# Episode Log — {task_id} ({mode})\n")

    def step_start(self, step: int, obs_prompt: str) -> None:
        self._md(f"\n## Step {step}\n")
        self._md(f"### Observation (sent to LLM)\n```\n{obs_prompt}\n```\n")

    def llm_response(
        self, step: int, raw: str, action: Optional[AdReviewAction], fallback: bool
    ) -> None:
        tag = " [FALLBACK]" if fallback else ""
        self._md(f"### LLM Response\n```json\n{raw.strip()}\n```\n")
        if action:
            act_dict = action.model_dump(exclude_none=True, exclude={"metadata"})
            self._md(f"### Parsed Action{tag}\n```json\n{json.dumps(act_dict, indent=2)}\n```\n")

    def env_feedback(self, step: int, reward: float, done: bool, feedback: str) -> None:
        self._md(
            f"### Environment Response\n- **Reward:** `{reward:+.2f}`\n"
            f"- **Done:** `{done}`\n"
        )
        self._md(f"- **Feedback:** {feedback}\n")

    def role_turn(
        self, step: int, role: str, action_str: str, reward: float, phase: str
    ) -> None:
        self._md(
            f"\n## Step {step} — {role.upper()}\n"
            f"- Action: `{action_str}`\n"
            f"- Reward: `{reward:+.2f}`\n"
            f"- New phase: `{phase}`\n"
        )

    def episode_end(self, score: float, steps: int, verdicts: int, total: int) -> None:
        summary = f"Score: {score:.3f} | Steps: {steps} | Verdicts: {verdicts}/{total}"
        self._md(f"\n---\n## Result\n**{summary}**\n")
        self._flush()

    def episode_end_r2(self, state: Dict[str, Any]) -> None:
        lines = [
            "\n---\n## Result (3-agent)\n",
            f"- grader_score: `{state.get('grader_score')}`",
            f"- fraudster_reward: `{state.get('fraudster_reward')}`",
            f"- investigator_reward: `{state.get('investigator_reward')}`",
            f"- auditor_reward: `{state.get('auditor_reward')}`",
            f"- end_reason: `{state.get('end_reason')}`",
        ]
        self._md("\n".join(lines))
        self._flush()

    def _md(self, text: str) -> None:
        self.lines.append(text)

    def _flush(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))


def run_single_task(
    task_id: str,
    seed: int = 42,
    env_base_url: str = ENV_URL,
) -> Dict[str, Any]:
    """R1 LLM-driven single-agent inference. Unchanged stdout contract."""
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=60.0)
    env = AdFraudEnv(base_url=env_base_url).sync()
    elog = EpisodeLogger(task_id, LOG_DIR, mode="single-agent")

    config = TASK_CONFIGS.get(task_id)
    max_steps = config.action_budget if config else 25

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, mode="single-agent", model=MODEL_NAME)

    try:
        env.connect()
        result = env.reset(seed=seed, task_id=task_id)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        while not result.done and steps_taken < max_steps:
            obs = result.observation
            user_prompt = build_obs_prompt(obs)
            messages.append({"role": "user", "content": user_prompt})

            elog.step_start(steps_taken, user_prompt)

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or "{}"
            except Exception as exc:
                logger.warning("Model request failed on step %d: %s", steps_taken, exc)
                response_text = "{}"

            messages.append({"role": "assistant", "content": response_text})

            error_msg = None
            fallback = False
            try:
                action_data = _extract_json(response_text)
                action = AdReviewAction(**action_data)
            except Exception as e:
                logger.warning("Failed to parse action on step %d: %s", steps_taken, e)
                fallback = True
                error_msg = str(e)
                if obs.available_ads:
                    action = AdReviewAction(
                        action_type="verdict",
                        ad_id=obs.available_ads[0],
                        verdict=FALLBACK_VERDICT,
                        confidence=0.3,
                    )
                else:
                    elog.llm_response(steps_taken, response_text, None, True)
                    break

            elog.llm_response(steps_taken, response_text, action, fallback)

            result = env.step(action)
            steps_taken += 1
            reward = result.reward or 0.0
            rewards.append(reward)

            log_step_r1(
                step=steps_taken,
                action=_format_investigator_action(action),
                reward=reward,
                done=result.done,
                error=error_msg,
            )

            elog.env_feedback(
                steps_taken, reward, result.done, result.observation.feedback
            )

            if len(messages) > 6:
                messages = messages[:1] + messages[-4:]

        state = env.state()
        score = state.grader_score if state.grader_score is not None else 0.0
        score = max(0.0, min(1.0, score))
        success = score > 0.0

        elog.episode_end(score, steps_taken, state.reviewed_count, state.total_ads)

        return {
            "task_id": task_id,
            "score": score,
            "steps": steps_taken,
            "verdicts": state.reviewed_count,
            "total_ads": state.total_ads,
        }

    except Exception as e:
        logger.error("Task %s failed: %s", task_id, e)
        return {"task_id": task_id, "score": 0.0, "error": str(e)}

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr, flush=True)
        log_end_r1(success=success, steps=steps_taken, score=score, rewards=rewards)


def run_baseline(
    env_base_url: str = ENV_URL,
) -> Dict[str, Any]:
    """Run the R1 LLM baseline across all tasks."""
    results: Dict[str, Any] = {}
    for task_id in ("task_1", "task_2", "task_3"):
        logger.info("Running baseline for %s...", task_id)
        try:
            task_result = run_single_task(task_id, seed=42, env_base_url=env_base_url)
            results[task_id] = task_result
            logger.info("  %s score: %.3f", task_id, task_result["score"])
        except Exception as e:
            logger.error("  %s failed: %s", task_id, e)
            results[task_id] = {"task_id": task_id, "score": 0.0, "error": str(e)}
    return {"baseline_model": MODEL_NAME, "seed": 42, "tasks": results}


# ---------------------------------------------------------------------------
# R2: three-agent driver (scripted policies)
# ---------------------------------------------------------------------------


PHASE_TO_ROLE = {
    "fraudster_turn": "fraudster",
    "investigator_turn": "investigator",
    "audit_phase": "auditor",
}


async def arun_three_agent_episode(
    task_id: str,
    *,
    fraudster_policy: Any,
    investigator_policy: Any,
    auditor_policy: Any,
    env_base_url: str = ENV_URL,
    seed: int = 42,
    max_steps: int = 200,
    reset_kwargs: Optional[Dict[str, Any]] = None,
    log: bool = True,
) -> Dict[str, Any]:
    """
    Drive a full three-agent match end-to-end.

    Returns a dict with final state, rewards, and steps metadata. Emits the
    mandatory [START]/[STEP]/[END] STDOUT lines when `log=True`.
    """
    elog = EpisodeLogger(task_id, LOG_DIR, mode="three-agent") if log else None
    reset_kwargs = dict(reset_kwargs or {})
    reset_kwargs.setdefault("seed", seed)
    reset_kwargs.setdefault("task_id", task_id)

    for policy in (fraudster_policy, investigator_policy, auditor_policy):
        if hasattr(policy, "reset"):
            policy.reset()

    model_tag = (
        f"fraudster={type(fraudster_policy).__name__}|"
        f"investigator={type(investigator_policy).__name__}|"
        f"auditor={type(auditor_policy).__name__}"
    )

    if log:
        log_start(task=task_id, mode="three-agent", model=model_tag)

    step_idx = 0
    rewards_by_role: Dict[str, List[float]] = {
        "fraudster": [],
        "investigator": [],
        "auditor": [],
    }
    final_state: Dict[str, Any] = {}
    end_reason: Optional[str] = None
    success = False

    async with MatchClient(env_base_url) as match:
        initial_obs = await match.reset(**reset_kwargs)
        current_obs: Dict[str, Dict[str, Any]] = {"fraudster": initial_obs}

        while step_idx < max_steps:
            state_payload = await match.fraudster.state()
            phase = state_payload.get("phase", "done")
            if phase == "done":
                final_state = state_payload
                end_reason = state_payload.get("end_reason")
                break

            role = PHASE_TO_ROLE.get(phase)
            if role is None:
                logger.warning("Unknown phase %r; ending loop.", phase)
                final_state = state_payload
                break

            client_for_role = {
                "fraudster": match.fraudster,
                "investigator": match.investigator,
                "auditor": match.auditor,
            }[role]
            policy_for_role = {
                "fraudster": fraudster_policy,
                "investigator": investigator_policy,
                "auditor": auditor_policy,
            }[role]

            obs_payload = current_obs.get(role) or await client_for_role.obs()
            try:
                action = policy_for_role.act(obs_payload)
            except Exception as exc:
                logger.exception("Policy for %s raised: %s", role, exc)
                if log:
                    log_step_r2(
                        step=step_idx + 1,
                        role=role,
                        round_number=int(state_payload.get("round_number", 0)),
                        action="policy_error",
                        reward=0.0,
                        phase=phase,
                        done=False,
                        error=str(exc),
                    )
                break

            try:
                step_resp = await client_for_role.step(action)
            except Exception as exc:
                logger.exception("Step failed for %s: %s", role, exc)
                if log:
                    log_step_r2(
                        step=step_idx + 1,
                        role=role,
                        round_number=int(state_payload.get("round_number", 0)),
                        action=_format_action(action),
                        reward=0.0,
                        phase=phase,
                        done=False,
                        error=str(exc),
                    )
                break

            reward_val = float(step_resp.get("reward") or 0.0)
            done_val = bool(step_resp.get("done", False))
            new_phase = step_resp.get("phase", phase)
            round_num = int(step_resp.get("round_number", state_payload.get("round_number", 0)))
            rewards_by_role[role].append(reward_val)

            current_obs[role] = step_resp
            for other_role in ("fraudster", "investigator", "auditor"):
                if other_role != role:
                    current_obs.pop(other_role, None)

            step_idx += 1
            action_str = _format_action(action)

            if log:
                log_step_r2(
                    step=step_idx,
                    role=role,
                    round_number=round_num,
                    action=action_str,
                    reward=reward_val,
                    phase=new_phase,
                    done=done_val,
                )
                if elog is not None:
                    elog.role_turn(step_idx, role, action_str, reward_val, new_phase)

            if done_val or new_phase == "done":
                final_state = await match.fraudster.state()
                end_reason = final_state.get("end_reason")
                break

        if not final_state:
            final_state = await match.fraudster.state()
            end_reason = final_state.get("end_reason")

    grader_score = final_state.get("grader_score") or 0.0
    grader_score = max(0.0, min(1.0, float(grader_score)))
    success = grader_score > 0.0 and final_state.get("phase") == "done"

    role_totals = {role: sum(vals) for role, vals in rewards_by_role.items()}

    if log:
        log_end_r2(
            success=success,
            steps=step_idx,
            rounds_played=int(final_state.get("round_number", 0)),
            grader_score=grader_score,
            rewards_by_role=role_totals,
            end_reason=end_reason,
        )
        if elog is not None:
            elog.episode_end_r2(final_state)

    return {
        "task_id": task_id,
        "mode": "three-agent",
        "grader_score": grader_score,
        "steps": step_idx,
        "end_reason": end_reason,
        "rewards_by_role": role_totals,
        "final_state": final_state,
    }


def run_three_agent_episode(
    task_id: str,
    *,
    fraudster_policy: Optional[Any] = None,
    investigator_policy: Optional[Any] = None,
    auditor_policy: Optional[Any] = None,
    env_base_url: str = ENV_URL,
    seed: int = 42,
    max_steps: int = 200,
    reset_kwargs: Optional[Dict[str, Any]] = None,
    log: bool = True,
) -> Dict[str, Any]:
    """Synchronous wrapper around `arun_three_agent_episode` using scripted defaults."""
    return asyncio.run(
        arun_three_agent_episode(
            task_id,
            fraudster_policy=fraudster_policy or ReactiveFraudster(seed=seed),
            investigator_policy=investigator_policy or ScriptedInvestigator(),
            auditor_policy=auditor_policy or HeuristicAuditor(),
            env_base_url=env_base_url,
            seed=seed,
            max_steps=max_steps,
            reset_kwargs=reset_kwargs,
            log=log,
        )
    )


def run_three_agent_baseline(
    env_base_url: str = ENV_URL,
) -> Dict[str, Any]:
    """Run the scripted 3-agent baseline across all three task configurations."""
    results: Dict[str, Any] = {}
    for task_id in ("task_1", "task_2", "task_3"):
        logger.info("Running 3-agent baseline for %s...", task_id)
        try:
            task_result = run_three_agent_episode(
                task_id, seed=42, env_base_url=env_base_url,
            )
            results[task_id] = task_result
            logger.info("  %s score: %.3f", task_id, task_result["grader_score"])
        except Exception as e:
            logger.error("  %s failed: %s", task_id, e)
            results[task_id] = {"task_id": task_id, "grader_score": 0.0, "error": str(e)}
    return {
        "baseline_type": "three-agent-scripted",
        "seed": 42,
        "tasks": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if MODE == "single-agent":
        if not HF_TOKEN:
            print("Error: HF_TOKEN environment variable is required for single-agent mode.", file=sys.stderr)
            sys.exit(1)
        print(
            f"Running R1 single-agent inference against {ENV_URL} with model {MODEL_NAME}...",
            file=sys.stderr,
        )
        scores = run_baseline(env_base_url=ENV_URL)
        output_path = Path(__file__).resolve().parent / "baseline_scores.json"
    elif MODE == "three-agent":
        print(
            f"Running R2 three-agent scripted baseline against {ENV_URL}...",
            file=sys.stderr,
        )
        scores = run_three_agent_baseline(env_base_url=ENV_URL)
        output_path = Path(__file__).resolve().parent / "baseline_scores_r2.json"
    else:
        print(
            f"Unknown COUNTERFEINT_MODE={MODE!r}; expected 'single-agent' or 'three-agent'.",
            file=sys.stderr,
        )
        sys.exit(2)

    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\nBaseline scores saved to {output_path}", file=sys.stderr)
    print(json.dumps(scores, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
