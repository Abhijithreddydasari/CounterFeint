"""
Inference Script — Ad Fraud Investigation Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

try:
    from .client import AdFraudEnv
    from .models import AdReviewAction
    from .data.ad_generator import TASK_CONFIGS
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ad_fraud_env.client import AdFraudEnv
    from ad_fraud_env.models import AdReviewAction
    from ad_fraud_env.data.ad_generator import TASK_CONFIGS

from dotenv import load_dotenv
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
ENV_URL = os.getenv("AD_FRAUD_ENV_URL", "http://localhost:8000")
BENCHMARK = "ad_fraud_env"
TEMPERATURE = 0.1
MAX_TOKENS = 256
FALLBACK_VERDICT = "escalate"

logger = logging.getLogger(__name__)
LOG_DIR = Path(__file__).resolve().parent / "convo_logging"

# ---------------------------------------------------------------------------
# Mandatory structured stdout logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _format_action(action: AdReviewAction) -> str:
    """Compact action string for [STEP] log line."""
    if action.action_type == "investigate":
        return f"investigate({action.ad_id},{action.investigation_target})"
    elif action.action_type == "verdict":
        conf = action.confidence if action.confidence is not None else 0.5
        return f"verdict({action.ad_id},{action.verdict},{conf:.2f})"
    elif action.action_type == "link_accounts":
        return f"link_accounts({action.ad_id},{action.linked_ad_id})"
    return f"unknown({action.ad_id})"

# ---------------------------------------------------------------------------
# System prompt & helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an ad fraud investigator reviewing a queue of advertisements.
Your job is to investigate suspicious ads and render verdicts (approve, reject, or escalate).

For each step, you must output a single JSON action. The action schema is:

{
  "action_type": "investigate" | "verdict" | "link_accounts",
  "ad_id": "<ad ID, e.g. ad_001>",

  // For investigate actions:
  "investigation_target": "advertiser_history" | "landing_page" | "payment_method" | "targeting_overlap" | "creative_similarity" | "campaign_structure",

  // For verdict actions:
  "verdict": "approve" | "reject" | "escalate",
  "confidence": <float 0.0-1.0>,

  // For link_accounts actions:
  "linked_ad_id": "<other ad ID>",
  "link_reason": "<reason>"
}

Strategy:
1. Start by reading the queue summary and the first ad's information.
2. For obviously suspicious ads, investigate 1-2 signals then reject.
3. For clearly legitimate ads, approve quickly with high confidence.
4. For ambiguous ads, investigate more deeply before deciding.
5. Manage your budget — you cannot investigate everything.
6. For link_accounts, only flag connections when you see shared signals across ads (same payment method, similar creative template, targeting overlap).

Output ONLY the JSON action, no other text.
"""

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
_FINDING_BLOCK_RE = re.compile(r"\[(ad_\d+)\s*/\s*([a-z_]+)\]")
_ANALYSIS_HEADER_RE = re.compile(
    r"^(?:Payment Method|Targeting|Creative|Campaign Structure) Analysis for ad_\d+:$"
)


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
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
    """Compress one investigation finding block into a compact key-value line.

    Strips analysis headers and section labels but preserves ALL data fields
    (both signal and noise) so the agent must still reason about relevance.
    """
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
    """Parse findings blocks; full prose for focused ad, compact lines for others.

    Each block in the raw findings string is delimited by [ad_XXX / target].
    The focused ad keeps full investigation prose for deep reasoning.
    Other ads are compressed to single-line key-value summaries that still
    contain all extracted fields (payment type, IDs, template hashes,
    fingerprints, domain ages, etc.) mixed with noise fields — the agent
    must determine which values are meaningful for cross-ad comparison.
    """
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
    """Format an observation into the user prompt for the LLM.

    The focused ad gets full investigation prose for deep reasoning.
    Other investigated ads are compressed to compact key-value summaries
    that preserve all fields (signal + noise) for cross-ad comparison.
    """
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

# ---------------------------------------------------------------------------
# Episode logger (markdown, for debugging — separate from mandatory stdout)
# ---------------------------------------------------------------------------

class EpisodeLogger:
    """Logs the full agent-environment conversation to a markdown file."""

    def __init__(self, task_id: str, log_dir: Path) -> None:
        self.task_id = task_id
        self.lines: List[str] = []
        log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{task_id}_conversation.md"
        self._md(f"# Episode Log — {task_id}\n")

    def step_start(self, step: int, obs_prompt: str) -> None:
        self._md(f"\n## Step {step}\n")
        self._md(f"### Observation (sent to LLM)\n```\n{obs_prompt}\n```\n")

    def llm_response(self, step: int, raw: str, action: AdReviewAction | None, fallback: bool) -> None:
        tag = " [FALLBACK]" if fallback else ""
        self._md(f"### LLM Response\n```json\n{raw.strip()}\n```\n")
        if action:
            act_dict = action.model_dump(exclude_none=True, exclude={"metadata"})
            self._md(f"### Parsed Action{tag}\n```json\n{json.dumps(act_dict, indent=2)}\n```\n")

    def env_feedback(self, step: int, reward: float, done: bool, feedback: str) -> None:
        self._md(f"### Environment Response\n- **Reward:** `{reward:+.2f}`\n- **Done:** `{done}`\n")
        self._md(f"- **Feedback:** {feedback}\n")

    def episode_end(self, score: float, steps: int, verdicts: int, total: int) -> None:
        summary = f"Score: {score:.3f} | Steps: {steps} | Verdicts: {verdicts}/{total}"
        self._md(f"\n---\n## Result\n**{summary}**\n")
        self._flush()

    def _md(self, text: str) -> None:
        self.lines.append(text)

    def _flush(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))

# ---------------------------------------------------------------------------
# Core task runner
# ---------------------------------------------------------------------------

def run_single_task(
    task_id: str,
    seed: int = 42,
    env_base_url: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """Run the baseline agent on a single task with mandatory [START]/[STEP]/[END] logging."""
    base_url = os.getenv("API_BASE_URL") or API_BASE_URL
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or API_KEY
    model = os.getenv("MODEL_NAME") or MODEL_NAME

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
    env = AdFraudEnv(base_url=env_base_url).sync()
    elog = EpisodeLogger(task_id, LOG_DIR)

    config = TASK_CONFIGS.get(task_id)
    max_steps = config.action_budget if config else 25

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=model)

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
                    model=model,
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

            log_step(
                step=steps_taken,
                action=_format_action(action),
                reward=reward,
                done=result.done,
                error=error_msg,
            )

            elog.env_feedback(steps_taken, reward, result.done, result.observation.feedback)

            # Each observation is self-contained (all findings + verdict summary),
            # so we only keep system prompt + last 2 exchanges to stay within
            # context limits while preserving the agent's reasoning continuity.
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
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def run_baseline(
    env_base_url: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """Run baseline inference on all 3 tasks."""
    model = os.getenv("MODEL_NAME") or MODEL_NAME
    results: Dict[str, Any] = {}
    for task_id in ["task_1", "task_2", "task_3"]:
        logger.info("Running baseline for %s...", task_id)
        try:
            task_result = run_single_task(
                task_id, seed=42, env_base_url=env_base_url,
            )
            results[task_id] = task_result
            logger.info("  %s score: %.3f", task_id, task_result["score"])
        except Exception as e:
            logger.error("  %s failed: %s", task_id, e)
            results[task_id] = {"task_id": task_id, "score": 0.0, "error": str(e)}

    return {"baseline_model": model, "seed": 42, "tasks": results}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not API_KEY:
        print("Error: HF_TOKEN (or API_KEY) environment variable is required.", file=sys.stderr)
        sys.exit(1)
    if not MODEL_NAME:
        print("Error: MODEL_NAME environment variable is required.", file=sys.stderr)
        sys.exit(1)

    env_base_url = os.getenv("AD_FRAUD_ENV_URL", "http://localhost:8000")

    print(f"Running baseline inference against {env_base_url} with model {MODEL_NAME}...")
    scores = run_baseline(env_base_url=env_base_url)

    output_path = Path(__file__).resolve().parent / "baseline_scores.json"
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\nBaseline scores saved to {output_path}")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
