"""
Inference Script — Ad Fraud Investigation Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

try:
    from .client import AdFraudEnv
    from .models import AdReviewAction
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ad_fraud_env.client import AdFraudEnv
    from ad_fraud_env.models import AdReviewAction


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
MAX_STEPS = 100
TEMPERATURE = 0.1
MAX_TOKENS = 256
FALLBACK_VERDICT = "escalate"

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an ad fraud investigator reviewing a queue of advertisements.
Your job is to investigate suspicious ads and render verdicts (approve, reject, or escalate).

For each step, you must output a single JSON action. The action schema is:

{
  "action_type": "investigate" | "verdict" | "link_accounts",
  "ad_id": "<ad ID, e.g. ad_001>",

  // For investigate actions:
  "investigation_target": "advertiser_history" | "landing_page" | "payment_method" | "targeting_overlap" | "creative_similarity",

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


def build_obs_prompt(obs: Any) -> str:
    """Format an observation into a user prompt for the LLM."""
    parts = [
        f"Queue: {obs.queue_summary}",
        f"Current Ad: {obs.current_ad_info}",
        f"Feedback: {obs.feedback}",
        f"Available ads: {', '.join(obs.available_ads)}",
        f"Verdicts so far: {obs.verdict_history_summary}",
    ]
    if obs.investigation_findings:
        parts.append(f"Investigation findings:\n{obs.investigation_findings}")
    return "\n\n".join(parts)


def run_single_task(
    task_id: str,
    seed: int = 42,
    env_base_url: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """Run the baseline agent on a single task and return the score."""
    base_url = os.getenv("API_BASE_URL") or API_BASE_URL
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or API_KEY
    model = os.getenv("MODEL_NAME") or MODEL_NAME

    client = OpenAI(base_url=base_url, api_key=api_key)
    env = AdFraudEnv(base_url=env_base_url).sync()

    try:
        env.connect()
        result = env.reset(seed=seed, task_id=task_id)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        step_count = 0

        while not result.done and step_count < MAX_STEPS:
            obs = result.observation
            user_prompt = build_obs_prompt(obs)
            messages.append({"role": "user", "content": user_prompt})

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
                logger.warning("Model request failed on step %d: %s", step_count, exc)
                response_text = "{}"

            messages.append({"role": "assistant", "content": response_text})

            try:
                action_data = _extract_json(response_text)
                action = AdReviewAction(**action_data)
            except Exception as e:
                logger.warning("Failed to parse action on step %d: %s", step_count, e)
                if obs.available_ads:
                    action = AdReviewAction(
                        action_type="verdict",
                        ad_id=obs.available_ads[0],
                        verdict=FALLBACK_VERDICT,
                        confidence=0.3,
                    )
                else:
                    break

            result = env.step(action)
            step_count += 1

            if len(messages) > 20:
                messages = messages[:1] + messages[-18:]

        state = env.state()
        return {
            "task_id": task_id,
            "score": state.grader_score if state.grader_score is not None else 0.0,
            "steps": step_count,
            "verdicts": state.reviewed_count,
            "total_ads": state.total_ads,
        }

    finally:
        env.close()


def run_baseline(
    env_base_url: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """Run baseline inference on all 3 tasks."""
    model = os.getenv("MODEL_NAME") or MODEL_NAME or "unknown"
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
