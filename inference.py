"""
Baseline inference script for the Ad Fraud Investigation Environment.

Uses the OpenAI-compatible API client to run an LLM agent against all
3 tasks, producing reproducible baseline scores.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key

Usage:
    API_BASE_URL=https://... MODEL_NAME=meta-llama/... HF_TOKEN=hf_... python inference.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

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


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    return json.loads(text)


def _create_openai_client():
    """Create an OpenAI-compatible client using the required env vars."""
    from openai import OpenAI

    api_base_url = os.environ["API_BASE_URL"]
    hf_token = os.environ["HF_TOKEN"]

    return OpenAI(api_key=hf_token, base_url=api_base_url)


def run_single_task(
    task_id: str,
    seed: int = 42,
    env_base_url: str = "http://localhost:8000",
) -> Dict[str, Any]:
    """Run the baseline agent on a single task and return the score."""
    from .client import AdFraudEnv
    from .models import AdReviewAction

    model_name = os.environ["MODEL_NAME"]
    client = _create_openai_client()
    env = AdFraudEnv(base_url=env_base_url).sync()

    try:
        env.connect()
        result = env.reset(seed=seed, task_id=task_id)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        step_count = 0
        max_steps = 100

        while not result.done and step_count < max_steps:
            obs = result.observation
            obs_text = (
                f"Queue: {obs.queue_summary}\n\n"
                f"Current Ad: {obs.current_ad_info}\n\n"
                f"Feedback: {obs.feedback}\n\n"
                f"Available ads: {', '.join(obs.available_ads)}\n\n"
                f"Verdicts so far: {obs.verdict_history_summary}"
            )
            if obs.investigation_findings:
                obs_text += f"\n\nInvestigation findings:\n{obs.investigation_findings}"

            messages.append({"role": "user", "content": obs_text})

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
            )

            assistant_msg = response.choices[0].message.content or "{}"
            messages.append({"role": "assistant", "content": assistant_msg})

            try:
                action_data = _extract_json(assistant_msg)
                action = AdReviewAction(**action_data)
            except Exception as e:
                logger.warning("Failed to parse action from LLM on step %d: %s", step_count, e)
                if obs.available_ads:
                    action = AdReviewAction(
                        action_type="verdict",
                        ad_id=obs.available_ads[0],
                        verdict="escalate",
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
    model_name = os.environ.get("MODEL_NAME", "unknown")
    results = {}
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

    return {"baseline_model": model_name, "seed": 42, "tasks": results}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    missing = []
    for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
        if not os.getenv(var):
            missing.append(var)
    if missing:
        print(
            f"Error: required environment variables not set: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    env_base_url = os.getenv("AD_FRAUD_ENV_URL", "http://localhost:8000")
    model_name = os.environ["MODEL_NAME"]

    print(f"Running baseline inference against {env_base_url} with model {model_name}...")
    scores = run_baseline(env_base_url=env_base_url)

    output_path = Path(__file__).resolve().parent / "baseline_scores.json"
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"\nBaseline scores saved to {output_path}")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
