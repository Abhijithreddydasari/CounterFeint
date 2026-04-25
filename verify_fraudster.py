"""
Live diagnostic for the LLM Fraudster against Ollama.

Goal
----
Prove that the LLM Fraudster (1) receives the right per-turn observation,
(2) produces a *schema-valid* JSON action, and (3) reacts to the Investigator
in subsequent turns — *without* writing a permanent test that needs Ollama
to be up in CI.

What it does
~~~~~~~~~~~~
For each task tier (``task_1``, ``task_2``, ``task_3``):
  1. Spin up an in-process ``RefereeEnvironment`` (no WebSocket).
  2. Build the real Fraudster observation.
  3. Call Ollama once via ``LLMFraudster._call_llm_once`` (no retries).
  4. Print the SYSTEM prompt size, the USER prompt, the RAW response, and
     the PARSED action (or schema error).
  5. Step the env with that action so the next turn's observation reflects
     the real consequence (queue grows, proposals_remaining decrements).
  6. After the Fraudster's first turn, run one scripted Investigator turn so
     the Fraudster's *next* turn observation contains a verdict — proves
     the reactive feedback loop is wired end-to-end.
  7. Repeat steps 2-5 for the second Fraudster turn.

Run it
~~~~~~
::

    # ollama serve must be running with llama3.1:8b pulled
    python -m counterfeint.verify_fraudster

You can override the model + endpoint via env vars:
    MODEL_NAME=llama3.1:8b
    API_BASE_URL=http://localhost:11434/v1
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    # Allow `python counterfeint/verify_fraudster.py` as well as `python -m counterfeint.verify_fraudster`
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from counterfeint.agents import LLMFraudster
    from counterfeint.agents.prompts import FRAUDSTER_SYSTEM_PROMPT
    from counterfeint.models import AdReviewAction, FraudsterAction
    from counterfeint.scripted import ScriptedInvestigator
    from counterfeint.server.referee import RefereeEnvironment
else:
    from .agents import LLMFraudster
    from .agents.prompts import FRAUDSTER_SYSTEM_PROMPT
    from .models import AdReviewAction, FraudsterAction
    from .scripted import ScriptedInvestigator
    from .server.referee import RefereeEnvironment


_OLLAMA_BASE = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
_OLLAMA_MODEL = os.getenv("MODEL_NAME", "llama3.1:8b")
_TURNS_PER_TASK = 2


def _hr(title: str = "") -> None:
    bar = "=" * 78
    if title:
        print(f"\n{bar}\n{title}\n{bar}")
    else:
        print(bar)


def _wrap(text: str, indent: str = "  ") -> str:
    return textwrap.indent(text, indent)


def _build_fraudster() -> LLMFraudster:
    """Construct the Fraudster pointed at Ollama with retries=0 for clarity."""
    return LLMFraudster(
        model_name=_OLLAMA_MODEL,
        api_base_url=_OLLAMA_BASE,
        api_key="ollama",
        temperature=0.2,
        max_tokens=384,
        timeout_s=60.0,
        retries=0,
    )


def _summarise_obs(obs_dict: Dict[str, Any]) -> str:
    """One-line pre-flight summary of what the Fraudster is about to see."""
    queue = obs_dict.get("current_queue", []) or []
    verdicts = obs_dict.get("prior_verdicts", []) or []
    targets = obs_dict.get("investigation_targets_used", {}) or {}
    return (
        f"round={obs_dict.get('round_number')} "
        f"proposals_remaining={obs_dict.get('proposals_remaining')} "
        f"actions_left={obs_dict.get('actions_left_this_turn')} "
        f"queue_size={len(queue)} "
        f"prior_verdicts={len(verdicts)} "
        f"targets_pulled_on={len(targets)} ads"
    )


def _try_one_turn(
    fraudster: LLMFraudster,
    env: RefereeEnvironment,
    *,
    turn_label: str,
) -> Optional[Any]:
    """
    Run ONE Fraudster turn end-to-end and dump the full prompt+response
    chain. Returns the parsed FraudsterAction (or None on schema failure).
    """
    obs = env.build_fraudster_observation()
    obs_dict = obs.model_dump()

    print(f"\n--- {turn_label} | {_summarise_obs(obs_dict)} ---")

    user_prompt = fraudster._build_user_prompt(obs_dict)
    print("\n[USER PROMPT sent to LLM]")
    print(_wrap(user_prompt))

    messages = [
        {"role": "system", "content": fraudster.system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = fraudster._call_llm_once(messages)
    except Exception as exc:
        print(f"\n[ERROR] LLM call raised {type(exc).__name__}: {exc}")
        return None

    print("\n[RAW LLM RESPONSE]")
    print(_wrap(raw if raw.strip() else "(empty string)"))

    try:
        action = fraudster._parse_and_validate(raw)
    except Exception as exc:
        print(f"\n[SCHEMA FAIL] {type(exc).__name__}: {exc}")
        return None

    print("\n[PARSED ACTION (schema-valid)]")
    print(_wrap(json.dumps(action.model_dump(exclude_none=True), indent=2)))

    try:
        env.step_as_fraudster(action)
        print(
            f"\n[ENV ACK] phase={env.phase} "
            f"round={env.state.round_number} "
            f"proposals_used={env.state.proposals_used}"
        )
    except Exception as exc:
        print(f"\n[ENV REJECTED ACTION] {type(exc).__name__}: {exc}")

    return action


def _drain_investigator(
    env: RefereeEnvironment, *, max_actions: int = 3
) -> List[str]:
    """
    Hand control to a scripted Investigator for up to ``max_actions``
    actions so the Fraudster's NEXT turn observation will contain a real
    verdict / investigation target. Returns a compact log for printing.
    """
    inv = ScriptedInvestigator()
    log: List[str] = []
    actions_taken = 0
    while env.phase == "investigator_turn" and actions_taken < max_actions:
        obs = env.build_investigator_observation().model_dump()
        action = inv.act(obs)
        env.step_as_investigator(action)
        actions_taken += 1
        if action.action_type == "verdict":
            log.append(
                f"verdict({action.ad_id}={action.verdict}@{action.confidence})"
            )
        elif action.action_type == "investigate":
            log.append(f"investigate({action.ad_id}/{action.investigation_target})")
        else:
            log.append(action.action_type)
    return log


def verify_task(task_id: str, seed: int = 42) -> Dict[str, Any]:
    _hr(f"TASK = {task_id} (seed={seed}, model={_OLLAMA_MODEL})")

    env = RefereeEnvironment()
    env.reset_match(task_id=task_id, seed=seed)

    print(
        f"max_rounds={env.state.max_rounds} "
        f"max_proposals={env.state.max_proposals} "
        f"allowed_categories={env.build_fraudster_observation().allowed_categories}"
    )

    fraudster = _build_fraudster()

    schema_fails = 0
    actions_seen: List[str] = []

    for turn_idx in range(_TURNS_PER_TASK):
        if env.phase != "fraudster_turn":
            print(f"[skip] turn {turn_idx + 1}: phase is {env.phase}, stopping.")
            break

        action = _try_one_turn(
            fraudster, env, turn_label=f"FRAUDSTER TURN {turn_idx + 1}"
        )
        if action is None:
            schema_fails += 1
        else:
            actions_seen.append(action.action_type)

        if env.phase == "investigator_turn":
            inv_log = _drain_investigator(env, max_actions=3)
            print(f"\n[Investigator played] {' | '.join(inv_log) or '(no-op)'}")

        if env.phase in ("audit_phase", "done"):
            print(f"[Episode advanced to {env.phase}; stopping diagnostic.]")
            break

    return {
        "task_id": task_id,
        "schema_fails": schema_fails,
        "calls": fraudster.call_count,
        "fallback_count_via_act": fraudster.fallback_count,
        "actions_seen": actions_seen,
        "last_error": fraudster.last_error,
    }


def verify_reactive_path(task_id: str = "task_2", seed: int = 7) -> Dict[str, Any]:
    """
    Force the rejection-feedback path: Fraudster proposes → Investigator
    rejects that ad → Fraudster turn 2 sees ``prior_verdicts`` containing a
    rejection of its own proposal. We then check whether the LLM picks a
    different category / softer copy on the follow-up.

    This is the single most important behavioural check before we trust
    the Fraudster as a frozen training opponent — without a reactive
    Fraudster the trained Investigator's reward saturates trivially.
    """
    _hr(f"REACTIVE PROBE — task={task_id} seed={seed}")
    env = RefereeEnvironment()
    env.reset_match(
        task_id=task_id,
        seed=seed,
        max_investigator_actions_per_turn=2,
        max_fraudster_actions_per_turn=2,
    )
    fraudster = _build_fraudster()

    print("\n[Stage 1] One LLM-driven proposal, then forced end_turn.")
    obs1 = env.build_fraudster_observation().model_dump()
    user_prompt_1 = fraudster._build_user_prompt(obs1)
    messages_1 = [
        {"role": "system", "content": fraudster.system_prompt},
        {"role": "user", "content": user_prompt_1},
    ]
    raw_1 = fraudster._call_llm_once(messages_1)
    try:
        action_1 = fraudster._parse_and_validate(raw_1)
    except Exception as exc:
        print(f"[SCHEMA FAIL on Stage 1] {exc}")
        return {"task_id": task_id, "stage_1_schema_fail": True}

    if action_1.action_type != "propose_ad":
        print(
            f"[skip-reactive] Fraudster opened with {action_1.action_type!r}, "
            "not propose_ad; reactive probe needs at least one proposal."
        )
        return {"task_id": task_id, "skipped": True, "opening": action_1.action_type}

    env.step_as_fraudster(action_1)
    proposed_ad = env._proposal_slot_to_ad_id.get(0)
    print(
        f"  - Fraudster proposed: category={action_1.category!r} "
        f"-> ad_id={proposed_ad}"
    )

    if env.phase == "fraudster_turn":
        env.step_as_fraudster(FraudsterAction(action_type="end_turn"))
    print(f"  - phase after forced end_turn: {env.phase}")

    print("\n[Stage 2] Force-reject the Fraudster's ad to populate prior_verdicts.")
    if proposed_ad and env.phase == "investigator_turn":
        try:
            env.step_as_investigator(
                AdReviewAction(
                    action_type="verdict",
                    ad_id=proposed_ad,
                    verdict="reject",
                    confidence=0.85,
                    rationale=(
                        "Forced rejection for reactive probe: "
                        "soft urgency markers and unverifiable claims."
                    ),
                )
            )
            print(f"  - Investigator rejected {proposed_ad}.")
        except Exception as exc:
            print(f"  - Investigator probe step failed: {exc}")

    safety = 0
    while env.phase == "investigator_turn" and safety < 6:
        drained = _drain_investigator(env, max_actions=2)
        if drained:
            print(f"  - Investigator drain: {' | '.join(drained)}")
        safety += 1

    if env.phase != "fraudster_turn":
        print(f"  - Could not return to fraudster_turn (phase={env.phase}); abort.")
        return {"task_id": task_id, "phase_after_drain": env.phase}

    print("\n[Stage 3] Fraudster's reactive turn (sees its own ad rejected).")
    obs2 = env.build_fraudster_observation().model_dump()
    print(
        f"  - prior_verdicts now contains {len(obs2.get('prior_verdicts', []))} "
        f"item(s); first = {obs2.get('prior_verdicts', [None])[:1]}"
    )
    user_prompt_2 = fraudster._build_user_prompt(obs2)
    print("\n[USER PROMPT — turn 2]")
    print(_wrap(user_prompt_2))

    messages_2 = [
        {"role": "system", "content": fraudster.system_prompt},
        {"role": "user", "content": user_prompt_2},
    ]
    raw_2 = fraudster._call_llm_once(messages_2)
    print("\n[RAW LLM RESPONSE — turn 2]")
    print(_wrap(raw_2 if raw_2.strip() else "(empty)"))

    try:
        action_2 = fraudster._parse_and_validate(raw_2)
    except Exception as exc:
        print(f"\n[SCHEMA FAIL on Stage 3] {exc}")
        return {"task_id": task_id, "stage_3_schema_fail": True}

    print("\n[PARSED ACTION — turn 2]")
    print(_wrap(json.dumps(action_2.model_dump(exclude_none=True), indent=2)))

    adapted = (
        action_2.action_type == "modify_pending_ad"
        or (
            action_2.category is not None
            and action_2.category != action_1.category
        )
    )
    print(
        f"\n  >>> Adapted to rejection? {adapted} "
        f"(turn1 cat={action_1.category!r}, turn2 cat={action_2.category!r}, "
        f"action={action_2.action_type})"
    )

    return {
        "task_id": task_id,
        "turn1_category": action_1.category,
        "turn2_category": action_2.category,
        "turn2_action_type": action_2.action_type,
        "adapted": adapted,
        "schema_fails": 0,
    }


def main() -> int:
    _hr("CounterFeint Fraudster live diagnostic (Ollama)")
    print(f"endpoint  = {_OLLAMA_BASE}")
    print(f"model     = {_OLLAMA_MODEL}")
    print(f"system_prompt_chars = {len(FRAUDSTER_SYSTEM_PROMPT)}")
    print(f"turns_per_task = {_TURNS_PER_TASK}")

    summaries: List[Dict[str, Any]] = []
    for task_id in ("task_1", "task_2", "task_3"):
        try:
            summaries.append(verify_task(task_id))
        except Exception as exc:
            print(f"\n[FATAL] {task_id}: {type(exc).__name__}: {exc}")
            summaries.append(
                {"task_id": task_id, "schema_fails": -1, "error": str(exc)}
            )

    reactive_summary: Dict[str, Any]
    try:
        reactive_summary = verify_reactive_path()
    except Exception as exc:
        print(f"\n[FATAL] reactive probe: {type(exc).__name__}: {exc}")
        reactive_summary = {"reactive": False, "error": str(exc)}

    _hr("SUMMARY")
    for s in summaries:
        print(json.dumps(s, indent=2))
    print("\n[reactive probe]")
    print(json.dumps(reactive_summary, indent=2))

    total_fails = sum(max(0, s.get("schema_fails", 0)) for s in summaries)
    if total_fails:
        print(f"\nResult: {total_fails} schema failure(s) across all tasks.")
        return 1
    print("\nResult: all turns produced schema-valid Fraudster actions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
