"""
Live diagnostic for the local-transformers Investigator (``HFInvestigator``).

Symmetric counterpart to :mod:`.verify_fraudster`: where that script
proves the Fraudster receives the new ``my_proposal_signals`` and
produces schema-valid JSON via Ollama, this one proves the
trainable :class:`HFInvestigator` (Qwen3.5 / Qwen2.5 family in 4-bit)

  1. **receives** the new ``evidence_ledger`` block in its observation,
  2. **renders** it inside the user prompt,
  3. produces a **schema-valid** ``AdReviewAction`` (verdict /
     investigate / link_accounts) on each turn,
  4. and that ``last_prompt`` / ``last_completion`` are populated on the
     policy instance — that's what the GRPO rollout collector
     (:mod:`counterfeint.training.rollout`) records.

This is what we run *before* kicking off a real GRPO training run, so
that any "fallback to ScriptedInvestigator on every turn" failure
mode shows up here in 30s instead of 45 minutes into the rollout
collection cell of the notebook.

Run it
~~~~~~
::

    # Default model: Qwen/Qwen2.5-1.5B-Instruct (already cached locally;
    # the notebook trains on this exact model so verifying it here matches
    # the production code path 1:1)
    python -m counterfeint.diagnostics.verify_investigator

    # Try a Qwen-3.5 variant once it's downloaded
    python -m counterfeint.diagnostics.verify_investigator --model Qwen/Qwen3.5-0.8B

    # Quick smoke (one task, one investigator turn) for fast iteration
    python -m counterfeint.diagnostics.verify_investigator --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from counterfeint.agents import HFInvestigator
from counterfeint.agents.prompts import INVESTIGATOR_SYSTEM_PROMPT
from counterfeint.scripted import ReactiveFraudster
from counterfeint.server.referee import RefereeEnvironment


_DEFAULT_MODEL = os.getenv("INVESTIGATOR_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
_FRAUDSTER_TURNS_TO_SEED = 3  # how many Fraudster proposals before we hand control over
_INVESTIGATOR_TURNS_PER_TASK = 3


def _hr(title: str = "") -> None:
    bar = "=" * 78
    if title:
        print(f"\n{bar}\n{title}\n{bar}")
    else:
        print(bar)


def _wrap(text: str, indent: str = "  ") -> str:
    return textwrap.indent(text, indent)


def _seed_queue_via_scripted_fraudster(env: RefereeEnvironment, *, seed: int) -> None:
    """Run a few scripted-Fraudster turns so the Investigator has things to review."""
    fraudster = ReactiveFraudster(seed=seed)
    safety = 0
    while env.phase == "fraudster_turn" and safety < _FRAUDSTER_TURNS_TO_SEED * 2:
        obs = env.build_fraudster_observation().model_dump()
        action = fraudster.act(obs)
        env.step_as_fraudster(action)
        safety += 1
    # If the env hasn't auto-transitioned, force end_turn.
    if env.phase == "fraudster_turn":
        from counterfeint.models import FraudsterAction
        env.step_as_fraudster(FraudsterAction(action_type="end_turn"))


def _summarise_obs(obs_dict: Dict[str, Any]) -> str:
    queue_status = obs_dict.get("queue_status") or {}
    pending = obs_dict.get("available_ads") or []
    ledger = obs_dict.get("evidence_ledger") or {}
    return (
        f"task={queue_status.get('task_id')} "
        f"steps_remaining={queue_status.get('steps_remaining')} "
        f"investigation_budget={queue_status.get('investigation_budget')} "
        f"pending_ads={len(pending)} "
        f"ledger_rows={len(ledger)}"
    )


def _try_one_investigator_turn(
    investigator: HFInvestigator,
    env: RefereeEnvironment,
    *,
    turn_label: str,
) -> Optional[Any]:
    """Run ONE Investigator turn end-to-end and dump prompt+completion+action."""
    obs = env.build_investigator_observation()
    obs_dict = obs.model_dump()

    print(f"\n--- {turn_label} | {_summarise_obs(obs_dict)} ---")

    user_prompt = investigator._build_user_prompt(obs_dict)
    print("\n[USER PROMPT sent to LLM]")
    print(_wrap(user_prompt))

    if "Evidence ledger" not in user_prompt:
        print(
            "\n[WARN] 'Evidence ledger' marker NOT found in user prompt — "
            "this means the new evidence_ledger block did NOT render. Check "
            "INVESTIGATOR_USER_TEMPLATE in counterfeint/agents/prompts.py."
        )
    else:
        print("[OK] Evidence ledger rendered in user prompt.")

    t0 = time.perf_counter()
    try:
        action = investigator.act(obs_dict)
    except Exception as exc:
        print(f"\n[ERROR] act() raised {type(exc).__name__}: {exc}")
        return None
    dt_ms = (time.perf_counter() - t0) * 1000

    raw = investigator.last_completion
    print(f"\n[RAW LLM COMPLETION ({dt_ms:.0f} ms)]")
    if raw:
        print(_wrap(raw if raw.strip() else "(empty string)"))
    else:
        print(_wrap("(None — fallback fired; ScriptedInvestigator chose the action)"))

    print("\n[PARSED ACTION (schema-valid)]")
    print(_wrap(json.dumps(action.model_dump(exclude_none=True), indent=2)))

    try:
        env.step_as_investigator(action)
        print(
            f"\n[ENV ACK] phase={env.phase} "
            f"step_count={env.state.step_count}"
        )
    except Exception as exc:
        print(f"\n[ENV REJECTED ACTION] {type(exc).__name__}: {exc}")

    return action


def verify_task(
    investigator: HFInvestigator,
    *,
    task_id: str,
    seed: int = 42,
    n_turns: int = _INVESTIGATOR_TURNS_PER_TASK,
) -> Dict[str, Any]:
    _hr(f"TASK = {task_id} (seed={seed})")

    env = RefereeEnvironment()
    env.reset_match(task_id=task_id, seed=seed)

    print(
        f"max_rounds={env.state.max_rounds} "
        f"max_proposals={env.state.max_proposals}"
    )

    print(
        f"\n[seeding] Running scripted Fraudster for "
        f"{_FRAUDSTER_TURNS_TO_SEED} turns to build the queue ..."
    )
    _seed_queue_via_scripted_fraudster(env, seed=seed)
    print(f"[seeded]  phase={env.phase}, queue ready for the Investigator.")

    investigator.reset()
    actions_seen: List[str] = []
    schema_fails = 0
    fallbacks_before = investigator.fallback_count

    for turn_idx in range(n_turns):
        if env.phase != "investigator_turn":
            print(f"[skip] turn {turn_idx + 1}: phase is {env.phase}, stopping.")
            break

        action = _try_one_investigator_turn(
            investigator, env, turn_label=f"INVESTIGATOR TURN {turn_idx + 1}"
        )
        if action is None:
            schema_fails += 1
        else:
            actions_seen.append(action.action_type)

        if env.phase in ("audit_phase", "done"):
            print(f"[Episode advanced to {env.phase}; stopping diagnostic.]")
            break

    fallbacks_this_task = investigator.fallback_count - fallbacks_before

    return {
        "task_id": task_id,
        "schema_fails": schema_fails,
        "calls": investigator.call_count,
        "fallbacks_this_task": fallbacks_this_task,
        "actions_seen": actions_seen,
        "last_error": investigator.last_error,
    }


def _build_investigator(model_name: str, *, load_in_4bit: bool) -> HFInvestigator:
    """Load the HFInvestigator once, reused across all task tiers."""
    print(f"\n[load] Loading {model_name} (4bit={load_in_4bit}) ...")
    t0 = time.perf_counter()
    investigator = HFInvestigator.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        max_new_tokens=384,
        temperature=0.7,
        do_sample=True,
    )
    dt = time.perf_counter() - t0
    print(f"[load] OK in {dt:.1f}s.")

    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(0)
            print(
                f"[load] VRAM free={free_b/1e9:.2f} GB total={total_b/1e9:.2f} GB "
                f"allocated_by_us={torch.cuda.memory_allocated(0)/1e9:.2f} GB"
            )
    except Exception:
        pass

    return investigator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live diagnostic for the trainable HFInvestigator policy."
    )
    parser.add_argument(
        "--model", default=_DEFAULT_MODEL,
        help=f"HF model id or local path (default: {_DEFAULT_MODEL!r}; "
             "override via --model or INVESTIGATOR_MODEL env var)",
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Load in fp16/bf16 instead of 4-bit (uses more VRAM but avoids "
             "bitsandbytes if it's misconfigured on your machine).",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=("task_1", "task_2", "task_3"),
        choices=("task_1", "task_2", "task_3", "task_3_unseen"),
        help="Which task tiers to verify (default: all three).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Match seed for the scripted Fraudster + env setup.",
    )
    parser.add_argument(
        "--turns-per-task", type=int, default=_INVESTIGATOR_TURNS_PER_TASK,
        help=f"How many Investigator turns to run per task (default {_INVESTIGATOR_TURNS_PER_TASK}).",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Shortcut for --tasks task_1 --turns-per-task 1 (smoke run only).",
    )
    args = parser.parse_args()

    if args.quick:
        args.tasks = ("task_1",)
        args.turns_per_task = 1

    _hr("CounterFeint Investigator live diagnostic (HFInvestigator / local)")
    print(f"model              = {args.model}")
    print(f"load_in_4bit       = {not args.no_4bit}")
    print(f"system_prompt_chars= {len(INVESTIGATOR_SYSTEM_PROMPT)}")
    print(f"tasks              = {list(args.tasks)}")
    print(f"turns_per_task     = {args.turns_per_task}")

    try:
        investigator = _build_investigator(args.model, load_in_4bit=not args.no_4bit)
    except Exception as exc:
        print(f"\n[FATAL] failed to load model {args.model!r}: "
              f"{type(exc).__name__}: {exc}")
        return 2

    summaries: List[Dict[str, Any]] = []
    for task_id in args.tasks:
        try:
            summaries.append(verify_task(
                investigator,
                task_id=task_id,
                seed=args.seed,
                n_turns=args.turns_per_task,
            ))
        except Exception as exc:
            print(f"\n[FATAL] {task_id}: {type(exc).__name__}: {exc}")
            summaries.append(
                {"task_id": task_id, "schema_fails": -1, "error": str(exc)}
            )

    _hr("SUMMARY")
    for s in summaries:
        print(json.dumps(s, indent=2))

    total_fails = sum(max(0, s.get("schema_fails", 0)) for s in summaries)
    total_fb = sum(max(0, s.get("fallbacks_this_task", 0)) for s in summaries)
    if total_fails:
        print(f"\nResult: {total_fails} schema failure(s), {total_fb} fallback(s) total.")
        return 1
    if total_fb:
        print(
            f"\nResult: 0 schema failures, but {total_fb} fallback(s) total. "
            "The model produced JSON but it failed Pydantic validation — "
            "check ACTION_SCHEMA / _coerce_payload."
        )
        return 1
    print("\nResult: all turns produced schema-valid Investigator actions; no fallbacks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
