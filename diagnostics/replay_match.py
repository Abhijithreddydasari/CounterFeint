"""
Full multi-agent episode replay with both LLMs talking to each other.

Runs ONE complete CounterFeint episode (Fraudster ⇄ Investigator ⇄
Auditor) with the *exact* policies the training pipeline uses, then
prints every prompt, every raw completion, every parsed action and
every reward in chronological order. Optionally writes the same
transcript as a Markdown file under
``counterfeint/convo_logging/`` for easy sharing in slides / a PR.

Why this exists
---------------
Tests (``pytest``) prove the *individual* pieces work; the
``verify_*`` diagnostics prove each LLM produces schema-valid JSON in
isolation. This script is the single tool that proves both LLMs **talk
to each other coherently end-to-end** — the Fraudster proposes a
ring, the Investigator (a ~5× smaller different-family model) sees
the new ``evidence_ledger`` and (hopefully) starts linking accounts,
the Heuristic Auditor scores the trace.

Default setup matches the GRPO training cell in the notebook:

  * Fraudster:    Ollama ``llama3.1:8b`` (frozen)
  * Investigator: Local ``Qwen/Qwen2.5-1.5B-Instruct`` in 4-bit
                  (the trainable policy — same one the notebook trains)
  * Auditor:      Heuristic (in-process Python)

Run it
~~~~~~
::

    # Default — both LLMs, full task_2 episode
    python -m counterfeint.diagnostics.replay_match

    # Pick a different task / seed / Investigator model
    python -m counterfeint.diagnostics.replay_match --task task_3 --seed 7
    python -m counterfeint.diagnostics.replay_match --investigator-model Qwen/Qwen3.5-0.8B

    # Use the Ollama-backed HTTP Investigator (no transformers needed)
    python -m counterfeint.diagnostics.replay_match \\
        --investigator-backend ollama --investigator-model llama3.1:8b

    # Don't write a transcript file (stdout only)
    python -m counterfeint.diagnostics.replay_match --no-save
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

from counterfeint.scripted import (
    HeuristicAuditor,
    ReactiveFraudster,
    ScriptedInvestigator,
)
from counterfeint.server.referee import RefereeEnvironment


_DEFAULT_INVESTIGATOR_MODEL = os.getenv("INVESTIGATOR_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
_DEFAULT_FRAUDSTER_MODEL = os.getenv("MODEL_NAME", "llama3.1:8b")
_DEFAULT_OLLAMA_BASE = os.getenv("API_BASE_URL", "http://localhost:11434/v1")


# =============================================================================
# Output helpers — both stdout and Markdown transcript writers
# =============================================================================


class _TranscriptWriter:
    """Splits writes between stdout and an optional Markdown file."""

    def __init__(self, md_path: Optional[Path]) -> None:
        self._md_path = md_path
        self._md_lines: List[str] = []

    def stdout(self, text: str = "") -> None:
        print(text)

    def md(self, text: str = "") -> None:
        if self._md_path is not None:
            self._md_lines.append(text)

    def both(self, text: str = "") -> None:
        self.stdout(text)
        self.md(text)

    def hr_stdout(self, title: str = "") -> None:
        bar = "=" * 78
        if title:
            self.stdout(f"\n{bar}\n{title}\n{bar}")
        else:
            self.stdout(bar)

    def section_md(self, title: str, level: int = 2) -> None:
        self.md(f"\n{'#' * level} {title}\n")

    def code_block(
        self, content: str, *, lang: str = "", indent_stdout: str = "  "
    ) -> None:
        self.stdout(textwrap.indent(content, indent_stdout))
        self.md(f"```{lang}")
        self.md(content)
        self.md("```")

    def flush(self) -> Optional[Path]:
        if self._md_path is None:
            return None
        self._md_path.parent.mkdir(parents=True, exist_ok=True)
        self._md_path.write_text("\n".join(self._md_lines) + "\n", encoding="utf-8")
        return self._md_path


# =============================================================================
# Policy factories
# =============================================================================


def _build_fraudster(backend: str, model: str, base_url: str) -> Tuple[Any, str]:
    """Return (policy, label) for the chosen Fraudster backend."""
    if backend == "scripted":
        return ReactiveFraudster(seed=42), "ReactiveFraudster (scripted)"
    if backend == "ollama":
        from counterfeint.agents import LLMFraudster
        return (
            LLMFraudster(
                model_name=model,
                api_base_url=base_url,
                api_key="ollama",
                temperature=0.2,
                max_tokens=128,
                timeout_s=120.0,
                retries=1,
            ),
            f"LLMFraudster (Ollama / {model})",
        )
    raise ValueError(f"Unknown fraudster backend: {backend!r}")


def _build_investigator(
    backend: str,
    *,
    hf_model: str,
    load_in_4bit: bool,
    ollama_model: str,
    ollama_base: str,
) -> Tuple[Any, str]:
    """Return (policy, label) for the chosen Investigator backend."""
    if backend == "scripted":
        return ScriptedInvestigator(), "ScriptedInvestigator (scripted)"
    if backend == "hf":
        from counterfeint.agents import HFInvestigator
        print(f"[load] Loading HFInvestigator from {hf_model} (4bit={load_in_4bit}) ...")
        t0 = time.perf_counter()
        inv = HFInvestigator.from_pretrained(
            hf_model,
            load_in_4bit=load_in_4bit,
            max_new_tokens=128,
            temperature=0.3,
            do_sample=True,
        )
        print(f"[load] OK in {time.perf_counter() - t0:.1f}s.")
        return inv, f"HFInvestigator (local / {hf_model})"
    if backend == "ollama":
        from counterfeint.agents import LLMInvestigator
        return (
            LLMInvestigator(
                model_name=ollama_model,
                api_base_url=ollama_base,
                api_key="ollama",
                temperature=0.2,
                max_tokens=128,
                timeout_s=120.0,
                retries=1,
            ),
            f"LLMInvestigator (Ollama / {ollama_model})",
        )
    raise ValueError(f"Unknown investigator backend: {backend!r}")


# =============================================================================
# Per-turn dump
# =============================================================================


def _record_turn(
    out: _TranscriptWriter,
    *,
    step_idx: int,
    role: str,
    policy_label: str,
    obs_summary: str,
    user_prompt: Optional[str],
    raw_completion: Optional[str],
    parsed_action: Any,
    reward: float,
    current_phase: str,
    prev_phase: str,
    include_system: bool,
    system_prompt: Optional[str],
) -> None:
    """Print one turn to stdout AND append to the Markdown transcript."""
    role_upper = role.upper()
    title = f"Step {step_idx} — {role_upper}  ({policy_label})"
    out.hr_stdout(title)
    out.section_md(title, level=2)

    out.stdout(f"obs: {obs_summary}")
    out.md(f"**Observation summary:** {obs_summary}")

    if include_system and system_prompt:
        out.stdout("\n[SYSTEM PROMPT]")
        out.md("\n**System prompt:**")
        out.code_block(system_prompt, lang="text")

    if user_prompt is not None:
        out.stdout("\n[USER PROMPT]")
        out.md("\n**User prompt:**")
        out.code_block(user_prompt, lang="text")

    if raw_completion is not None:
        out.stdout("\n[RAW COMPLETION]")
        out.md("\n**Raw completion:**")
        out.code_block(raw_completion if raw_completion.strip() else "(empty)", lang="text")
    elif role in {"fraudster", "investigator"}:
        out.stdout("\n[RAW COMPLETION] (none — fallback fired; scripted action used)")
        out.md("\n**Raw completion:** _(none — fallback fired; scripted action used)_")

    parsed_dict = parsed_action.model_dump(exclude_none=True) if parsed_action else {}
    out.stdout("\n[PARSED ACTION]")
    out.md("\n**Parsed action:**")
    out.code_block(json.dumps(parsed_dict, indent=2), lang="json")

    # Use ``current_phase`` (post-step) so a step that stayed in the
    # same role for another action is labelled clearly rather than
    # looking like a phase transition. Add a ``(continued)`` marker
    # when the role keeps the turn — this is the multi-action-per-turn
    # case that previously read as a confusing "new_phase=fraudster_turn"
    # right after a fraudster step.
    phase_note = (
        f"current_phase={current_phase}"
        if current_phase != prev_phase
        else f"current_phase={current_phase} (continued — same role's turn)"
    )
    out.both(f"\nreward={reward:+.3f}   {phase_note}")


# =============================================================================
# Main episode loop
# =============================================================================


def _run_episode(
    *,
    task_id: str,
    seed: int,
    fraudster: Any,
    fraudster_label: str,
    investigator: Any,
    investigator_label: str,
    auditor: Any,
    auditor_label: str,
    out: _TranscriptWriter,
    max_steps: int,
    include_system: bool,
) -> Dict[str, Any]:
    env = RefereeEnvironment()
    env.reset_match(task_id=task_id, seed=seed)

    out.both(f"\n*Task:* `{task_id}` *Seed:* `{seed}`")
    out.both(f"*Fraudster:* `{fraudster_label}`")
    out.both(f"*Investigator:* `{investigator_label}`")
    out.both(f"*Auditor:* `{auditor_label}`")
    out.both(
        f"*Match knobs:* max_rounds={env.state.max_rounds}, "
        f"max_proposals={env.state.max_proposals}, "
        f"max_steps={max_steps}"
    )

    role_handlers: Dict[str, Tuple[Any, str, Callable[[], Any], Callable[[Any], Any], str]] = {
        "fraudster_turn": (
            fraudster, fraudster_label,
            env.build_fraudster_observation, env.step_as_fraudster,
            "fraudster",
        ),
        "investigator_turn": (
            investigator, investigator_label,
            env.build_investigator_observation, env.step_as_investigator,
            "investigator",
        ),
        "audit_phase": (
            auditor, auditor_label,
            env.build_auditor_observation, env.step_as_auditor,
            "auditor",
        ),
    }

    role_reward_acc: Dict[str, float] = {
        "fraudster": 0.0, "investigator": 0.0, "auditor": 0.0,
    }
    schema_fails: Dict[str, int] = {
        "fraudster": 0, "investigator": 0, "auditor": 0,
    }
    fallback_baseline: Dict[str, int] = {
        "fraudster": getattr(fraudster, "fallback_count", 0),
        "investigator": getattr(investigator, "fallback_count", 0),
        "auditor": 0,
    }

    step_idx = 0
    while env.phase in role_handlers:
        policy, label, build_obs_fn, step_fn, role_name = role_handlers[env.phase]
        phase_before = env.phase

        # Auditor steps are cheap deterministic flag actions — don't
        # count them against the gameplay max_steps cap, otherwise the
        # submit_audit_report (which triggers grader scoring) may fall
        # outside the budget and grader_score stays null.
        if role_name != "auditor" and step_idx >= max_steps:
            break

        obs = build_obs_fn()
        obs_dict = obs.model_dump()
        obs_summary = _summarise_obs(obs_dict, role=role_name)

        # Snapshot the LLM trace slots before/after .act() so we can capture
        # what the model actually saw and emitted on this turn.
        for slot in ("last_prompt", "last_completion", "last_error"):
            if hasattr(policy, slot):
                setattr(policy, slot, None)

        try:
            action = policy.act(obs_dict)
        except Exception as exc:
            out.both(
                f"\n[POLICY CRASH] {role_name} ({label}): "
                f"{type(exc).__name__}: {exc}"
            )
            schema_fails[role_name] += 1
            break

        try:
            new_obs = step_fn(action)
        except Exception as exc:
            out.both(
                f"\n[ENV REJECTED] {role_name} action: "
                f"{type(exc).__name__}: {exc}"
            )
            break

        new_obs_dict = new_obs.model_dump() if new_obs is not None else {}
        reward = float(new_obs_dict.get("reward", 0.0) or 0.0)
        role_reward_acc[role_name] += reward

        step_idx += 1
        _record_turn(
            out,
            step_idx=step_idx,
            role=role_name,
            policy_label=label,
            obs_summary=obs_summary,
            user_prompt=getattr(policy, "last_prompt", None),
            raw_completion=getattr(policy, "last_completion", None),
            parsed_action=action,
            reward=reward,
            current_phase=env.phase,
            prev_phase=phase_before,
            include_system=include_system,
            system_prompt=getattr(policy, "system_prompt", None),
        )

    if env.phase != "done":
        out.both(f"\n[STOP] Hit max_steps={max_steps}; episode not finished.")

    state = env.state
    final_summary = {
        "task_id": task_id,
        "seed": seed,
        "steps_played": step_idx,
        "stopped_phase": env.phase,
        "end_reason": getattr(state, "end_reason", None),
        "rewards_by_role": role_reward_acc,
        "fraudster_reward_total": getattr(state, "fraudster_reward", 0.0),
        "investigator_reward_total": getattr(state, "investigator_reward", 0.0),
        "auditor_reward_total": getattr(state, "auditor_reward", 0.0),
        "grader_score": getattr(state, "grader_score", None),
        "fallback_count_delta": {
            "fraudster": (
                getattr(fraudster, "fallback_count", 0)
                - fallback_baseline["fraudster"]
            ),
            "investigator": (
                getattr(investigator, "fallback_count", 0)
                - fallback_baseline["investigator"]
            ),
        },
    }

    out.hr_stdout("EPISODE SUMMARY")
    out.section_md("Episode summary", level=2)
    out.code_block(json.dumps(final_summary, indent=2), lang="json")

    return final_summary


def _summarise_obs(obs_dict: Dict[str, Any], *, role: str) -> str:
    """One-line preview to print before the full prompt block."""
    if role == "fraudster":
        queue = obs_dict.get("current_queue") or []
        verdicts = obs_dict.get("prior_verdicts") or []
        my_signals = obs_dict.get("my_proposal_signals") or []
        return (
            f"round={obs_dict.get('round_number')} "
            f"actions_left={obs_dict.get('actions_left_this_turn')} "
            f"queue={len(queue)} prior_verdicts={len(verdicts)} "
            f"my_proposals={len(my_signals)}"
        )
    if role == "investigator":
        queue_status = obs_dict.get("queue_status") or {}
        pending = obs_dict.get("available_ads") or []
        ledger = obs_dict.get("evidence_ledger") or {}
        return (
            f"task={queue_status.get('task_id')} "
            f"steps_remaining={queue_status.get('steps_remaining')} "
            f"pending_ads={len(pending)} ledger_rows={len(ledger)}"
        )
    if role == "auditor":
        flags = obs_dict.get("pending_flags") or []
        return (
            f"phase={obs_dict.get('phase')} "
            f"investigator_actions={len(obs_dict.get('investigator_actions') or [])} "
            f"fraudster_proposals={len(obs_dict.get('fraudster_proposals') or [])} "
            f"existing_flags={len(flags)}"
        )
    return f"role={role}"


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one CounterFeint multi-agent episode end-to-end and "
            "dump the full LLM conversation."
        )
    )
    parser.add_argument(
        "--task", default="task_2",
        choices=("task_1", "task_2", "task_3", "task_3_unseen"),
        help="Task tier (default task_2 — has fraud rings + reactive verdicts).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--investigator-backend", default="ollama",
        choices=("hf", "ollama", "scripted"),
        help=(
            "ollama = HTTP via Ollama (default), "
            "hf = local transformers (Qwen2.5-1.5B-Instruct, requires torch), "
            "scripted = no LLM."
        ),
    )
    parser.add_argument(
        "--investigator-model", default=_DEFAULT_INVESTIGATOR_MODEL,
        help=(
            f"HF model id for backend=hf (default {_DEFAULT_INVESTIGATOR_MODEL!r}), "
            "or Ollama model name for backend=ollama."
        ),
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Load HFInvestigator in fp16/bf16 instead of 4-bit.",
    )
    parser.add_argument(
        "--fraudster-backend", default="ollama",
        choices=("ollama", "scripted"),
        help="ollama (default — needs `ollama serve`) or scripted.",
    )
    parser.add_argument(
        "--fraudster-model", default=_DEFAULT_FRAUDSTER_MODEL,
        help=f"Ollama model name for the Fraudster (default {_DEFAULT_FRAUDSTER_MODEL!r}).",
    )
    parser.add_argument(
        "--ollama-base", default=_DEFAULT_OLLAMA_BASE,
        help=f"Ollama OpenAI-compat base URL (default {_DEFAULT_OLLAMA_BASE!r}).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=50,
        help="Hard cap on steps to play (safety net; episodes usually end earlier).",
    )
    parser.add_argument(
        "--include-system-prompt", action="store_true",
        help="Also print the LLMs' (long) system prompt with every turn.",
    )

    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save-transcript", default=None,
        help=(
            "Where to write the Markdown transcript. "
            "Default: counterfeint/convo_logging/replay_<task>_seed<seed>_<utc>.md"
        ),
    )
    save_group.add_argument(
        "--no-save", action="store_true",
        help="Don't write a Markdown transcript (stdout only).",
    )
    args = parser.parse_args()

    # ----- Resolve transcript path -----
    md_path: Optional[Path] = None
    if not args.no_save:
        if args.save_transcript:
            md_path = Path(args.save_transcript).resolve()
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            convo_dir = Path(__file__).resolve().parent.parent / "convo_logging"
            md_path = convo_dir / f"replay_{args.task}_seed{args.seed}_{stamp}.md"

    out = _TranscriptWriter(md_path)

    out.hr_stdout("CounterFeint multi-agent episode replay")
    out.section_md(f"CounterFeint replay — `{args.task}` seed `{args.seed}`", level=1)
    out.md(f"_Generated: {datetime.now(timezone.utc).isoformat()}_  ")

    # ----- Build policies -----
    print(f"\n[init] Building Fraudster (backend={args.fraudster_backend}) ...")
    fraudster, fraudster_label = _build_fraudster(
        args.fraudster_backend, args.fraudster_model, args.ollama_base,
    )

    print(f"[init] Building Investigator (backend={args.investigator_backend}) ...")
    try:
        investigator, investigator_label = _build_investigator(
            args.investigator_backend,
            hf_model=args.investigator_model,
            load_in_4bit=not args.no_4bit,
            ollama_model=args.investigator_model,
            ollama_base=args.ollama_base,
        )
    except Exception as exc:
        print(
            f"[FATAL] could not build Investigator backend={args.investigator_backend} "
            f"({type(exc).__name__}: {exc})"
        )
        return 2

    auditor = HeuristicAuditor()
    auditor_label = "HeuristicAuditor (in-process Python)"

    # ----- Run the episode -----
    try:
        summary = _run_episode(
            task_id=args.task,
            seed=args.seed,
            fraudster=fraudster, fraudster_label=fraudster_label,
            investigator=investigator, investigator_label=investigator_label,
            auditor=auditor, auditor_label=auditor_label,
            out=out,
            max_steps=args.max_steps,
            include_system=args.include_system_prompt,
        )
    except Exception as exc:
        print(f"\n[FATAL] episode crashed: {type(exc).__name__}: {exc}")
        out.flush()
        raise

    # ----- Persist transcript -----
    written = out.flush()
    if written is not None:
        print(f"\n[transcript] Wrote {written} ({written.stat().st_size} bytes).")
    else:
        print("\n[transcript] --no-save was set; skipped writing.")

    # Exit non-zero only on hard crashes — schema fails are already printed.
    return 0 if summary.get("steps_played", 0) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
