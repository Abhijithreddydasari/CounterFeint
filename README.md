---
title: CounterFeint
emoji: "\U0001F575\uFE0F"
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - ad-fraud
  - reinforcement-learning
---

# CounterFeint — Ad Fraud Investigation Environment

An OpenEnv environment that simulates ad fraud review - a real-world task where AI agents investigate queues of advertisements, uncover fraud signals, and render verdicts under budget constraints.

Ad fraud costs the digital advertising industry over **$100 billion annually**. Platforms like Meta process billions of ads daily and ban advertisers only at high confidence thresholds. Unlike simple classification, real ad review is a **sequential decision-making** problem: a reviewer starts with limited surface-level signals, actively chooses what to investigate within a constrained budget, and must decide when enough evidence exists to commit to a verdict. This environment captures that workflow and provides a training ground for agents to learn it.

## Quick Start

### Install

```bash
pip install -e .
```

### Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Use the client

```python
from counterfeint import AdFraudEnv, AdReviewAction

with AdFraudEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=42, task_id="task_1")
    print(result.observation.queue_summary)

    # Investigate an ad
    result = env.step(AdReviewAction(
        action_type="investigate",
        ad_id="ad_001",
        investigation_target="landing_page",
    ))
    print(result.observation.feedback)

    # Render a verdict
    result = env.step(AdReviewAction(
        action_type="verdict",
        ad_id="ad_001",
        verdict="reject",
        confidence=0.9,
    ))
    print(f"Reward: {result.reward}, Done: {result.done}")
```

### Run with Docker

```bash
docker build -t counterfeint .
docker run -p 8000:8000 counterfeint
```

## Environment Design

### Episode flow

Each episode is a review session. The agent receives a queue of ads and must process them within a limited action budget:

```
reset(task_id, seed)
  |
  v
+----------------------------------+<----------------------+
|  Observe queue + first ad info   |                       |
+------------------+---------------+                       |
                   |                                       |
                   v                                       |
        +-------------+     +------------------+           |
        | investigate |---->| Reveal one signal |----------+
        +-------------+     | (costs 1 budget)  |
               |            +------------------+
               v
        +-------------+     +------------------+
        |   verdict   |---->| approve / reject  |----------+
        +-------------+     |  / escalate       |          |
               |            +------------------+           |
               v                                           |
        +--------------+    +------------------+           |
        | link_accounts|---->| Flag fraud ring   |----------+
        +--------------+    | (Task 3 only)     |
               |            +------------------+
               v
        Budget exhausted or all ads reviewed -> episode ends
```

### Tasks

Three tasks with increasing difficulty test different capabilities:

| Task | Name | Ads | Budget | Composition | Challenge |
|---|---|---:|---:|---|---|
| 1 | Basic Ad Triage | 5 | 25 | 2 legit, 3 obvious fraud | Learn the investigate -> verdict loop |
| 2 | Sophisticated Fraud | 12 | 30 | 5 legit, 5 sophisticated scams, 2 gray-area | Triage under budget pressure (~2.5 actions/ad) |
| 3 | Fraud Network Detection | 20 | 35 | 6 legit, 10 fraud (3 hidden rings), 4 gray-area | Cross-ad reasoning to detect coordinated networks (~1.75 actions/ad) |

Task 3 introduces **fraud rings** - clusters of 3-5 ads controlled by the same actor, using varied topologies (cliques, chains, hub-and-spoke). Individual ring members look borderline; the fraud signal is only visible by cross-referencing investigation data across ads (shared payment IDs, matching template hashes, overlapping targeting fingerprints).

### Action Space

Actions are JSON objects. Three types:

**`investigate`** - spend one budget point to reveal a signal about an ad.

```json
{
  "action_type": "investigate",
  "ad_id": "ad_001",
  "investigation_target": "landing_page"
}
```

Each ad has six investigation dimensions:

| Target | What it reveals |
|---|---|
| `advertiser_history` | Account age, spend history, violation record, verification status |
| `landing_page` | Domain age, SSL, registrar, redirect chains, scam template similarity |
| `payment_method` | Payment type, chargeback history, cross-account velocity |
| `targeting_overlap` | Targeting fingerprint, audience overlap percentages |
| `campaign_structure` | Objective, bid strategy, budget/age ratio, placement distribution |
| `policy_classifier` | Llama Guard 3 / Purple Llama mock: safe / unsafe verdict, triggered LG categories (S1–S13), TS-Fraud markers (urgency, fake authority, get-rich-quick, etc.) |

**`verdict`** - render a final decision on an ad.

```json
{
  "action_type": "verdict",
  "ad_id": "ad_001",
  "verdict": "reject",
  "confidence": 0.9
}
```

`verdict` options: `approve`, `reject`, `escalate`. `confidence`: 0.0-1.0.

**`link_accounts`** - flag two ads as part of the same fraud network (Task 3).

```json
{
  "action_type": "link_accounts",
  "ad_id": "ad_003",
  "linked_ad_id": "ad_007",
  "link_reason": "shared payment ID pmt_ring_48231 and matching template hash"
}
```

### Observation Space

Observations are text-heavy by design so LLM agents can reason naturally:

| Field | Type | Description |
|---|---|---|
| `queue_summary` | `str` | Task name, total/reviewed/pending counts, budget remaining |
| `current_ad_info` | `str` | Ad copy, category, targeting, risk signals for the focused ad |
| `investigation_findings` | `str` | Accumulated findings from all investigations so far |
| `verdict_history_summary` | `str` | Verdicts rendered so far |
| `feedback` | `str` | Natural language feedback on the last action |
| `available_ads` | `list[str]` | Ad IDs still pending review |
| `queue_status` | `dict` | Structured status for programmatic access |
| `done` | `bool` | Whether the episode is complete |
| `reward` | `float` | Step reward |

## Reward Design

| Action | Reward | Rationale |
|---|---:|---|
| Investigation | -0.02 | Simulates time/latency cost |
| Correct rejection (fraud -> reject) | +0.30 to +0.40 | Scaled by fraud severity |
| Correct approval (legit -> approve) | +0.10 | Revenue preserved |
| Correct escalation | +0.15 | Appropriate caution |
| False positive (legit -> reject) | -0.35 | Lost advertiser revenue |
| False negative (fraud -> approve) | -0.50 | Worst outcome - fraud goes live |
| Escalate (when wrong) | -0.05 | Human reviewer cost |
| Correct network link | +0.40 | High-value coordinated fraud detection |
| Incorrect network link | -0.25 | False accusation cost |

Unreviewed ads are auto-approved at episode end - missed fraud incurs the full -0.50 false-negative penalty.

## Grading & Scoring

Each task has a dedicated grader that produces a normalized **0.0-1.0 score**. Raw reward is normalized between theoretical worst-case (every decision wrong + full budget wasted) and best-case (every decision correct + efficient budget use).

| Component | Task 1 | Task 2 | Task 3 |
|---|:---:|:---:|:---:|
| Verdict accuracy | Yes | Yes | Yes |
| Budget efficiency bonus | Yes | Yes | Yes |
| Calibration bonus | - | Yes | Yes |
| Network detection (edge coverage) | - | - | Yes |
| Investigation coverage bonus | - | - | Yes |

**Calibration bonus** rewards agents whose stated confidence correlates with actual accuracy - high confidence on correct verdicts and low confidence on uncertain ones.

**Network detection** uses edge coverage: what fraction of ground-truth fraud ring connections did the agent discover via `link_accounts`?

**Coverage bonus** rewards breadth over depth - agents that review more ads (rather than deep-diving a single one) score higher on Task 3.

## Baseline Scores

Generated with `seed=42` using `meta-llama/Llama-3.1-8B-Instruct`. Reproducible via `python inference.py`.

| Task | Score | Steps | Verdicts |
|---|---:|---:|---:|
| Task 1 (Easy) | 0.953 | 10 | 5/5 |
| Task 2 (Medium) | 0.882 | 23 | 12/12 |
| Task 3 (Hard) | 0.415 | 35 | 20/20 |

The sharp drop on Task 3 reflects the difficulty of cross-ad reasoning under tight budget - the baseline agent investigates and renders verdicts well but struggles to detect coordinated fraud rings.

## Three-Agent FraudArena (R2)

CounterFeint also ships a three-agent multi-policy mode where an
adversarial **Fraudster** proposes ads, an **Investigator** reviews them
under a budget, and an **Auditor** grades the Investigator's reasoning
post-hoc. This adds Themes #1 (multi-agent interaction) and #4
(self-improvement) to the original single-agent task.

```
python -m counterfeint.inference                       # scripted vs scripted vs scripted
python -m counterfeint.inference --llm-fraudster       # adversarial LLM Fraudster (frozen)
python -m counterfeint.inference --llm-fraudster --llm-investigator  # full LLM bench
```

The Fraudster has its own escalating curriculum (`max_proposals`,
`allowed_fraud_categories`) that scales with task difficulty, and
`--llm-fraudster` swaps the deterministic `ReactiveFraudster` for an
OpenAI-compatible LLM (works with Ollama at
`http://localhost:11434/v1`). LLM failures (timeout, bad JSON,
validation error) silently fall back to the scripted policy and are
counted in the `[END] ... fallbacks=fraudster:N,investigator:N`
STDOUT line.

### Self-play curriculum

During training the Investigator faces a fixed 70/30 split of opponents:

| Rollout share | Fraudster                                       | Why |
|--------------:|-------------------------------------------------|-----|
|         70 % | `ReactiveFraudster` (deterministic, programmatic) | Stable gradient signal; cheap rollouts |
|         30 % | `LLMFraudster` (frozen Llama-3.1-8B via Ollama)  | Open-ended adversarial pressure |

This avoids the "two moving targets" problem (we never co-train two
LLMs simultaneously) while still giving the trained Investigator
exposure to genuinely open-vocabulary fraud copy. See
[`counterfeint/training/rollout_config.py`](training/rollout_config.py)
for the canonical split spec.

## Before vs After Training

The held-out eval lane lives in
[`counterfeint/eval_suite.py`](eval_suite.py).  It runs the
Investigator over 30 fixed `(task_id, seed)` tuples (10 per task,
**disjoint from training seeds**) against the stable
`ReactiveFraudster` adversary, and writes three artefacts to
`eval_outputs/`:

```
eval_outputs/
├── eval_results.json   # per-episode metrics for both tags
├── eval_summary.md     # markdown delta table (before / after / delta)
└── eval_plot.png       # 2×2 bar chart, headline visual
```

Reproduce with:

```bash
# Pre-training baseline (scripted Investigator)
python -m counterfeint.eval_suite --before-tag scripted --after-tag scripted_rerun

# After training (programmatic, used at the end of the Colab notebook):
python -c "from counterfeint.eval_suite import run_before_after; \
  from counterfeint.scripted import ScriptedInvestigator; \
  from pathlib import Path; \
  run_before_after(before_tag='scripted', after_tag='trained_v1', \
    before_investigator_factory=ScriptedInvestigator, \
    after_investigator_factory=lambda: load_trained_investigator('checkpoints/v1'), \
    out_dir=Path('eval_outputs'))"
```

![Before vs After bar chart](eval_outputs/eval_plot.png)

*Bar chart compares grader score, Track A reasoning score, mean fraud
leaks (false approvals on ground-truth fraud), and budget consumption
across all three tasks.  The PNG above is a **placeholder** generated
from synthetic metrics so the README rendering stays self-contained;
the real artefact is regenerated end-to-end by every
`run_before_after` call (final cell of
[`training/train_investigator.ipynb`](training/train_investigator.ipynb))
and is committed to `eval_outputs/` pre-submission. The full
per-episode breakdown lives in
[`eval_outputs/eval_summary.md`](eval_outputs/eval_summary.md).*

Per-episode metrics tracked: `grader_score`, `track_a_score`,
`track_b_score`, `n_fraud_leaks`, `budget_used_pct`,
`fallback_count`, `rewards_by_role`. The `fallback_count` makes
silent LLM degradations visible; an eval run with > 30 % fallback
rate is treated as inconclusive.

### Evaluated against Meta-CIB-modeled ads

Beyond the procedurally generated tasks, an optional **realism sweep**
routes the same eval lane through a small held-out dataset
(`counterfeint/data/real_world_test_set.json`) of synthetic ads
authored to match the patterns described in Meta's published
quarterly **Adversarial Threat Reports** — e.g. the Ghana DigitSol
clique, the Benin Digited chain, and the China–Russia hub-spoke
operations now wired into our network generator
([`counterfeint/data/network_generator.py`](data/network_generator.py)).
Every ad in the holdout JSON carries:

* `case_study_source`     — the Meta CIB report it descends from
* `provenance_quarter`    — the report quarter (e.g. "Q3 2020")
* `ring_membership`       — the synthetic ring it belongs to

These ads never appear in training and are loaded via the
eval-only `counterfeint/data/real_world_loader.py` helper, so the
"trained on Meta CIB-modeled cases" claim is grounded and
reproducible.

## Project Structure

```
counterfeint/
+-- __init__.py              # Package exports
+-- client.py                # WebSocket client (extends EnvClient)
+-- models.py                # Action, Observation, State types
+-- inference.py             # R1 single-agent + R2 three-agent driver
+-- eval_suite.py            # Held-out eval lane: run_before_after()
+-- openenv.yaml             # OpenEnv manifest
+-- pyproject.toml           # Dependencies and package config
+-- requirements-dev.txt     # Dev-only deps (pytest, matplotlib for eval_plot.png)
+-- Dockerfile               # Multi-stage Docker build
+-- baseline_scores.json     # Cached baseline results
+-- agents/                  # LLM-backed policy classes (R2)
|   +-- base.py              # LLMPolicyBase: retry + timeout + scripted fallback
|   +-- prompts.py           # FRAUDSTER_SYSTEM_PROMPT, INVESTIGATOR_SYSTEM_PROMPT
|   +-- llm_fraudster.py     # LLMFraudster (fallback: ReactiveFraudster)
|   +-- llm_investigator.py  # LLMInvestigator (fallback: ScriptedInvestigator)
+-- scripted/                # Deterministic baseline policies (R2)
|   +-- fraudster.py         # Scripted, Reactive, and Gibberish (control) Fraudsters
|   +-- investigator.py      # ScriptedInvestigator (cites Meta policy IDs)
|   +-- auditor.py           # HeuristicAuditor (rule-based, fixed reward channel)
+-- data/
|   +-- ad_generator.py      # Episode generation, task configs (with Fraudster knobs)
|   +-- advertiser_profiles.py  # Synthetic advertiser history
|   +-- fraud_patterns.py    # Fraud + legit ad templates (easy/medium/hard)
|   +-- landing_pages.py     # Simulated landing page investigation data
|   +-- network_generator.py # Named CIB ring topologies (Ghana / Benin / China-Russia)
|   +-- meta_policy_taxonomy.py  # Meta policy citation metadata layer
|   +-- audit_heuristics.py  # Regex helpers (incl. Meta citation IDs)
+-- graders/
|   +-- base_grader.py       # Shared normalization and reward logic
|   +-- task1_grader.py      # Verdict accuracy only
|   +-- task2_grader.py      # + calibration bonus
|   +-- task3_grader.py      # + network detection + coverage bonus
|   +-- auditor_track_a.py   # Investigator reasoning audit (rationales, citations)
|   +-- auditor_track_b.py   # Fraudster ad-plausibility audit
|   +-- multi_agent_rewards.py  # Combined reward computation
+-- server/
|   +-- app.py               # FastAPI app with /tasks, /baseline, /grader endpoints
|   +-- environment.py       # Core environment (reset/step/state)
|   +-- referee.py           # 3-agent state machine, applies TaskConfig curriculum
|   +-- investigate_ui.py    # HTML dashboard routes (/investigate, /web redirect)
|   +-- static/
|       +-- investigate_hq.html  # Interactive investigation dashboard
+-- training/
|   +-- rollout_config.py    # 70/30 (ReactiveFraudster, LLMFraudster) split spec
|   +-- train_investigator.ipynb  # Colab scaffold; calls run_before_after at the end
+-- tests/
    +-- test_data_generation.py  # Determinism, cross-ref, decoy, CIB topology
    +-- test_environment.py      # Step logic, state tracking, anti-exploit
    +-- test_graders.py          # Score ranges, calibration, network scoring
    +-- test_three_agent_episode.py  # Three-agent state machine + curriculum
    +-- test_llm_agents.py       # LLMPolicyBase fallback + retry semantics
    +-- test_eval_suite.py       # Eval-lane parser + writer round-trips
    +-- test_meta_policy_taxonomy.py  # Citation coverage + evidence-token recognition
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/schema` | GET | Action/Observation JSON schemas |
| `/ws` | WS | WebSocket for `step()` / `reset()` / `state()` |
| `/tasks` | GET | Task list with configs and action schema |
| `/baseline` | GET | Baseline scores (cached or live) |
| `/grader` | GET | Last episode's grader result |
| `/investigate` | GET | HTML investigation dashboard (also `/` redirects here) |

## License

BSD 3-Clause License
