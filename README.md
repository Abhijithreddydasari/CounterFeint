---
title: Ad Fraud Investigation Environment
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

# Ad Fraud Investigation Environment

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
from ad_fraud_env import AdFraudEnv, AdReviewAction

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
docker build -t ad-fraud-env .
docker run -p 8000:8000 ad-fraud-env
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
| `creative_similarity` | Template hash, image dimensions, scam template similarity score |
| `campaign_structure` | Objective, bid strategy, budget/age ratio, placement distribution |

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

## Project Structure

```
ad_fraud_env/
+-- __init__.py              # Package exports
+-- client.py                # WebSocket client (extends EnvClient)
+-- models.py                # Action, Observation, State types
+-- inference.py             # Baseline LLM agent with mandatory stdout logging
+-- openenv.yaml             # OpenEnv manifest
+-- pyproject.toml           # Dependencies and package config
+-- Dockerfile               # Multi-stage Docker build
+-- baseline_scores.json     # Cached baseline results
+-- data/
|   +-- ad_generator.py      # Episode generation, task configs, campaign profiles
|   +-- advertiser_profiles.py  # Synthetic advertiser history
|   +-- fraud_patterns.py    # Fraud + legit ad templates (easy/medium/hard)
|   +-- landing_pages.py     # Simulated landing page investigation data
|   +-- network_generator.py # Fraud ring topologies via networkx
+-- graders/
|   +-- base_grader.py       # Shared normalization and reward logic
|   +-- task1_grader.py      # Verdict accuracy only
|   +-- task2_grader.py      # + calibration bonus
|   +-- task3_grader.py      # + network detection + coverage bonus
+-- server/
|   +-- app.py               # FastAPI app with /tasks, /baseline, /grader endpoints
|   +-- environment.py       # Core environment (reset/step/state)
|   +-- requirements.txt     # Server dependencies
+-- tests/
    +-- test_data_generation.py  # Determinism, cross-ref checks, decoy validation
    +-- test_environment.py      # Step logic, state tracking, anti-exploit
    +-- test_graders.py          # Score ranges, calibration, network scoring
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
