# Ad Fraud Investigation Environment

An OpenEnv environment that simulates ad fraud review — a real-world task where AI agents investigate queues of advertisements, uncover fraud signals, and render verdicts under budget constraints.

## Motivation

Ad fraud costs the digital advertising industry billions annually. Meta alone processes 15 billion higher-risk ads daily and bans advertisers only at 95% confidence. This environment models the investigative workflow of an ad integrity reviewer: triaging a queue, deciding what to investigate, and making approve/reject/escalate decisions under time pressure.

Unlike simple classification tasks, this environment requires **sequential decision-making** — the agent starts with limited information and must actively choose which signals to investigate before committing to a verdict.

## Action Space

Actions are structured JSON objects with three types:

### `investigate`
Spend one budget point to reveal information about an ad.

| Field | Type | Description |
|---|---|---|
| `action_type` | `"investigate"` | |
| `ad_id` | `str` | Target ad (e.g., `"ad_001"`) |
| `investigation_target` | `str` | One of: `advertiser_history`, `landing_page`, `payment_method`, `targeting_overlap`, `creative_similarity`, `campaign_structure` |

### `verdict`
Render a final decision on an ad.

| Field | Type | Description |
|---|---|---|
| `action_type` | `"verdict"` | |
| `ad_id` | `str` | Target ad |
| `verdict` | `str` | `"approve"`, `"reject"`, or `"escalate"` |
| `confidence` | `float` | 0.0-1.0, agent's confidence in the decision |

### `link_accounts`
Flag two ads as part of the same fraud network (Task 3).

| Field | Type | Description |
|---|---|---|
| `action_type` | `"link_accounts"` | |
| `ad_id` | `str` | First ad in the suspected link |
| `linked_ad_id` | `str` | Second ad |
| `link_reason` | `str` | Why the agent believes these are connected |

## Observation Space

Observations are text-heavy to support LLM reasoning:

| Field | Type | Description |
|---|---|---|
| `queue_summary` | `str` | Task name, total/reviewed/pending counts, budget remaining |
| `current_ad_info` | `str` | Ad copy, category, targeting, risk signals for the focused ad |
| `investigation_findings` | `str` | Accumulated findings from all investigations |
| `verdict_history_summary` | `str` | Verdicts rendered so far |
| `feedback` | `str` | Natural language feedback on the last action |
| `available_ads` | `list[str]` | Ad IDs still pending review |
| `queue_status` | `dict` | Structured status for programmatic access |
| `done` | `bool` | Whether the episode is complete |
| `reward` | `float` | Step reward |

## Tasks

### Task 1: Basic Ad Triage (Easy)
- **Queue:** 5 ads (2 legit, 3 obviously fraudulent)
- **Budget:** 25 actions (5 per ad)
- **Challenge:** Learn the investigate-then-verdict loop
- **Expected score:** 0.6 - 0.8

### Task 2: Sophisticated Fraud Under Budget Pressure (Medium)
- **Queue:** 12 ads (5 legit, 5 sophisticated scams, 2 gray-area)
- **Budget:** 30 actions (~2.5 per ad)
- **Challenge:** Triage under constraints — cannot investigate everything
- **Expected score:** 0.3 - 0.5

### Task 3: Coordinated Fraud Network Detection (Hard)
- **Queue:** 20 ads including 3 hidden fraud rings (clusters of 3-5 ads)
- **Budget:** 40 actions (2 per ad)
- **Challenge:** Cross-ad reasoning to detect coordinated networks
- **Expected score:** 0.1 - 0.3

## Reward Design

| Action | Reward | Rationale |
|---|---:|---|
| Investigation | -0.02 | Simulates latency cost |
| Correct rejection (TP) | +0.30 to +0.40 | Scaled by fraud severity |
| Correct approval (TN) | +0.10 | Revenue preserved |
| False positive (reject legit) | -0.35 | Lost advertiser revenue |
| False negative (approve fraud) | -0.50 | Worst outcome |
| Escalate | -0.05 | Human reviewer cost |
| Correct network link | +0.40 | High-value detection |
| Incorrect network link | -0.20 | False accusation cost |

Episode-end bonuses: budget efficiency, calibration accuracy, full network detection.

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install
```bash
# Clone and install
cd ad_fraud_env
uv sync

# Or with pip
pip install -e .
```

### Run locally
```bash
# Start the server
uvicorn ad_fraud_env.server.app:app --host 0.0.0.0 --port 8000

# Or with uv
uv run server
```

### Run with Docker
```bash
docker build -f server/Dockerfile -t ad-fraud-env .
docker run -p 8000:8000 ad-fraud-env
```

### Run baseline

The inference script requires three environment variables:

| Variable | Description |
|---|---|
| `API_BASE_URL` | The API endpoint for the LLM |
| `MODEL_NAME` | The model identifier to use |
| `HF_TOKEN` | Your Hugging Face / API key |

```bash
API_BASE_URL=https://api-inference.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=hf_... \
python inference.py
```

**Infra constraints:** The inference script runs within 20 minutes on vcpu=2, memory=8GB.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/schema` | GET | Action/Observation/State JSON schemas |
| `/ws` | WS | WebSocket endpoint for `step()`/`reset()`/`state()` |
| `/tasks` | GET | List of tasks with action schema |
| `/baseline` | GET | Baseline scores (cached or live) |
| `/grader` | GET | Last episode's grader score |
| `/web` | GET | Auto-generated Gradio UI |

## Baseline Scores

Scores are generated with seed=42 using the model specified by `MODEL_NAME`. Set `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`, then run `python inference.py` to reproduce.

| Task | Score | Steps | Verdicts |
|---|---:|---:|---:|
| Task 1 (Easy) | 0.936 | 10 | 5/5 |
| Task 2 (Medium) | 0.868 | 30 | 12/12 |
| Task 3 (Hard) | 0.804 | 40 | 20/20 |

## License

BSD 3-Clause License
