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
  - multi-agent
  - grpo
  - trust-and-safety
---

# CounterFeint - Ad Fraud Investigation Environment

> A three-agent adversarial RL environment where a 0.6B Investigator learns to catch ad fraud that a frozen 8B Fraudster tries to sneak past it - trained end-to-end with GRPO on a single T4 GPU.

**[Deployed Env (HF Space)](https://huggingface.co/spaces/QuantumTransformer/CounterFeint)** ·
**[Blog Post](blog.md)** ·
**[Training Notebook](training/official_hf_training.ipynb)** ·
**[GitHub](https://github.com/Abhijithreddydasari/CounterFeint)**

---

## Why Ad Fraud?

Ad fraud costs the digital advertising industry over **$100 billion annually**.

![Ad fraud statistics](assets/ad%20fraud%20asset.png)

Platforms like Meta process billions of ads daily and reject advertisers only at high confidence - because a false positive kills a legitimate business's revenue, while a false negative puts a scam in front of millions of real people.

The critical insight: catching sophisticated ad fraud isn't a classification problem. It's **investigation**. A reviewer starts with surface-level signals, actively chooses what to dig into under time and budget constraints, and decides when the evidence is sufficient to commit to a verdict. That sequential decision-making workflow - triage, investigate, decide - is exactly what CounterFeint simulates.

I built this environment because this kind of investigative reasoning under uncertainty is precisely where RL-trained LLMs can add real value, and it maps directly to how trust & safety teams at platforms like Meta actually operate.

## The FraudArena - Three-Agent Setup

CounterFeint is an [OpenEnv](https://github.com/open-env/open-env)-compatible environment built around a three-agent adversarial game:

| Role | What it does | Model |
|---|---|---|
| **Fraudster** | Proposes ads designed to evade detection - from obvious scams to sophisticated, borderline-plausible creatives | `llama3.1:8b` (frozen) or scripted baseline |
| **Investigator** | Reviews a queue of ads, investigates suspicious signals within a budget, renders verdicts | `Qwen3-0.6B` + QLoRA (**trained via GRPO**) |
| **Auditor** | Grades the Investigator's reasoning quality and the Fraudster's plausibility | Rule-based scorecards (deterministic) |

The Investigator is the agent I train - a **0.6B parameter model** learning to outperform a frozen **8B adversary** (~13x parameter gap). The goal: train the small model with GRPO until it catches what the big one tries to hide.

![CounterFeint on HuggingFace Spaces](assets/HF%20Spaces.png)

### Episode Flow

Each episode is a multi-round match between the three agents:

```
reset(task_id, seed)
  |
  v
+-- Round 1..N (up to max_rounds) --------------------------+
|                                                            |
|  FRAUDSTER TURN                                            |
|    -> propose_ad / modify_pending_ad / end_turn            |
|    (capped at max_proposals across the match)              |
|                                                            |
|  INVESTIGATOR TURN                                         |
|    -> investigate (spend 1 budget to reveal a signal)      |
|    -> verdict (approve / reject / escalate)                |
|    -> link_accounts (flag fraud ring connections)           |
|    (capped at action_budget across the match)              |
|                                                            |
+------------------------------------------------------------+
  |
  v
AUDIT PHASE
  -> Auditor grades Investigator reasoning (Track A)
     + Fraudster plausibility (Track B)
  -> Episode ends, rewards computed
```

### Three Escalating Tasks

| Task | Base Ads | Budget | Fraudster Curriculum | Challenge |
|---|---:|---:|---|---|
| **Basic Triage** | 5 | 25 | 3 proposals, 4 rounds, easy categories only | Learn the investigate -> verdict loop |
| **Sophisticated Fraud** | 12 | 30 | 6 proposals, 4 rounds, mid-tier categories | Triage under budget pressure (~2.5 actions/ad) |
| **Fraud Network Detection** | 20 | 35 | 7 proposals, 5 rounds, full category palette + rings | Cross-ad reasoning to detect coordinated networks (~1.75 actions/ad) |

Task 3 introduces **fraud rings** - clusters of 3-5 ads controlled by the same actor using varied graph topologies (cliques, chains, hub-and-spoke), modeled after real CIB operations from Meta's Adversarial Threat Reports. The Fraudster's curriculum also escalates: more proposals, more rounds, and access to network-level fraud categories that produce coordinated ring patterns.

## What Makes It Realistic

**Budget constraints force triage.** Every investigation costs budget. Miss a fraud ad? It auto-approves at episode end with the full false-negative penalty (-0.50). Over-investigate one ad? You run out before reviewing the rest. This is the same tradeoff real reviewers face daily.

**Meta Purple Llama integration.** One of the six investigation tools is a `policy_classifier` simulating **Llama Guard 3** / Purple Llama safety screening - returning safe/unsafe verdicts, triggered categories (S1-S13), and trust & safety fraud markers (urgency language, fake authority, get-rich-quick signals).

**Meta policy taxonomy.** Every ad carries a citation grounded in Meta's published transparency policies (e.g., `FSDP-IF-03` for Fraud, Scams and Deceptive Practices). The Auditor checks whether the Investigator cites the correct policy section in its rationale.

**CIB-modeled fraud rings.** Task 3 networks are modeled on real Coordinated Inauthentic Behaviour operations from Meta's quarterly threat reports - Ghana DigitSol-style cliques (Q3 2020), Benin Digited-style chains (Q1 2021), and China-Russia-style hub-spoke networks (Q3 2022).

**Asymmetric rewards.** False negatives (fraud goes live) are penalized 5x more than false positives. Correct network link detection pays +0.40 because coordinated fraud is the highest-value catch. A calibration bonus rewards agents whose stated confidence correlates with actual accuracy.

| Investigator Reward Flow | Fraudster Reward Flow |
|:---:|:---:|
| ![Investigator Reward](assets/Investigator%20Reward.png) | ![Fraudster Reward](assets/Fraudster%20Reward.png) |

## Training

I train the Investigator using **GRPO** (Group Relative Policy Optimization) with **QLoRA** on `Qwen/Qwen3-0.6B` via HuggingFace TRL. The entire trainable footprint is ~600MB, fitting comfortably on a free Colab T4.

**Pipeline:**
1. **Collect rollouts** - run multi-agent episodes in-process (no HTTP server) with a scripted Fraudster and Auditor
2. **Score with a proxy reward** - a deterministic, verifiable reward function that scores each completion on schema validity, observation coherence, action correctness, and evidence citation quality
3. **GRPO update** - TRL's `GRPOTrainer` uses reward variance across completions to compute advantages

The proxy reward uses **continuous components** - partial credit for almost-valid JSON, graduated bonuses for evidence density, conciseness rewards - so GRPO always has a meaningful signal.

**Training notebook:** [`training/official_hf_training.ipynb`](training/official_hf_training.ipynb)

### Results

![GRPO Training Curves - Loss, Reward, KL Divergence](assets/Loss-reward-KL%20curve.png)

**Training dynamics:** 24 GRPO steps showed consistent non-zero advantage signal (loss oscillating between -0.09 and +0.20), confirming the pipeline produces meaningful gradients. Mean reward trends upward from -0.16 to -0.10, while KL divergence grows steadily — the model is learning to diverge from the base policy in a controlled way. Training was early-stopped at step 24/71 due to hackathon time constraints.

See the full training log and curves in [`training/official_hf_training.ipynb`](training/official_hf_training.ipynb).

## Quick Start

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

```python
from counterfeint import AdFraudEnv, AdReviewAction

with AdFraudEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=42, task_id="task_1")

    result = env.step(AdReviewAction(
        action_type="investigate",
        ad_id="ad_001",
        investigation_target="landing_page",
    ))

    result = env.step(AdReviewAction(
        action_type="verdict",
        ad_id="ad_001",
        verdict="reject",
        confidence=0.9,
    ))
```

**Train:** Open [`training/official_hf_training.ipynb`](training/official_hf_training.ipynb) on a T4 GPU. Set `MODE = "demo"` for ~30 min, or `MODE = "proper"` for ~3 hours.

## Project Structure

```
counterfeint/
├── server/           # FastAPI app, core environment, referee state machine
├── agents/           # LLM policy classes (HFInvestigator, LLMFraudster, prompts)
├── scripted/         # Deterministic baselines (ScriptedInvestigator, ReactiveFraudster, HeuristicAuditor)
├── data/             # Ad generation, fraud patterns, CIB network topologies, Meta policy taxonomy
├── graders/          # Task graders, auditor tracks (reasoning quality + plausibility)
├── training/         # GRPO training notebook, proxy reward, rollout collector
├── tests/            # Test suite
├── openenv.yaml      # OpenEnv manifest
└── Dockerfile        # Docker build for HF Spaces
```

## Links

| Resource | URL |
|---|---|
| Live Environment | [HF Space](https://huggingface.co/spaces/QuantumTransformer/CounterFeint) |
| Training Notebook | [`training/official_hf_training.ipynb`](training/official_hf_training.ipynb) |
| Blog Post | [`blog.md`](blog.md) |
| GitHub | [CounterFeint](https://github.com/Abhijithreddydasari/CounterFeint) |

## License

BSD 3-Clause License
