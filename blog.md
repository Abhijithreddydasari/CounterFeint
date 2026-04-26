# CounterFeint: Teaching a 0.6B LLM to Investigate Ad Fraud

Ad fraud is a $100B/year problem, and catching the sophisticated stuff isn't classification - it's investigation. I built an RL environment to teach a small LLM how to do it.



---

## Why This Problem?

Platforms like Meta review billions of ads daily. The easy scams get caught by rule-based filters. The hard part is **sophisticated fraud** - ads that look legitimate on the surface but reveal their nature only when you dig into payment histories, inspect landing page registrars, or notice that five seemingly unrelated accounts share the same payment processor. This isn't classification. It's **sequential investigation under uncertainty**, where every investigation has a cost and every missed fraud ad goes live.

That workflow - triage, investigate, decide under budget pressure - is exactly where RL-trained LLMs could add real value. Not replacing human reviewers, but handling the first pass so humans focus on the hardest cases. I participated in the Meta Scalar OpenEnv Hackathon solo and chose this problem because it maps directly to how trust & safety teams operate, and sits at the intersection of multi-agent interaction, long-horizon planning, and real-world applicability.

## How CounterFeint Works

CounterFeint is an [OpenEnv](https://github.com/open-env/open-env)-compatible environment built around a **FraudArena** - a three-agent adversarial game:

- A **Fraudster** proposes ads designed to slip past review
- An **Investigator** reviews a queue of ads under a limited budget, investigates suspicious signals, and renders verdicts (approve / reject / escalate)
- An **Auditor** grades the Investigator's reasoning using deterministic scorecards

The Investigator is the agent I train. It's a **Qwen3-0.6B** model with QLoRA - going up against a frozen **Llama 3.1 8B** Fraudster. That's a ~13x parameter gap. The claim I wanted to test: task-specific GRPO can close a large capability gap when the task is well-defined and the reward signal is rich.

![CounterFeint on HuggingFace Spaces](assets/HF%20Spaces.png)

### Making It Realistic

Most RL environments for LLMs are clean abstractions - math problems, grid worlds, coding tasks. I wanted something messier, closer to how the real world works. A few design decisions that drove realism:

**Budget constraints force real triage.** Each investigation costs one budget point. On Task 3, there are 20 ads but only 35 budget points (~1.75 actions/ad) - you physically cannot investigate everything. Miss a fraud ad? It auto-approves at episode end with a -0.50 penalty. Same resource-allocation problem real reviewers face.

**Meta's safety stack, simulated.** One of six investigation tools is a `policy_classifier` simulating Llama Guard 3 / Purple Llama - returning safe/unsafe verdicts, triggered safety categories (S1-S13), and fraud markers (urgency language, fake authority, get-rich-quick). Every ad carries a citation from Meta's published policy taxonomy (e.g., `FSDP-IF-03`). The Auditor checks whether the Investigator cites the correct policy in its rationale - pushing toward evidence-based reasoning.

**CIB-modeled fraud rings.** Task 3 fraud networks are modeled on real Coordinated Inauthentic Behaviour operations from Meta's quarterly Adversarial Threat Reports - Ghana DigitSol-style cliques, Benin Digited-style chains, China-Russia hub-spoke networks. Individual ring members look borderline; the signal only emerges through cross-ad reasoning.

**Asymmetric rewards.** False negatives are penalized 5x more than false positives. Correct network link detection pays +0.40. A calibration bonus rewards agents whose stated confidence actually tracks their accuracy.

| Investigator Reward Flow | Fraudster Reward Flow |
|:---:|:---:|
| ![Investigator Reward](assets/Investigator%20Reward.png) | ![Fraudster Reward](assets/Fraudster%20Reward.png) |

## Training: GRPO on a T4

I used **GRPO** (Group Relative Policy Optimization) via HuggingFace TRL - the same algorithm family used in DeepSeek-R1's training. The key property: GRPO doesn't need human preference labels, just a **verifiable reward function** and sufficient reward variance across completions.

### The proxy reward

This was the biggest technical challenge. TRL's `GRPOTrainer` generates fresh completions internally and passes them to your reward function. My first runs showed a flat reward curve. Zero learning. The lookup was returning a constant penalty for everything.

The fix: a **per-completion proxy reward** that scores any (prompt, completion) pair without touching the environment server:

1. **Schema validity** - does the completion parse as valid JSON matching the action schema? (Partial credit for almost-valid JSON)
2. **Coherence** - does the referenced `ad_id` appear in the prompt's observation?
3. **Action correctness** - does the action class match what a successful episode recorded?
4. **Evidence quality** - does the rationale cite concrete signals (payment IDs, policy codes)?
5. **Conciseness** - shorter valid completions score higher

These components are **continuous**, not binary - giving GRPO meaningful variance even when most completions are mediocre. This design is consistent with how verifiable-reward GRPO recipes work in projects like Open-R1.

### QLoRA keeps it accessible

The entire setup fits on a free Colab T4: Qwen3-0.6B in 4-bit quantization (~400MB), LoRA adapter (rank 16, ~12MB trainable), plus the in-process environment and scripted opponents. No separate inference server needed during training.

## Challenges

A few things that cost real time:

**System prompt specificity.** Early prompts were too vague and the model invented action types, hallucinated ad IDs, or produced nonexistent investigation targets. Small models need precise instructions - explicit enum lists, concrete examples for every action type, and an explicit "output one JSON object per step, nothing else" directive. The difference between a working and broken prompt was surprisingly narrow.

**Context window overflow.** On Task 3 with 20 ads, accumulated investigation findings can exceed the context limit. I implemented truncation strategies that keep recent and relevant findings while trimming older history - but getting this right without losing critical cross-ad signals for fraud ring detection required careful tuning.

**HF vs local prompt handling.** The chat template for system/user prompts differs between local `model.generate()` and the HuggingFace Inference API. Ensuring `HFInvestigator` (training) and `LLMInvestigator` (inference) produce identical prompt assembly was essential for the trained adapter to transfer cleanly.

## Results

**Baseline (Qwen3-0.6B, 4-bit, no fine-tuning):**


| Task                         | Score     |
| ---------------------------- | --------- |
| Task 1 (Basic Triage)        | 0.543     |
| Task 2 (Sophisticated Fraud) | 0.576     |
| Task 3 (Fraud Networks)      | 0.180     |
| **Mean**                     | **0.433** |








## Try It

**[Live environment on HuggingFace Spaces](https://huggingface.co/spaces/QuantumTransformer/CounterFeint)**

**[Training notebook](training/official_hf_training.ipynb)** - self-contained, runs on a Colab T4. Set `MODE = "demo"` for ~30 min, `MODE = "proper"` for ~3 hours.

**[GitHub](https://github.com/Abhijithreddydasari/CounterFeint)**

*Built by [Abhijith Reddy](https://github.com/Abhijithreddydasari) (solo) for the Meta Scalar OpenEnv PyTorch HuggingFace Hackathon, 2026.*