# CounterFeint — Training Results

Live tracking of every baseline + training run. **Only cite numbers from this file when the Source column points to a committed artifact.**

---

## Baseline (BEFORE training)

| Model | task_1 | task_2 | task_3 | Mean | Fallback Rate | Source |
|-------|-------:|-------:|-------:|-----:|--------------:|--------|
| Qwen/Qwen3-0.6B | 0.543 | 0.576 | 0.180 | 0.433 | 83.51% | HF Space 2026-04-26 (not in repo) |

---

## Interrupted / smoke runs (NOT headline results)

| Run | Notes | Source |
|-----|-------|--------|
| official_hf_training MODE=demo | KeyboardInterrupt step 24/71; Δ grader +0.005 on 3 episodes | notebook output |
| outputs/smoke checkpoint-3 | 3 GRPO steps, proxy reward only | `outputs/smoke/` |
| local_smoke | Constant -0.5 reward, zero grad | `training_outputs/local_smoke/` |

---

## Trained (AFTER training)

| Model + Config | task_1 | task_2 | task_3 | Mean | Delta vs base | Source |
|----------------|-------:|-------:|-------:|-----:|--------------:|--------|
| _pending_ | — | — | — | — | — | — |

---

## Eval protocol (for new runs)

- Held-out seeds: `eval_suite.EVAL_SEEDS` (35 episodes)
- Adversary: `ReactiveFraudster` for baselines; live LLM fraudster for robustness eval
- Decoding: guided JSON (vLLM XGrammar) — re-baseline under constraints before trained-vs-base
- Report: mean ± bootstrap 95% CI, fraud leaks, fallback rate separately
