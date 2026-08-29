# CounterFeint — Frozen Claims Registry

Last updated: 2026-08-29. Every paper claim must trace to an artifact in `experiments/outputs/`.

## Supported claims (with evidence)

| Claim | Evidence | Status |
|-------|----------|--------|
| Three-agent FraudArena with budgeted investigation | `server/referee.py`, `tests/test_three_agent_episode.py` | Implemented |
| Deterministic dual-track Auditor (Track A + Track B) | `graders/auditor_track_a.py`, `graders/auditor_track_b.py` | Implemented |
| Plausibility gate blocks gibberish fraud reward | `graders/multi_agent_rewards.py` fraudster_reward | Implemented |
| Baseline Qwen3-0.6B mean grader 0.433 (task_1/2/3) | `training/RESULTS.md`, HF Space 2026-04-26 | Documented, artifact off-repo |
| Partial GRPO demo: Δ grader +0.005 on 3 eval episodes | `training/official_hf_training.ipynb` cell output | Weak, not for headline |

## Unsupported — do NOT claim in papers

| Claim | Why |
|-------|-----|
| Trained 0.6B beats frozen 8B fraudster | Training uses `ReactiveFraudster`; 8B Ollama path 0/6 episodes |
| +0.18 / +0.35 grader improvement | `eval_outputs/eval_summary.md` is synthetic placeholder |
| GRPO materially improves fraud detection | Only +0.005 on 3 non-held-out episodes |
| Reduces real ad fraud / production validated | Simulation only |
| Solved multi-agent oversight | Deterministic rules, one domain |

## Paper 1 experiment results (2026-08-29)

Corruption suite (30 base × 7 corruptions): see `experiments/outputs/corruption/corruption_summary.md`

| Flag | F1 | Notes |
|------|---:|-------|
| branding_anomaly | 1.000 | grader-trigger corruption |
| gibberish | 0.455 | high recall, FP from clean episodes |
| missing_citation | 0.444 | high recall |
| miscalibration | 0.456 | partial on overconfidence |
| incoherent_rationale | 0.000 | needs corruption tuning |

Anti-gaming ablation: gibberish fraudster reward drops from ~3.77 (full) to ~0.62 (full) vs 4.0 (no plausibility gate). See `experiments/outputs/ablation/`.

1. Environment supports inspectable long-horizon investigation episodes.
2. Track A/B decompose verification into outcome, reasoning, and plausibility.
3. Corruption suite shows per-flag precision/recall on independent ground truth.
4. Anti-gaming ablation: reward hacking succeeds when verifier components removed.

## Paper 2 (Trustworthy AI for Good) — additional claims (require new runs)

1. Fraud leak rate before/after training on held-out splits (with CIs).
2. Diverse synthetic data with compositional + OOD test splits.
3. Ring-link F1, policy-citation accuracy, real-world holdout eval.
4. Honest limitations: synthetic, dual-use, human oversight required.

## Artifact locations

- Corruption results: `experiments/outputs/corruption/`
- Ablation results: `experiments/outputs/ablation/`
- Eval sweeps: `experiments/outputs/eval/`
- Training runs: `experiments/outputs/training/`
- Papers: `papers/verify_agents/`, `papers/trustworthy_ai/`
