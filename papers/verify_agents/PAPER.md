# CounterFeint: Verifying Investigative Agents Under Adaptive Adversarial Pressure

**Target:** Who Verifies the Agents @ NeurIPS 2026 (4-page demo)

## Abstract

We present CounterFeint, an OpenEnv-compatible three-agent environment for budget-constrained ad-fraud investigation. A deterministic dual-track Auditor decomposes verification into outcome correctness (Track A: investigator reasoning) and adversary plausibility (Track B: fraud surface realism). We validate the Auditor with a corruption suite that injects known defects into trajectories and measures per-flag precision and recall against independent ground truth. Anti-gaming ablations show that removing Track B allows severe-but-gibberish fraud to earn reward, and removing Track A allows fabricated rationales to pass.

## 1. Problem

Long-horizon investigative agents need verifiers that are inspectable, reproducible, and resistant to specification gaming. LLM-as-judge evaluators are costly and unstable; rule-only outcome metrics miss reasoning quality and adversarial plausibility.

## 2. CounterFeint Environment

Three roles: Fraudster (proposes ads), Investigator (investigate / verdict / link), Auditor (post-episode audit). Budget pressure forces triage; unreviewed fraud auto-approves.

## 3. Verification Stack

**Track A (Investigator):** calibration, citation, coherence, consistency, bias.

**Track B (Fraudster):** intrinsic consistency, grounding, gibberish detection, grader-trigger leakage, template novelty.

**Rewards:** `fraudster_reward = Σ(severity × plausibility)` for unbanned ads; investigator bonus weighted by per-ad plausibility.

## 4. Corruption Suite (Core Experiment)

For each base trajectory, apply controlled corruptions (flip verdicts, missing citations, gibberish ads, grader triggers, template spam). Ground truth = expected flag types. Report per-flag P/R/F1.

**Artifacts:** `experiments/outputs/corruption/corruption_report.json`

**Key result:** On gibberish-ads corruption, mean fraudster reward is 0.62 (full verifier) vs 4.0 (no plausibility gate) — gaming blocked by Track B.

## 5. Anti-Gaming Ablation

Arms: full / no Track A / no Track B / no plausibility gate / outcome-only.

**Hypothesis:** Gaming success rate rises when verifier components are removed.

**Artifacts:** `experiments/outputs/ablation/ablation_results.json`

## 6. Limitations

Synthetic domain; deterministic auditor (not LLM judge); training results pending on held-out eval. LLM auditor agreement study left for future work.

## References

- AJ-Bench (environment-aware agent judges)
- AgentAuditor (learned safety evaluator)
- AuditFlow (symbolic verification environments)
