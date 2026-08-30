# Who Verifies the Verifier? Stress-Testing a Dual-Track Auditor in a Three-Agent Fraud Arena

**Target:** Who Verifies the Agents @ NeurIPS 2026 (4-page demo)

## Abstract

Reward hacking of verifiers is the failure mode of long-horizon agent eval. We study the verifier itself. CounterFeint is an OpenEnv three-agent ad-fraud arena whose Auditor decomposes verification into Track A (investigator reasoning: citation, coherence, calibration, consistency, bias) and Track B (fraudster plausibility: gibberish, parameter mismatch, template novelty, grader-trigger leakage). On 100 paired trajectories (1500 corruption trials) we measure *new* flags caused by controlled attacks, not flags already present on unmodified rollouts. Exact checks reach F1 0.98–1.00. Removing Track A raises Investigator reward by Δ≈+2.11 on fabricated/unsupported rationales; removing the Track B plausibility gate raises Fraudster reward from 1.02 to 4.50 on high-severity gibberish. Weak checks (bias, calibration, template novelty) are reported as failure analysis. An offline Qwen3.5-9B judge (46/50 parsed, 14.1 s, $0.008/ep) does not beat deterministic type recall on the attack key (0.35 vs 0.93). A 50-case human sheet is specified; anti-gaming does not depend on it.

## 1. Problem

Scalar outcome rewards miss reasoning quality and adversary plausibility. LLM-as-judge verifiers are expensive and gameable. We ask: which verifier *components* actually block specification gaming, and which fail?

**Workshop fit.** Pillar 1: red-team the verifier. Pillar 3: heterogeneous signals (correctness, calibration, grounding, consistency, plausibility, cost). Pillar 2 (supporting): a replayable simulator where every action, finding, verdict, and reward is auditable.

## 2. Environment

Three roles share a budgeted episode (Figure 1): **Fraudster** proposes ads; **Investigator** investigates / verdicts / links under an action budget; **Auditor** emits typed flags after the episode. Unreviewed fraud auto-approves. The default policies here are scripted (`ScriptedInvestigator`, `ReactiveFraudster`) so trajectories are CPU-reproducible; the Auditor under test is the deterministic Track A/B pipeline, not a learned overseer.

**Rewards.** Investigator: outcome grader + per-verdict rationale bonus (stripped by `missing_citation` / `incoherent_rationale`) − capped inconsistency penalty. Fraudster: $\sum_i \mathrm{severity}_i \cdot \mathrm{plausibility}_i$ for unbanned ads, minus a ban penalty. Plausibility is the sole realism chokepoint.

## 3. Evaluation (paired delta)

Unmodified scripted trajectories already fire many flags (clean-control rates: missing_citation 0.99, parameter_mismatch 1.00, miscalibration 0.58). Counting those as corruption false positives overstates FPs. Protocol:

1. Audit the natural trajectory (clean-control rates).
2. Sanitize a control so the target check is a true negative, then apply the attack.
3. Score **new** flags at `(flag_type, ad_id)`, with bootstrap 95% CIs.

100 base episodes, stratified across tasks 1–3 (5 / 12 / 20 ads). 15 attacks (8 Investigator, 7 Fraudster). 1148 applicable trials.

## 4. Results

**Table 2 (detection).** Exact, locally checkable properties are strong: `incoherent_rationale` F1=1.00, `branding_anomaly` (grader-trigger strings) F1=1.00, `missing_citation` F1=0.98, `parameter_mismatch` F1=0.86. `gibberish` has perfect recall but precision 0.50 (F1=0.67) because other surface attacks also look non-wordlike. Slice/calibration/consistency/template checks have F1 0.33–0.44: they fire on the intended attack (high recall) but also on collateral mutations. We keep them as failure analysis, not silent deletions.

Documented gaps with no deterministic flag: investigation farming, unrelated account links, paraphrased template spam (LLM-expected).

**Table 3 (anti-gaming).** On the *same* attacked trajectory:

- Citation / coherence attacks: Investigator reward 1.01 (full) vs 3.13 (`no_track_a`), Δinv ≈ **+2.11**.
- Gibberish ads: Fraudster reward 1.02 (full) vs 4.50 (`no_plausibility_gate`), Δfrd ≈ **+3.48**.
- Grader-trigger, country/TLD, category/targeting, and exact template clones: Δfrd +1.88 to +2.53 when the gate is removed.
- Per-check drops (Table 3b): removing only `missing_citation` or `incoherent_rationale` reproduces the +2.11 Investigator lift; removing `template_repetition` or `branding_anomaly` lifts Fraudster +1.65–2.01. `overconfidence` / `slice_bias` per-check drops are ~0 (those flags do not bind reward).

Outcome-only (grader, no plausibility, no Track A) does not substitute for either track.

**Cost.** Deterministic audit ≈ 0.12 ms/episode on CPU. Offline Qwen3.5-9B (vLLM, A100-80GB): 46/50 cases parsed, mean 14.1 s, ≈$0.008/episode. The judge is not used for inline training rewards.

## 5. Hybrid policy (specified; not superior on this run)

Exact flags (`gibberish`, `miscalibration`, `bias`, `inconsistency`, `branding_anomaly`) stay deterministic. Semantic flags (`missing_citation`, `incoherent_rationale`, `parameter_mismatch`, `template_repetition`) may be confirmed or added by Qwen3.5-9B.

**Table 4A (vs attack key, 50 cases, 10 clean).** Deterministic presence F1=0.89 (flags every case, including all 10 clean; TN=0). LLM F1=0.84, type recall 0.35 vs 0.93 deterministic. Hybrid presence = deterministic. This is *not* a hybrid-superiority result. Table 4B vs one human annotator is pending `labels.csv`.

## 6. Limitations

Synthetic ads and scripted counterparties; one annotator planned, not crowd GT; weak checks remain noisy; no claim of production fraud reduction or self-evolving agents. Training/GRPO belongs to a companion paper.

## References

- AJ-Bench (environment-aware agent judges)
- AgentAuditor (learned safety evaluator)
- AuditFlow (symbolic verification environments)
- OpenEnv (Hugging Face / Meta, 2026)
