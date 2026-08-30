# CounterFeint — Frozen Claims Registry

Last updated: 2026-08-29. Every paper claim must trace to an artifact in `experiments/outputs/`.

## Supported claims (with evidence)

| Claim | Evidence | Status |
|-------|----------|--------|
| Three-agent FraudArena with budgeted investigation | `server/referee.py`, `tests/test_three_agent_episode.py` | Implemented |
| Deterministic dual-track Auditor (Track A + Track B) | `graders/auditor_track_a.py`, `graders/auditor_track_b.py` | Implemented |
| Paired-delta corruption eval (100 trajectories, 1500 trials) | `experiments/outputs/auditor_expanded/table2_per_check.md` | Measured |
| Exact checks: branding_anomaly F1=1.00, incoherent_rationale F1=1.00, missing_citation F1=0.98 | Table 2 | Strong |
| parameter_mismatch F1=0.86, gibberish F1=0.67 | Table 2 | Useful, extra FPs |
| Removing Track A raises Investigator reward on citation/coherence attacks (Δinv ≈ +2.11) | `table3_ablation.md`, `acceptance_gate.json` | Strong anti-gaming |
| Removing the plausibility gate raises Fraudster reward on gibberish (1.02 → 4.50, Δ ≈ +3.48) | Table 3 | Strong anti-gaming |
| Same gate raises Fraudster reward on grader-trigger, TLD, template, and mismatch attacks (Δ +1.9 to +2.5) | Table 3 | Strong |
| Clean-control flag rates on unmodified scripted trajectories are high for citation/mismatch | Table 2 clean-control | Reported, not treated as corruption FPs |
| Deterministic Auditor runtime ~0.12 ms/call (CPU) | `runtime_cost.json` | Measured |
| Offline Qwen3.5-9B judge: 46/50 parsed, 14.1 s, ~$0.008/ep | `runtime_cost.json`, `llm_flags.json` | Measured |
| LLM type recall vs attack key 0.35 (det 0.93); presence F1 0.84 vs 0.89 | `table4_human.md` §A | LLM does not beat det; TN=0 both |

## Weak / failure-analysis (report, do not headline)

| Claim | Evidence | Notes |
|-------|----------|-------|
| bias F1=0.33, miscalibration F1=0.41, inconsistency F1=0.44, template_repetition F1=0.33 | Table 2 | High recall, low precision; extra flags on other attacks |
| slice_bias hit rate 0.60; template_spam 0.67; overconfidence 0.88 | Table 2 hit rates | Partial coverage |
| investigation_farming, unrelated_links | no Track A flag | Documented verifier gaps |
| paraphrased_spam | expected LLM-only | Deterministic novelty misses paraphrase (hit 0.00) |

## Unsupported — do NOT claim

| Claim | Why |
|-------|-----|
| Hybrid Auditor empirically beats deterministic on semantic coverage | Table 4A: hybrid type recall = det (0.93); LLM 0.35. Human Table 4B pending |
| LLM-only Auditor is calibrated | Flags 10/10 clean cases (TN=0 vs attack key); type recall 0.35 |
| Trained 0.6B beats frozen 8B fraudster | Training uses `ReactiveFraudster`; 8B Ollama path 0/6 episodes |
| +0.18 / +0.35 grader improvement | Placeholder eval purged |
| GRPO materially improves fraud detection | Only +0.005 on 3 non-held-out episodes |
| Reduces real ad fraud / production validated | Simulation only |
| Solved multi-agent oversight | Deterministic rules, one domain |

## Paper 1 results (2026-08-29, `experiments/outputs/auditor_expanded/`)

- 100 base trajectories (tasks 1–3), 1500 corruption trials, 1148 applicable.
- Ablation: 30 trajectories × 15 attacks × verifier arms (7200 rows). Anti-gaming gate **pass**.
- Human sheet: 50 blinded cases in `annotation/` (labels pending).
- LLM judge: 46/50 ok, 14.1 s, ~$0.008/ep. Type recall 0.35 vs attack key. Hybrid claim **blocked**.

## Artifact locations

- Expanded suite: `experiments/outputs/auditor_expanded/`
- Tables 1–4, Figures 1–3, `acceptance_gate.json`, `runtime_cost.json`
- Papers: `papers/verify_agents/`, `papers/trustworthy_ai/`
