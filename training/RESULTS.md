# CounterFeint - Training Results

Live tracking of every baseline + training run. Append rows as runs finish.

---

## Baseline (BEFORE training)

Hardware: T4 medium (HF Spaces), 4-bit quantisation, no fine-tuning.

| Model              | task_1 | task_2 | task_3 |  Mean  | Fallback Rate | Run Date     |
|--------------------|-------:|-------:|-------:|-------:|--------------:|--------------|
| Qwen/Qwen3-0.6B    |  0.543 |  0.576 |  0.180 |  0.433 |        83.51% | 2026-04-26   |

Source: `baseline_outputs/qwen3-0.6b/baseline_results.json` on HF Space `QuantumTransformer/CounterFeint-train` (path `/data/baseline_outputs/`).

---

## Trained (AFTER training)

| Model + Config                | task_1 | task_2 | task_3 |  Mean  | Delta vs base | Run Date |
|-------------------------------|-------:|-------:|-------:|-------:|--------------:|----------|
| _pending Qwen3.5-2B demo r1_  |    -   |    -   |    -   |    -   |             - | -        |

Source: `outputs/<TRAINED_TAG>/eval_summary.json` on HF Space (path `/data/outputs/`).

---

## Notes

- Fallback rate = % of LLM calls that produced invalid JSON / wrong schema and fell back to ScriptedInvestigator. High fallback rate at baseline = strong learning signal for GRPO.
- task_3 is hardest (24 ads + cross-ad linking via `link_accounts`). 0.6B baseline of 0.18 is expected — small models can't handle the link-accounts logic without training.
