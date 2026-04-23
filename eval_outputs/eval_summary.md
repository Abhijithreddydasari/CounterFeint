# CounterFeint Eval: `scripted_baseline` → `after_grpo (placeholder)`

Held-out sweep across 3 tasks × 10 seeds each, evaluated against `ReactiveFraudster` (stable adversary).

| Task | Metric | scripted_baseline | after_grpo (placeholder) | Delta |
|------|--------|--------|-------|-------|
| task_1 | grader_score (↑) | 0.500 | 0.850 | +0.350 ↑ |
| task_1 | track_a_score (↑) | 0.800 | 0.940 | +0.140 ↑ |
| task_1 | n_fraud_leaks (↓) | 4.000 | 1.000 | -3.000 ↑ |
| task_1 | budget_used_pct (↓) | 1.000 | 0.700 | -0.300 ↑ |
| task_2 | grader_score (↑) | 0.420 | 0.780 | +0.360 ↑ |
| task_2 | track_a_score (↑) | 0.768 | 0.912 | +0.144 ↑ |
| task_2 | n_fraud_leaks (↓) | 4.000 | 1.000 | -3.000 ↑ |
| task_2 | budget_used_pct (↓) | 1.000 | 0.700 | -0.300 ↑ |
| task_3 | grader_score (↑) | 0.300 | 0.620 | +0.320 ↑ |
| task_3 | track_a_score (↑) | 0.720 | 0.848 | +0.128 ↑ |
| task_3 | n_fraud_leaks (↓) | 4.000 | 1.000 | -3.000 ↑ |
| task_3 | budget_used_pct (↓) | 1.000 | 0.700 | -0.300 ↑ |

