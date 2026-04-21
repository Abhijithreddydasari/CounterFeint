"""Smoke test: three-agent scripted baseline against the running server.

Run from workspace root:
    python counterfeint/_tmp_driver_smoke.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from counterfeint.inference import run_three_agent_episode
from counterfeint.scripted import (
    GibberishFraudster,
    HeuristicAuditor,
    ReactiveFraudster,
    ScriptedInvestigator,
)

BASE = "http://127.0.0.1:8765"

print("\n=== ReactiveFraudster vs ScriptedInvestigator + HeuristicAuditor ===")
result = run_three_agent_episode(
    "task_1",
    fraudster_policy=ReactiveFraudster(seed=7),
    investigator_policy=ScriptedInvestigator(),
    auditor_policy=HeuristicAuditor(),
    env_base_url=BASE,
    seed=7,
)
print(json.dumps({k: v for k, v in result.items() if k != "final_state"}, indent=2))

print("\n=== GibberishFraudster (plausibility sanity check) ===")
result2 = run_three_agent_episode(
    "task_1",
    fraudster_policy=GibberishFraudster(seed=11),
    investigator_policy=ScriptedInvestigator(),
    auditor_policy=HeuristicAuditor(),
    env_base_url=BASE,
    seed=11,
)
print(json.dumps({k: v for k, v in result2.items() if k != "final_state"}, indent=2))
