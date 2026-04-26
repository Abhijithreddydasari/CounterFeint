"""
Training utilities for the CounterFeint Investigator.

The training notebook (``train_investigator.ipynb``) imports everything
it needs from this package, keeping the notebook to thin orchestration
cells.

Modules:

  * :mod:`.rollout`      — episode-collection helpers (entry point:
                           :func:`.rollout.collect_dataset`).
  * :mod:`.proxy_reward` — verifiable per-completion reward function
                           used by ``GRPOTrainer`` (entry point:
                           :func:`.proxy_reward.make_proxy_reward_fn`).
"""

from .proxy_reward import (
    build_gold_lookup,
    make_proxy_reward_fn,
    proxy_reward_one,
)
from .rollout import (
    InvestigatorTrainingSample,
    RecordingHFInvestigator,
    TracingPolicy,
    classify_action,
    collect_dataset,
    collect_dataset_in_process,
    collect_episode,
    collect_episode_in_process,
    records_to_samples,
    samples_to_hf_dataset,
    summarise_action,
)

__all__ = [
    "InvestigatorTrainingSample",
    "RecordingHFInvestigator",
    "TracingPolicy",
    "build_gold_lookup",
    "classify_action",
    "collect_dataset",
    "collect_dataset_in_process",
    "collect_episode",
    "collect_episode_in_process",
    "make_proxy_reward_fn",
    "proxy_reward_one",
    "records_to_samples",
    "samples_to_hf_dataset",
    "summarise_action",
]
