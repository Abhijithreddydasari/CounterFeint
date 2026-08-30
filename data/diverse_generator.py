"""
Two-stage diverse synthetic episode scenario sampling.

Stage 1: programmatic axis sampling (ground truth lives here).
Stage 2: optional LLM surface realization (see ``surface_realizer.py``).

Diversity filter: reject ads above cosine similarity threshold vs corpus.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Scenario axes
VIOLATION_TYPES = [
    "fake_giveaway",
    "miracle_cure",
    "counterfeit_goods",
    "advance_fee",
    "fake_crypto",
    "celebrity_endorsement_fraud",
    "clone_brand",
    "gray_area_supplements",
    "network_clique",
    "network_chain",
    "network_hub",
]
INDUSTRIES = [
    "ecommerce",
    "finance",
    "health",
    "gaming",
    "education",
    "local_services",
]
GEOGRAPHIES = ["US", "UK", "DE", "NG", "IN", "BR", "AU"]
SOPHISTICATION = ["novice", "intermediate", "advanced"]
RING_TOPOLOGIES = ["none", "star", "chain", "bipartite", "clique"]
BUDGET_REGIMES = ["generous", "standard", "tight"]
FRAUD_PREVALENCE = [0.1, 0.25, 0.4, 0.55, 0.6]

# Held-out for OOD test split
OOD_VIOLATION_HOLDOUT = {"network_hub"}
OOD_TOPOLOGY_HOLDOUT = {"bipartite"}


@dataclass
class ScenarioSpec:
    """Stage-1 structured scenario; labels are authoritative."""

    scenario_id: str
    seed: int
    violation_type: str
    industry: str
    geography: str
    sophistication: str
    ring_topology: str
    budget_regime: str
    fraud_prevalence: float
    n_ads: int
    n_fraud: int
    split: str = "train"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "violation_type": self.violation_type,
            "industry": self.industry,
            "geography": self.geography,
            "sophistication": self.sophistication,
            "ring_topology": self.ring_topology,
            "budget_regime": self.budget_regime,
            "fraud_prevalence": self.fraud_prevalence,
            "n_ads": self.n_ads,
            "n_fraud": self.n_fraud,
            "split": self.split,
            "metadata": dict(self.metadata),
        }


def _scenario_id(seed: int, axes: Tuple[str, ...]) -> str:
    raw = f"{seed}|{'|'.join(axes)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _n_ads_for_budget(regime: str) -> int:
    return {"generous": 12, "standard": 20, "tight": 25}[regime]


def sample_scenario(
    *,
    seed: int,
    split: str,
    rng: Optional[random.Random] = None,
    compositional_holdout: Optional[Set[Tuple[str, str]]] = None,
) -> ScenarioSpec:
    """Sample one scenario spec from the axis product."""
    rng = rng or random.Random(seed)
    violation = rng.choice(VIOLATION_TYPES)
    industry = rng.choice(INDUSTRIES)
    geography = rng.choice(GEOGRAPHIES)
    sophistication = rng.choice(SOPHISTICATION)
    topology = rng.choice(RING_TOPOLOGIES)
    budget = rng.choice(BUDGET_REGIMES)
    prevalence = rng.choice(FRAUD_PREVALENCE)

    n_ads = _n_ads_for_budget(budget)
    n_fraud = max(1, int(n_ads * prevalence))

    if split == "test_ood":
        violation = rng.choice(list(OOD_VIOLATION_HOLDOUT))
        topology = rng.choice(list(OOD_TOPOLOGY_HOLDOUT))
    elif split == "test_compositional" and compositional_holdout:
        pair = rng.choice(list(compositional_holdout))
        violation, topology = pair

    sid = _scenario_id(
        seed,
        (violation, industry, geography, sophistication, topology, budget, split),
    )
    return ScenarioSpec(
        scenario_id=sid,
        seed=seed,
        violation_type=violation,
        industry=industry,
        geography=geography,
        sophistication=sophistication,
        ring_topology=topology,
        budget_regime=budget,
        fraud_prevalence=prevalence,
        n_ads=n_ads,
        n_fraud=n_fraud,
        split=split,
    )


def generate_split(
    *,
    split: str,
    n_scenarios: int,
    seed_start: int = 10000,
    compositional_holdout: Optional[Set[Tuple[str, str]]] = None,
) -> List[ScenarioSpec]:
    specs: List[ScenarioSpec] = []
    for i in range(n_scenarios):
        specs.append(
            sample_scenario(
                seed=seed_start + i,
                split=split,
                compositional_holdout=compositional_holdout,
            )
        )
    return specs


def jaccard_similarity(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard as a cheap diversity proxy without embeddings."""
    def ngrams(s: str) -> Set[str]:
        s = s.lower()
        return {s[i : i + n] for i in range(max(0, len(s) - n + 1))}

    sa, sb = ngrams(a), ngrams(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def passes_diversity_filter(
    text: str,
    corpus: List[str],
    *,
    max_similarity: float = 0.85,
) -> bool:
    for existing in corpus:
        if jaccard_similarity(text, existing) > max_similarity:
            return False
    return True


def build_dataset_manifest(
    output_path: Path,
    *,
    train_n: int = 200,
    val_n: int = 40,
    test_comp_n: int = 30,
    test_ood_n: int = 30,
) -> Dict[str, Any]:
    """Write a manifest of scenario specs for all splits."""
    compositional_holdout = {
        (v, t)
        for v in VIOLATION_TYPES[:4]
        for t in RING_TOPOLOGIES[1:3]
    }
    manifest = {
        "train": [
            s.to_dict()
            for s in generate_split(split="train", n_scenarios=train_n, seed_start=10000)
        ],
        "val": [
            s.to_dict()
            for s in generate_split(split="val", n_scenarios=val_n, seed_start=50000)
        ],
        "test_compositional": [
            s.to_dict()
            for s in generate_split(
                split="test_compositional",
                n_scenarios=test_comp_n,
                seed_start=60000,
                compositional_holdout=compositional_holdout,
            )
        ],
        "test_ood": [
            s.to_dict()
            for s in generate_split(
                split="test_ood", n_scenarios=test_ood_n, seed_start=70000
            )
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/data/dataset_manifest.json"),
    )
    args = parser.parse_args()
    build_dataset_manifest(args.output)
    print(f"Wrote {args.output}")
