"""
Fraud network (ring) generation for Task 3.

A fraud ring is a clique of 3-5 ads that share 2+ signals:
payment_method_id, creative_template, targeting_overlap, domain_registrar.
Individual ads in a ring may look borderline; the signal is in the connections.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class FraudRing:
    ring_id: str
    member_ad_ids: List[str]
    shared_signals: Dict[str, str]  # signal_type -> shared_value

    @property
    def size(self) -> int:
        return len(self.member_ad_ids)


def generate_fraud_networks(
    rng: random.Random,
    n_rings: int,
    available_fraud_ad_ids: List[str],
) -> Tuple[List[FraudRing], Dict[str, List[str]]]:
    """
    Generate fraud ring structures from available fraud ad IDs.

    Returns:
        rings: list of FraudRing objects
        ad_to_rings: mapping from ad_id to list of ring_ids it belongs to
    """
    rings: List[FraudRing] = []
    ad_to_rings: Dict[str, List[str]] = {}
    used_ads: Set[str] = set()

    remaining = list(available_fraud_ad_ids)
    rng.shuffle(remaining)

    for i in range(n_rings):
        ring_size = rng.randint(3, min(5, len(remaining)))
        if ring_size < 3 or len(remaining) < 3:
            break

        members = remaining[:ring_size]
        remaining = remaining[ring_size:]

        shared_payment = f"pmt_ring_{rng.randint(10000, 99999)}"
        shared_registrar = rng.choice(["Njalla (privacy)", "Epik", "NameSilo", "Tucows (privacy proxy)"])
        shared_template = f"tmpl_{rng.randint(1000, 9999)}"
        shared_targeting = rng.choice([
            "Men 25-45, crypto+investing, US+UK+AU",
            "Adults 18-35, tech+gaming, worldwide",
            "Women 25-55, health+beauty, US+CA",
            "Adults 30-60, finance+real-estate, US+UK",
        ])

        signal_pool = {
            "payment_method": shared_payment,
            "domain_registrar": shared_registrar,
            "creative_template": shared_template,
            "targeting_overlap": shared_targeting,
        }

        signal_keys = list(signal_pool.keys())
        rng.shuffle(signal_keys)
        n_shared = rng.randint(2, len(signal_keys))
        shared_signals = {k: signal_pool[k] for k in signal_keys[:n_shared]}

        ring_id = f"ring_{i}"
        ring = FraudRing(
            ring_id=ring_id,
            member_ad_ids=members,
            shared_signals=shared_signals,
        )
        rings.append(ring)

        for ad_id in members:
            used_ads.add(ad_id)
            ad_to_rings.setdefault(ad_id, []).append(ring_id)

    return rings, ad_to_rings


def get_ring_shared_signal_text(ring: FraudRing) -> str:
    """Describe the shared signals in a ring (for grader/debug use)."""
    lines = [f"Fraud Ring {ring.ring_id} ({ring.size} members):"]
    lines.append(f"  Members: {', '.join(ring.member_ad_ids)}")
    lines.append("  Shared signals:")
    for signal_type, value in ring.shared_signals.items():
        lines.append(f"    - {signal_type}: {value}")
    return "\n".join(lines)
