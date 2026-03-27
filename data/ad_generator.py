"""
Synthetic ad queue generation.

Generates a complete queue of ads for a given task configuration,
including all pre-generated investigation data. When the agent
investigates, the environment just reveals pre-computed data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .advertiser_profiles import AdvertiserProfile, generate_advertiser_profile
from .fraud_patterns import FRAUD_TEMPLATES, LEGIT_TEMPLATES, AdTemplate
from .landing_pages import LandingPageData, generate_landing_page
from .network_generator import FraudRing, generate_fraud_networks


@dataclass
class TaskConfig:
    task_id: str
    name: str
    difficulty: str
    queue_size: int
    action_budget: int
    n_legit: int
    n_fraud: int
    n_escalate: int
    include_networks: bool
    n_fraud_rings: int
    allowed_difficulties: List[str]
    description: str


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "task_1": TaskConfig(
        task_id="task_1",
        name="Basic Ad Triage",
        difficulty="easy",
        queue_size=5,
        action_budget=25,
        n_legit=2,
        n_fraud=3,
        n_escalate=0,
        include_networks=False,
        n_fraud_rings=0,
        allowed_difficulties=["easy"],
        description=(
            "Learn the investigation loop. Queue of 5 ads with obviously "
            "fraudulent or clearly legitimate signals. Generous budget of 25 "
            "actions (5 per ad)."
        ),
    ),
    "task_2": TaskConfig(
        task_id="task_2",
        name="Sophisticated Fraud Under Budget Pressure",
        difficulty="medium",
        queue_size=12,
        action_budget=30,
        n_legit=5,
        n_fraud=5,
        n_escalate=2,
        include_networks=False,
        n_fraud_rings=0,
        allowed_difficulties=["easy", "medium"],
        description=(
            "Triage under budget constraints. Mix of legit ads, sophisticated "
            "scams, and gray-area cases. 12 ads but only 30 actions (~2.5 per ad). "
            "Agent must prioritize which ads to investigate deeply."
        ),
    ),
    "task_3": TaskConfig(
        task_id="task_3",
        name="Coordinated Fraud Network Detection",
        difficulty="hard",
        queue_size=20,
        action_budget=40,
        n_legit=6,
        n_fraud=10,
        n_escalate=4,
        include_networks=True,
        n_fraud_rings=3,
        allowed_difficulties=["easy", "medium", "hard"],
        description=(
            "Full challenge including coordinated fraud rings. 20 ads with 3 "
            "hidden fraud networks (clusters of 3-5 ads sharing signals). "
            "Budget of 40 actions (2 per ad). Individual ring ads may look "
            "borderline — the signal is in the connections."
        ),
    ),
}


@dataclass
class Ad:
    ad_id: str
    ad_copy: str
    category: str
    targeting_summary: str
    initial_risk_signals: List[str]
    ground_truth_label: str  # "fraud", "legit", or "escalate"
    fraud_type: str
    severity: float
    difficulty: str


@dataclass
class GeneratedEpisode:
    """All pre-generated data for one episode."""
    task_config: TaskConfig
    ads: List[Ad]
    advertiser_profiles: Dict[str, AdvertiserProfile]
    landing_pages: Dict[str, LandingPageData]
    fraud_rings: List[FraudRing]
    ad_to_rings: Dict[str, List[str]]
    # Pre-generated investigation data keyed by (ad_id, investigation_type)
    investigation_data: Dict[str, Dict[str, str]]


def generate_episode(seed: int, task_id: str = "task_1") -> GeneratedEpisode:
    """Generate a complete episode with all pre-computed investigation data."""
    rng = random.Random(seed)
    config = TASK_CONFIGS[task_id]

    ads = _generate_ad_queue(rng, config)

    fraud_ad_ids = [a.ad_id for a in ads if a.ground_truth_label == "fraud"]

    fraud_rings: List[FraudRing] = []
    ad_to_rings: Dict[str, List[str]] = {}
    ring_shared_payments: Dict[str, str] = {}

    if config.include_networks and config.n_fraud_rings > 0:
        fraud_rings, ad_to_rings = generate_fraud_networks(
            rng, config.n_fraud_rings, fraud_ad_ids
        )
        for ring in fraud_rings:
            if "payment_method" in ring.shared_signals:
                for ad_id in ring.member_ad_ids:
                    ring_shared_payments[ad_id] = ring.shared_signals["payment_method"]

    advertiser_profiles: Dict[str, AdvertiserProfile] = {}
    landing_pages: Dict[str, LandingPageData] = {}
    investigation_data: Dict[str, Dict[str, str]] = {}

    for ad in ads:
        is_fraud = ad.ground_truth_label in ("fraud", "escalate")

        profile = generate_advertiser_profile(
            rng, ad.ad_id, is_fraud,
            payment_method_id=ring_shared_payments.get(ad.ad_id),
        )
        advertiser_profiles[ad.ad_id] = profile

        landing_page_kwargs = {}
        if ad.ad_id in ad_to_rings:
            ring = next(r for r in fraud_rings if ad.ad_id in r.member_ad_ids)
            if "domain_registrar" in ring.shared_signals:
                landing_page_kwargs["registrar_override"] = ring.shared_signals["domain_registrar"]

        lp = generate_landing_page(
            rng, ad.ad_id, is_fraud, ad.fraud_type, **landing_page_kwargs
        )
        landing_pages[ad.ad_id] = lp

        inv = {}
        inv["advertiser_history"] = profile.to_investigation_text()
        inv["landing_page"] = lp.to_investigation_text()
        inv["payment_method"] = _generate_payment_investigation(rng, profile, ad.ad_id, ad_to_rings, fraud_rings)
        inv["targeting_overlap"] = _generate_targeting_investigation(rng, ad, ads, ad_to_rings, fraud_rings)
        inv["creative_similarity"] = _generate_creative_investigation(rng, ad, ads, ad_to_rings, fraud_rings)
        investigation_data[ad.ad_id] = inv

    return GeneratedEpisode(
        task_config=config,
        ads=ads,
        advertiser_profiles=advertiser_profiles,
        landing_pages=landing_pages,
        fraud_rings=fraud_rings,
        ad_to_rings=ad_to_rings,
        investigation_data=investigation_data,
    )


def _generate_ad_queue(rng: random.Random, config: TaskConfig) -> List[Ad]:
    """Build the ad queue by sampling from templates."""
    ads: List[Ad] = []
    ad_counter = 0

    legit_templates = [t for t in LEGIT_TEMPLATES]
    fraud_templates = [
        t for t in FRAUD_TEMPLATES
        if t.difficulty in config.allowed_difficulties and t.label == "fraud"
    ]
    escalate_templates = [
        t for t in FRAUD_TEMPLATES
        if t.difficulty in config.allowed_difficulties and t.label == "escalate"
    ]

    if not escalate_templates:
        escalate_templates = [
            t for t in FRAUD_TEMPLATES if t.label == "escalate"
        ]

    for _ in range(config.n_legit):
        template = rng.choice(legit_templates)
        idx = rng.randint(0, len(template.ad_copies) - 1)
        ad_counter += 1
        ads.append(Ad(
            ad_id=f"ad_{ad_counter:03d}",
            ad_copy=template.ad_copies[idx],
            category=template.category,
            targeting_summary=template.targeting_hints[idx % len(template.targeting_hints)],
            initial_risk_signals=list(template.risk_signals),
            ground_truth_label=template.label,
            fraud_type=template.fraud_type,
            severity=template.severity,
            difficulty=template.difficulty,
        ))

    for _ in range(config.n_fraud):
        if fraud_templates:
            template = rng.choice(fraud_templates)
        else:
            template = rng.choice(FRAUD_TEMPLATES)
        idx = rng.randint(0, len(template.ad_copies) - 1)
        ad_counter += 1
        ads.append(Ad(
            ad_id=f"ad_{ad_counter:03d}",
            ad_copy=template.ad_copies[idx],
            category=template.category,
            targeting_summary=template.targeting_hints[idx % len(template.targeting_hints)],
            initial_risk_signals=list(template.risk_signals),
            ground_truth_label="fraud",
            fraud_type=template.fraud_type,
            severity=template.severity,
            difficulty=template.difficulty,
        ))

    for _ in range(config.n_escalate):
        if escalate_templates:
            template = rng.choice(escalate_templates)
            idx = rng.randint(0, len(template.ad_copies) - 1)
            ad_counter += 1
            ads.append(Ad(
                ad_id=f"ad_{ad_counter:03d}",
                ad_copy=template.ad_copies[idx],
                category=template.category,
                targeting_summary=template.targeting_hints[idx % len(template.targeting_hints)],
                initial_risk_signals=list(template.risk_signals),
                ground_truth_label="escalate",
                fraud_type=template.fraud_type,
                severity=template.severity,
                difficulty=template.difficulty,
            ))

    rng.shuffle(ads)

    renumbered = []
    for i, ad in enumerate(ads):
        ad.ad_id = f"ad_{i + 1:03d}"
        renumbered.append(ad)

    return renumbered


def _generate_payment_investigation(
    rng: random.Random,
    profile: AdvertiserProfile,
    ad_id: str,
    ad_to_rings: Dict[str, List[str]],
    fraud_rings: List[FraudRing],
) -> str:
    """Generate payment method investigation text."""
    lines = [
        f"Payment Method Analysis for {ad_id}:",
        f"  Method type: {profile.payment_method_type}",
        f"  Payment ID: {profile.payment_method_id}",
    ]

    if ad_id in ad_to_rings:
        ring = next(r for r in fraud_rings if ad_id in r.member_ad_ids)
        if "payment_method" in ring.shared_signals:
            other_count = ring.size - 1
            lines.append(f"  WARNING: This payment method is shared with {other_count} other advertiser account(s) in the current queue.")
            lines.append(f"  Shared payment method ID: {ring.shared_signals['payment_method']}")
        else:
            lines.append("  No other accounts in the current queue share this payment method.")
    else:
        if profile.payment_method_type in ("prepaid_card", "crypto", "virtual_card"):
            lines.append(f"  Note: {profile.payment_method_type} payment methods have higher fraud correlation.")
        lines.append("  No other accounts in the current queue share this payment method.")

    if profile.previous_violations > 0:
        lines.append(f"  Payment history: {profile.previous_violations} chargebacks/disputes on record.")

    return "\n".join(lines)


def _generate_targeting_investigation(
    rng: random.Random,
    ad: Ad,
    all_ads: List[Ad],
    ad_to_rings: Dict[str, List[str]],
    fraud_rings: List[FraudRing],
) -> str:
    """Generate targeting overlap investigation text."""
    lines = [
        f"Targeting Overlap Analysis for {ad.ad_id}:",
        f"  Current targeting: {ad.targeting_summary}",
    ]

    if ad.ad_id in ad_to_rings:
        ring = next(r for r in fraud_rings if ad.ad_id in r.member_ad_ids)
        if "targeting_overlap" in ring.shared_signals:
            other_ids = [mid for mid in ring.member_ad_ids if mid != ad.ad_id]
            lines.append(f"  HIGH OVERLAP detected with: {', '.join(other_ids)}")
            lines.append(f"  Shared targeting pattern: {ring.shared_signals['targeting_overlap']}")
            lines.append("  This level of targeting similarity across different advertiser accounts is unusual.")
        else:
            lines.append("  No significant targeting overlap with other ads in the queue.")
    else:
        similar = [
            a for a in all_ads
            if a.ad_id != ad.ad_id and a.category == ad.category
        ]
        if similar:
            lines.append(f"  {len(similar)} other ad(s) in the same category ({ad.category}) found in queue.")
            lines.append("  Targeting overlap is within normal range for the category.")
        else:
            lines.append("  No significant targeting overlap with other ads in the queue.")

    return "\n".join(lines)


def _generate_creative_investigation(
    rng: random.Random,
    ad: Ad,
    all_ads: List[Ad],
    ad_to_rings: Dict[str, List[str]],
    fraud_rings: List[FraudRing],
) -> str:
    """Generate creative similarity investigation text."""
    lines = [
        f"Creative Similarity Analysis for {ad.ad_id}:",
    ]

    if ad.ad_id in ad_to_rings:
        ring = next(r for r in fraud_rings if ad.ad_id in r.member_ad_ids)
        if "creative_template" in ring.shared_signals:
            other_ids = [mid for mid in ring.member_ad_ids if mid != ad.ad_id]
            lines.append(f"  STRONG SIMILARITY detected with: {', '.join(other_ids)}")
            lines.append(f"  Shared template hash: {ring.shared_signals['creative_template']}")
            lines.append("  Ads appear to use the same creative template with minor text variations.")
            lines.append("  Image assets share identical dimensions and color palettes.")
        else:
            lines.append("  No significant creative similarity with other ads in the queue.")
    else:
        lines.append("  No significant creative similarity with other ads in the queue.")

    if ad.ground_truth_label == "fraud" and ad.fraud_type:
        similarity = rng.uniform(0.3, 0.7)
        lines.append(f"  Similarity to known scam templates: {similarity:.0%}")
    else:
        similarity = rng.uniform(0.0, 0.15)
        lines.append(f"  Similarity to known scam templates: {similarity:.0%}")

    return "\n".join(lines)
