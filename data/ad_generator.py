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

# Decoy pools: values that can appear in both legit and fraud ads,
# making naive pattern-matching unreliable.
_DECOY_REGISTRARS = ["NameSilo", "Cloudflare Registrar", "GoDaddy", "Tucows (privacy proxy)"]
_DECOY_PAYMENT_TYPES = ["credit_card", "prepaid_card", "corporate_card"]
_COMMON_TARGETING_SEGMENTS = [
    "Adults 25-54, interests: shopping, lifestyle",
    "Adults 18-45, interests: technology, gadgets",
    "Adults 30-55, interests: finance, investing",
]


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
        action_budget=35,
        n_legit=6,
        n_fraud=10,
        n_escalate=4,
        include_networks=True,
        n_fraud_rings=3,
        allowed_difficulties=["easy", "medium", "hard"],
        description=(
            "Full challenge including coordinated fraud rings. 20 ads with 3 "
            "hidden fraud networks using varied topologies (cliques, chains, "
            "hub-and-spoke). Budget of 35 actions (~1.75 per ad). Ring member "
            "ads look borderline individually — the agent must cross-reference "
            "investigation data across ads to detect shared signals."
        ),
    ),
}


@dataclass
class CampaignProfile:
    """Campaign-level metadata associated with an ad."""
    objective: str          # e.g. "conversions", "traffic", "awareness", "app_installs"
    bid_strategy: str       # e.g. "lowest_cost", "cost_cap", "bid_cap"
    daily_budget_usd: float
    ad_set_count: int
    placements: List[str]

    def to_investigation_text(self, account_age_days: int) -> str:
        budget_age_ratio = (
            self.daily_budget_usd / max(account_age_days, 1)
        )
        placements_str = ", ".join(self.placements)

        lines = [
            f"Campaign Objective: {self.objective}",
            f"Bid Strategy: {self.bid_strategy}",
            f"Daily Budget: ${self.daily_budget_usd:,.2f} "
            f"(account is {account_age_days} days old — "
            f"budget/age ratio: ${budget_age_ratio:,.2f}/day)",
            f"Active Ad Sets: {self.ad_set_count}",
            f"Placements: {placements_str}",
        ]

        warnings = []
        if budget_age_ratio > 50:
            warnings.append(
                "Budget-to-account-age ratio exceeds typical thresholds."
            )
        if self.ad_set_count > 15:
            warnings.append(
                f"High ad set count ({self.ad_set_count}) — "
                "possible policy evasion testing via creative variation."
            )
        if self.objective in ("traffic", "awareness") and self.bid_strategy == "lowest_cost":
            warnings.append(
                f"Optimizing for {self.objective} with lowest-cost bidding "
                "— common in spray-and-pray fraud campaigns."
            )
        if "Audience Network" in self.placements and len(self.placements) <= 2:
            warnings.append(
                "Heavy reliance on Audience Network placement — "
                "higher bot traffic exposure."
            )

        if warnings:
            for w in warnings:
                lines.append(f"  WARNING: {w}")
        else:
            lines.append("Budget and pacing consistent with historical account behavior.")

        return "\n".join(lines)


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
    campaign_profiles: Dict[str, CampaignProfile]
    landing_pages: Dict[str, LandingPageData]
    fraud_rings: List[FraudRing]
    ad_to_rings: Dict[str, List[str]]
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
    campaign_profiles: Dict[str, CampaignProfile] = {}
    landing_pages: Dict[str, LandingPageData] = {}
    investigation_data: Dict[str, Dict[str, str]] = {}

    ring_campaign_overrides: Dict[str, Dict[str, Any]] = {}
    ring_created_dates: Dict[str, str] = {}
    for ring in fraud_rings:
        shared_objective = rng.choice(["traffic", "awareness"])
        shared_bid = "lowest_cost"
        # Ring members share account creation dates within the same week
        from datetime import date, timedelta
        base_date = date(2026, 4, 6) - timedelta(days=rng.randint(5, 45))
        for ad_id in ring.member_ad_ids:
            ring_campaign_overrides[ad_id] = {
                "objective": shared_objective,
                "bid_strategy": shared_bid,
            }
            offset = timedelta(days=rng.randint(0, 6))
            ring_created_dates[ad_id] = (base_date + offset).isoformat()

    for ad in ads:
        is_fraud = ad.ground_truth_label in ("fraud", "escalate")

        profile = generate_advertiser_profile(
            rng, ad.ad_id, is_fraud,
            payment_method_id=ring_shared_payments.get(ad.ad_id),
            ring_created_date=ring_created_dates.get(ad.ad_id),
        )
        advertiser_profiles[ad.ad_id] = profile

        campaign = _generate_campaign_profile(
            rng, ad, is_fraud,
            ring_overrides=ring_campaign_overrides.get(ad.ad_id),
        )
        campaign_profiles[ad.ad_id] = campaign

        landing_page_kwargs = {}
        if ad.ad_id in ad_to_rings:
            ring = next(r for r in fraud_rings if ad.ad_id in r.member_ad_ids)
            if "domain_registrar" in ring.shared_signals:
                landing_page_kwargs["registrar_override"] = ring.shared_signals["domain_registrar"]
        elif not is_fraud and rng.random() < 0.25:
            landing_page_kwargs["registrar_override"] = rng.choice(_DECOY_REGISTRARS)

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
        inv["campaign_structure"] = _generate_campaign_investigation(
            rng, ad, campaign, profile, ad_to_rings, fraud_rings,
        )
        investigation_data[ad.ad_id] = inv

    return GeneratedEpisode(
        task_config=config,
        ads=ads,
        advertiser_profiles=advertiser_profiles,
        campaign_profiles=campaign_profiles,
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
    """Generate payment method investigation text.

    Ring signals are embedded as raw data values (shared payment IDs) without
    explicitly naming other ads. The agent must cross-reference across ads.
    """
    lines = [
        f"Payment Method Analysis for {ad_id}:",
        f"  Method type: {profile.payment_method_type}",
        f"  Payment ID: {profile.payment_method_id}",
    ]

    if profile.payment_method_type in ("prepaid_card", "crypto", "virtual_card"):
        lines.append(f"  Note: {profile.payment_method_type} payments have elevated fraud correlation in platform data.")

    if profile.previous_violations > 0:
        lines.append(f"  Chargeback/dispute history: {profile.previous_violations} incident(s) on record.")
    else:
        lines.append("  Chargeback/dispute history: Clean record.")

    velocity = rng.randint(1, 5) if ad_id not in ad_to_rings else rng.randint(3, 12)
    lines.append(f"  Payment method added to {velocity} advertiser account(s) in the last 90 days.")

    if profile.account_age_days < 30:
        lines.append(f"  First charge on this method: {profile.account_age_days} days ago.")

    return "\n".join(lines)


def _generate_targeting_investigation(
    rng: random.Random,
    ad: Ad,
    all_ads: List[Ad],
    ad_to_rings: Dict[str, List[str]],
    fraud_rings: List[FraudRing],
) -> str:
    """Generate targeting overlap investigation text.

    Ring members share an exact targeting fingerprint, presented as raw data.
    The agent must compare fingerprints across ads to detect collusion.
    """
    lines = [
        f"Targeting Analysis for {ad.ad_id}:",
        f"  Declared targeting: {ad.targeting_summary}",
    ]

    if ad.ad_id in ad_to_rings:
        ring = next(r for r in fraud_rings if ad.ad_id in r.member_ad_ids)
        if "targeting_overlap" in ring.shared_signals:
            lines.append(f"  Targeting fingerprint: {ring.shared_signals['targeting_overlap']}")
            overlap_pct = rng.randint(85, 98)
            lines.append(f"  Audience overlap with platform average for category: {overlap_pct}%")
        else:
            fingerprint = f"seg_{rng.randint(10000, 99999)}"
            lines.append(f"  Targeting fingerprint: {fingerprint}")
            overlap_pct = rng.randint(20, 55)
            lines.append(f"  Audience overlap with platform average for category: {overlap_pct}%")
    else:
        fingerprint = f"seg_{rng.randint(10000, 99999)}"
        lines.append(f"  Targeting fingerprint: {fingerprint}")
        similar = [a for a in all_ads if a.ad_id != ad.ad_id and a.category == ad.category]
        if similar:
            overlap_pct = rng.randint(30, 65)
            lines.append(f"  {len(similar)} other ad(s) in same category ({ad.category}) in queue.")
            lines.append(f"  Audience overlap with platform average for category: {overlap_pct}%")
        else:
            overlap_pct = rng.randint(10, 40)
            lines.append(f"  Audience overlap with platform average for category: {overlap_pct}%")

    geo_regions = rng.randint(1, 8) if ad.ground_truth_label != "legit" else rng.randint(1, 3)
    lines.append(f"  Geographic regions targeted: {geo_regions}")

    return "\n".join(lines)


def _generate_creative_investigation(
    rng: random.Random,
    ad: Ad,
    all_ads: List[Ad],
    ad_to_rings: Dict[str, List[str]],
    fraud_rings: List[FraudRing],
) -> str:
    """Generate creative similarity investigation text.

    Ring members share a template hash, presented as raw data. The agent
    must compare hashes across ads to detect reuse.
    """
    lines = [
        f"Creative Analysis for {ad.ad_id}:",
    ]

    if ad.ad_id in ad_to_rings:
        ring = next(r for r in fraud_rings if ad.ad_id in r.member_ad_ids)
        if "creative_template" in ring.shared_signals:
            lines.append(f"  Template hash: {ring.shared_signals['creative_template']}")
            lines.append(f"  Image dimensions: {rng.choice(['1200x628', '1080x1080', '1200x1200'])} px")
            lines.append(f"  Color palette hash: pal_{rng.randint(100, 999)}")
            lines.append(f"  Text-to-image ratio: {rng.randint(18, 25)}%")
        else:
            lines.append(f"  Template hash: tmpl_{rng.randint(10000, 99999)}")
            lines.append(f"  Image dimensions: {rng.choice(['1200x628', '1080x1080', '1200x1200', '600x600'])} px")
            lines.append(f"  Text-to-image ratio: {rng.randint(10, 30)}%")
    else:
        lines.append(f"  Template hash: tmpl_{rng.randint(10000, 99999)}")
        lines.append(f"  Image dimensions: {rng.choice(['1200x628', '1080x1080'])} px")
        lines.append(f"  Text-to-image ratio: {rng.randint(8, 22)}%")

    if ad.ground_truth_label == "fraud" and ad.fraud_type:
        similarity = rng.uniform(0.3, 0.7)
    else:
        similarity = rng.uniform(0.0, 0.15)
    lines.append(f"  Similarity to known scam templates: {similarity:.0%}")

    return "\n".join(lines)


_LEGIT_OBJECTIVES = ["conversions", "leads", "sales", "app_installs"]
_FRAUD_OBJECTIVES = ["traffic", "awareness", "reach", "engagement"]
_LEGIT_BID_STRATEGIES = ["cost_cap", "bid_cap", "target_cost"]
_FRAUD_BID_STRATEGIES = ["lowest_cost", "lowest_cost", "lowest_cost", "cost_cap"]

_LEGIT_PLACEMENTS = [
    ["Facebook Feed", "Instagram Feed"],
    ["Facebook Feed", "Instagram Feed", "Instagram Stories"],
    ["Facebook Feed"],
    ["Facebook Feed", "Instagram Feed", "Instagram Reels"],
]
_FRAUD_PLACEMENTS = [
    ["Audience Network", "Facebook Feed"],
    ["Audience Network", "Facebook Feed", "Instagram Stories"],
    ["Facebook Feed", "Instagram Feed", "Audience Network", "Messenger"],
    ["Audience Network"],
]


def _generate_campaign_profile(
    rng: random.Random,
    ad: Ad,
    is_fraud: bool,
    *,
    ring_overrides: Optional[Dict[str, Any]] = None,
) -> CampaignProfile:
    """Generate campaign-level metadata for an ad."""
    if is_fraud:
        objective = rng.choice(_FRAUD_OBJECTIVES)
        bid_strategy = rng.choice(_FRAUD_BID_STRATEGIES)
        daily_budget = round(rng.uniform(500, 5000), 2)
        ad_set_count = rng.randint(8, 50)
        placements = rng.choice(_FRAUD_PLACEMENTS)
    else:
        objective = rng.choice(_LEGIT_OBJECTIVES)
        bid_strategy = rng.choice(_LEGIT_BID_STRATEGIES)
        daily_budget = round(rng.uniform(20, 500), 2)
        ad_set_count = rng.randint(1, 5)
        placements = rng.choice(_LEGIT_PLACEMENTS)

    if ring_overrides:
        objective = ring_overrides.get("objective", objective)
        bid_strategy = ring_overrides.get("bid_strategy", bid_strategy)

    return CampaignProfile(
        objective=objective,
        bid_strategy=bid_strategy,
        daily_budget_usd=daily_budget,
        ad_set_count=ad_set_count,
        placements=list(placements),
    )


def _generate_campaign_investigation(
    rng: random.Random,
    ad: Ad,
    campaign: CampaignProfile,
    profile: AdvertiserProfile,
    ad_to_rings: Dict[str, List[str]],
    fraud_rings: List[FraudRing],
) -> str:
    """Generate campaign structure investigation text.

    Ring members share campaign configurations but no explicit cross-references.
    The agent must compare objective/bid/budget patterns across ads.
    """
    lines = [
        f"Campaign Structure Analysis for {ad.ad_id}:",
        campaign.to_investigation_text(profile.account_age_days),
    ]

    config_hash = f"cfg_{hash((campaign.objective, campaign.bid_strategy)) & 0xFFFF:04x}"
    lines.append(f"  Campaign configuration fingerprint: {config_hash}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fraudster-proposal extension (Round 2)
# ---------------------------------------------------------------------------


def _category_to_fraud_template(category: str) -> AdTemplate:
    """Pick the closest matching FRAUD_TEMPLATE for a Fraudster-declared category."""
    for tmpl in FRAUD_TEMPLATES:
        if tmpl.category == category:
            return tmpl
    return FRAUD_TEMPLATES[0]


def generate_proposal_data(
    *,
    rng: random.Random,
    ad_id: str,
    ad_copy: str,
    category: str,
    landing_page_blurb: Optional[str] = None,
    targeting_summary: Optional[str] = None,
    existing_ads: Optional[List[Ad]] = None,
) -> Tuple[Ad, Dict[str, str], AdvertiserProfile, CampaignProfile, "LandingPageData"]:
    """
    Build a fully-formed Ad + investigation_data for a Fraudster-proposed ad.

    The Fraudster controls the *surface*: ad_copy, category, landing page blurb,
    targeting summary.  Underlying account / payment / campaign signals are
    sampled from the fraud-mode distribution so the Investigator has a real
    detection task.

    Returns
    -------
    ad
        The Ad object (ground_truth_label="fraud").
    investigation_data
        Dict[str, str] keyed by investigation target name (the 6 canonical
        targets), already rendered to text.
    profile, campaign, landing_page
        The auxiliary data structures, returned in case the caller wants to
        register them on a GeneratedEpisode.
    """
    template = _category_to_fraud_template(category)

    ad = Ad(
        ad_id=ad_id,
        ad_copy=ad_copy.strip()[:2000] if ad_copy else template.ad_copies[0],
        category=category,
        targeting_summary=(
            targeting_summary.strip()[:512]
            if targeting_summary
            else template.targeting_hints[0]
        ),
        initial_risk_signals=list(template.risk_signals),
        ground_truth_label="fraud",
        fraud_type=template.fraud_type or "fraudster_proposal",
        severity=template.severity if template.severity > 0 else 0.6,
        difficulty=template.difficulty,
    )

    profile = generate_advertiser_profile(rng, ad_id, is_fraud=True)
    campaign = _generate_campaign_profile(rng, ad, is_fraud=True)
    landing_page = generate_landing_page(rng, ad_id, is_fraud=True, fraud_type=ad.fraud_type)

    if landing_page_blurb:
        from dataclasses import replace
        landing_page = replace(
            landing_page,
            content_summary=landing_page_blurb.strip()[:2000],
        )

    siblings = list(existing_ads or [])
    siblings.append(ad)

    investigation_data: Dict[str, str] = {
        "advertiser_history": profile.to_investigation_text(),
        "landing_page": landing_page.to_investigation_text(),
        "payment_method": _generate_payment_investigation(
            rng, profile, ad_id, ad_to_rings={}, fraud_rings=[]
        ),
        "targeting_overlap": _generate_targeting_investigation(
            rng, ad, siblings, ad_to_rings={}, fraud_rings=[]
        ),
        "creative_similarity": _generate_creative_investigation(
            rng, ad, siblings, ad_to_rings={}, fraud_rings=[]
        ),
        "campaign_structure": _generate_campaign_investigation(
            rng, ad, campaign, profile, ad_to_rings={}, fraud_rings=[]
        ),
    }

    return ad, investigation_data, profile, campaign, landing_page
