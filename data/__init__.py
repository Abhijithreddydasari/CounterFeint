"""Synthetic data generation for the Ad Fraud Investigation Environment."""

from .ad_generator import TASK_CONFIGS, CampaignProfile, generate_episode
from .advertiser_profiles import generate_advertiser_profile
from .fraud_patterns import FRAUD_TEMPLATES, LEGIT_TEMPLATES
from .landing_pages import generate_landing_page
from .network_generator import generate_fraud_networks

__all__ = [
    "generate_episode",
    "generate_advertiser_profile",
    "generate_landing_page",
    "generate_fraud_networks",
    "CampaignProfile",
    "FRAUD_TEMPLATES",
    "LEGIT_TEMPLATES",
    "TASK_CONFIGS",
]
