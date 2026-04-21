"""Synthetic data generation for the Ad Fraud Investigation Environment."""

from .ad_generator import (
    TASK_CONFIGS,
    Ad,
    CampaignProfile,
    GeneratedEpisode,
    generate_episode,
    generate_proposal_data,
)
from .advertiser_profiles import generate_advertiser_profile
from .episode_loader import (
    CallableEpisodeLoader,
    EpisodeLoader,
    SyntheticEpisodeLoader,
    extend_episode_with_proposal,
)
from .fraud_patterns import FRAUD_TEMPLATES, LEGIT_TEMPLATES
from .landing_pages import generate_landing_page
from .network_generator import generate_fraud_networks
from .tool_registry import (
    INVESTIGATION_TARGETS,
    InvestigationToolRegistry,
)

__all__ = [
    "Ad",
    "CallableEpisodeLoader",
    "CampaignProfile",
    "EpisodeLoader",
    "FRAUD_TEMPLATES",
    "GeneratedEpisode",
    "INVESTIGATION_TARGETS",
    "InvestigationToolRegistry",
    "LEGIT_TEMPLATES",
    "SyntheticEpisodeLoader",
    "TASK_CONFIGS",
    "extend_episode_with_proposal",
    "generate_advertiser_profile",
    "generate_episode",
    "generate_fraud_networks",
    "generate_landing_page",
    "generate_proposal_data",
]
