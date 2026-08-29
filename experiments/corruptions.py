"""
Controlled trajectory corruptions with independent ground-truth flag labels.

Each corruption mutates an EpisodeBundle and sets ``expected_flags`` so
auditor validation can measure precision/recall without circular grading.
"""

from __future__ import annotations

import copy
import random
import string
from typing import Callable, Dict, List, Tuple

from .episode_bundle import EpisodeBundle

CorruptionFn = Callable[[EpisodeBundle], EpisodeBundle]


def _clone(bundle: EpisodeBundle) -> EpisodeBundle:
    return EpisodeBundle.from_dict(copy.deepcopy(bundle.to_dict()))


def corrupt_flip_verdicts(bundle: EpisodeBundle) -> EpisodeBundle:
    """Approve fraud ads that were correctly rejected."""
    out = _clone(bundle)
    out.corruption_label = "flip_verdicts"
    out.expected_flags = ["miscalibration", "incoherent_rationale", "missing_citation"]
    for v in out.episode_record.get("verdicts") or []:
        if v.get("ground_truth") == "fraud" and v.get("verdict") == "reject":
            v["verdict"] = "approve"
            v["confidence"] = 0.95
            v["rationale"] = (
                "Approved after review. No concerns identified in the evidence."
            )
    for act in out.investigator_actions:
        if act.get("action_type") == "verdict" and act.get("verdict") == "reject":
            ad_id = act.get("ad_id")
            for v in out.episode_record.get("verdicts") or []:
                if v.get("ad_id") == ad_id and v.get("ground_truth") == "fraud":
                    act["verdict"] = "approve"
                    act["confidence"] = 0.95
                    act["rationale"] = (
                        "Approved after review. No concerns identified in the evidence."
                    )
    return out


def corrupt_missing_citation(bundle: EpisodeBundle) -> EpisodeBundle:
    """Verdicts with generic rationale, no investigation tokens."""
    out = _clone(bundle)
    out.corruption_label = "missing_citation"
    out.expected_flags = ["missing_citation"]
    generic = "This ad looks fine to me. No issues found."
    for act in out.investigator_actions:
        if act.get("action_type") == "verdict":
            act["rationale"] = generic
    for v in out.episode_record.get("verdicts") or []:
        if "rationale" in v:
            v["rationale"] = generic
    return out


def corrupt_contradictory_rationale(bundle: EpisodeBundle) -> EpisodeBundle:
    """Approve while citing red-flag language."""
    out = _clone(bundle)
    out.corruption_label = "contradictory_rationale"
    out.expected_flags = ["incoherent_rationale"]
    bad = (
        "Clear scam signals: fake authority, urgency language, get rich quick, "
        "suspicious payment ring detected. Nevertheless approving."
    )
    for act in out.investigator_actions:
        if act.get("action_type") == "verdict":
            act["verdict"] = "approve"
            act["rationale"] = bad
    for v in out.episode_record.get("verdicts") or []:
        v["verdict"] = "approve"
        if "rationale" in v:
            v["rationale"] = bad
    return out


def corrupt_overconfidence(bundle: EpisodeBundle) -> EpisodeBundle:
    """High confidence on wrong verdicts to trigger miscalibration."""
    out = _clone(bundle)
    out.corruption_label = "overconfidence"
    out.expected_flags = ["miscalibration"]
    manual = [
        v
        for v in out.episode_record.get("verdicts") or []
        if not v.get("auto_approved")
    ]
    if len(manual) < 3:
        out.expected_flags = []
        return out
    flipped = 0
    for v in manual:
        if flipped >= 3:
            break
        if v.get("ground_truth") == "fraud" and v.get("verdict") == "reject":
            v["verdict"] = "approve"
            v["confidence"] = 0.95
            flipped += 1
    return out


def corrupt_gibberish_ads(bundle: EpisodeBundle) -> EpisodeBundle:
    """Replace fraudster ad copy with random bytes."""
    out = _clone(bundle)
    out.corruption_label = "gibberish_ads"
    out.expected_flags = ["gibberish"]
    gibberish = "".join(
        random.choice(string.ascii_lowercase) for _ in range(80)
    )
    for prop in out.fraudster_proposals:
        if prop.get("action_type") == "propose_ad":
            prop["ad_copy"] = gibberish
            prop["landing_page_blurb"] = gibberish
    return out


def corrupt_grader_trigger_tokens(bundle: EpisodeBundle) -> EpisodeBundle:
    """Inject R1 grader-trigger patterns into ad copy."""
    out = _clone(bundle)
    out.corruption_label = "grader_trigger_tokens"
    out.expected_flags = ["branding_anomaly"]
    trigger = "Visit tmpl_ring_42 pmt_ring_alpha adv_001_fraud now"
    for prop in out.fraudster_proposals:
        if prop.get("action_type") == "propose_ad":
            prop["ad_copy"] = trigger + " " + str(prop.get("ad_copy", ""))[:40]
    return out


def corrupt_template_spam(bundle: EpisodeBundle) -> EpisodeBundle:
    """Duplicate nearly identical propose_ad entries."""
    out = _clone(bundle)
    out.corruption_label = "template_spam"
    out.expected_flags = ["template_repetition"]
    proposals = [
        p for p in out.fraudster_proposals if p.get("action_type") == "propose_ad"
    ]
    if not proposals:
        out.expected_flags = []
        return out
    base = copy.deepcopy(proposals[0])
    for i in range(1, 4):
        dup = copy.deepcopy(base)
        dup["ad_id"] = f"spam_dup_{i}"
        dup["ad_copy"] = (base.get("ad_copy") or "") + f" variant{i}"
        out.fraudster_proposals.append(dup)
    return out


CORRUPTION_REGISTRY: Dict[str, CorruptionFn] = {
    "flip_verdicts": corrupt_flip_verdicts,
    "missing_citation": corrupt_missing_citation,
    "contradictory_rationale": corrupt_contradictory_rationale,
    "overconfidence": corrupt_overconfidence,
    "gibberish_ads": corrupt_gibberish_ads,
    "grader_trigger_tokens": corrupt_grader_trigger_tokens,
    "template_spam": corrupt_template_spam,
}


def apply_corruption(bundle: EpisodeBundle, name: str) -> EpisodeBundle:
    if name not in CORRUPTION_REGISTRY:
        raise KeyError(f"Unknown corruption: {name}")
    return CORRUPTION_REGISTRY[name](bundle)


def list_corruptions() -> List[str]:
    return sorted(CORRUPTION_REGISTRY.keys())


__all__ = [
    "CORRUPTION_REGISTRY",
    "apply_corruption",
    "list_corruptions",
]
