"""
Unit tests for :mod:`counterfeint.agents`.

No live LLM is called — we inject a fake OpenAI-compatible client that returns
pre-canned responses (or raises canned exceptions) so every branch of the
retry / fallback state machine is exercised deterministically.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from counterfeint.agents import LLMFraudster, LLMInvestigator
from counterfeint.agents.base import LLMPolicyBase
from counterfeint.models import AdReviewAction, FraudsterAction
from counterfeint.scripted._base import PolicyBase


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeClient:
    """Minimal ``openai.OpenAI``-compatible surface: ``.chat.completions.create``.

    Each call pops the next response (either a string to return as the
    message content, or an ``Exception`` instance to raise).
    """

    def __init__(self, script: List[Any]):
        self._script = list(script)
        self.call_count = 0
        self.last_kwargs: Optional[Dict[str, Any]] = None

        outer = self

        class _Completions:
            def create(self_inner, **kwargs):  # noqa: N805
                outer.call_count += 1
                outer.last_kwargs = kwargs
                if not outer._script:
                    raise RuntimeError("no more scripted responses")
                item = outer._script.pop(0)
                if isinstance(item, Exception):
                    raise item
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=item)
                        )
                    ]
                )

        self.chat = SimpleNamespace(completions=_Completions())


class _SentinelFallback(PolicyBase):
    """Fallback policy that records every call without doing any real logic."""

    def __init__(self, kind: str = "fraudster") -> None:
        self.kind = kind
        self.calls: List[Dict[str, Any]] = []
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, observation: Dict[str, Any]):
        self.calls.append(observation)
        if self.kind == "fraudster":
            return FraudsterAction(
                action_type="end_turn",
                rationale="sentinel fallback",
            )
        return AdReviewAction(
            action_type="verdict",
            ad_id="ad_000",
            verdict="escalate",
            confidence=0.3,
            rationale="sentinel fallback",
        )


# ---------------------------------------------------------------------------
# Observation fixtures
# ---------------------------------------------------------------------------


def _fraudster_obs() -> Dict[str, Any]:
    return {
        "feedback": "OK",
        "phase": "fraudster_turn",
        "round_number": 1,
        "rounds_remaining": 3,
        "proposals_used": 0,
        "proposals_remaining": 5,
        "actions_left_this_turn": 3,
        "current_queue": [
            {"ad_id": "ad_001", "category": "ecommerce", "status": "pending"},
        ],
        "prior_verdicts": [],
        "investigation_targets_used": {},
        "allowed_categories": ["ecommerce", "fake_giveaway"],
    }


def _investigator_obs() -> Dict[str, Any]:
    return {
        "feedback": "start of episode",
        "queue_summary": "5 ads pending",
        "current_ad_info": (
            "=== Ad in Focus: ad_001 ===\n"
            "Category: fake_giveaway\n"
            "Meta policy lens: FSDP-IF-03 — Fraud > Fake Giveaways\n"
            "Ad copy: \"Free iPhone\"\n"
        ),
        "investigation_findings": "",
        "verdict_history_summary": "",
        "available_ads": ["ad_001", "ad_002"],
        "queue_status": {
            "task_id": "task_1",
            "steps_remaining": 25,
            "investigation_budget": 25,
            "reviewed": 0,
            "step": 0,
        },
        "queue_may_grow": False,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidResponses:
    def test_fraudster_parses_clean_json(self) -> None:
        payload = {
            "action_type": "propose_ad",
            "ad_copy": "Trial our SaaS free 14 days",
            "category": "ecommerce",
            "targeting_summary": "SMB owners, US",
        }
        fake = _FakeClient([json.dumps(payload)])
        fallback = _SentinelFallback("fraudster")
        policy = LLMFraudster(fallback_policy=fallback, client=fake, retries=0)

        action = policy.act(_fraudster_obs())

        assert isinstance(action, FraudsterAction)
        assert action.action_type == "propose_ad"
        assert action.category == "ecommerce"
        assert policy.fallback_count == 0
        assert fallback.calls == []

    def test_investigator_parses_clean_json(self) -> None:
        payload = {
            "action_type": "investigate",
            "ad_id": "ad_001",
            "investigation_target": "landing_page",
            "rationale": "check landing copy",
        }
        fake = _FakeClient([json.dumps(payload)])
        fallback = _SentinelFallback("investigator")
        policy = LLMInvestigator(fallback_policy=fallback, client=fake, retries=0)

        action = policy.act(_investigator_obs())

        assert isinstance(action, AdReviewAction)
        assert action.action_type == "investigate"
        assert action.ad_id == "ad_001"
        assert policy.fallback_count == 0

    def test_fraudster_strips_markdown_code_fences(self) -> None:
        payload = (
            "```json\n"
            + json.dumps(
                {
                    "action_type": "end_turn",
                    "rationale": "no more to propose",
                }
            )
            + "\n```"
        )
        fake = _FakeClient([payload])
        fallback = _SentinelFallback("fraudster")
        policy = LLMFraudster(fallback_policy=fallback, client=fake, retries=0)

        action = policy.act(_fraudster_obs())
        assert action.action_type == "end_turn"
        assert policy.fallback_count == 0


# ---------------------------------------------------------------------------
# Failure modes → fallback
# ---------------------------------------------------------------------------


class _FakeTimeout(Exception):
    """Stand-in for openai.APITimeoutError matched by class name."""

    pass


_FakeTimeout.__name__ = "APITimeoutError"


class _FakeApiError(Exception):
    pass


_FakeApiError.__name__ = "APIError"


class TestFailureFallback:
    def test_json_decode_error_falls_back(self) -> None:
        fake = _FakeClient(["this is not json, sorry"])
        fallback = _SentinelFallback("fraudster")
        policy = LLMFraudster(fallback_policy=fallback, client=fake, retries=0)

        action = policy.act(_fraudster_obs())
        assert action.action_type == "end_turn"
        assert action.rationale == "sentinel fallback"
        assert policy.fallback_count == 1
        assert len(fallback.calls) == 1
        assert "invalid JSON" in (policy.last_error or "")

    def test_timeout_retried_then_fallback(self) -> None:
        timeout = _FakeTimeout("boom")
        fake = _FakeClient([timeout, timeout, timeout])
        fallback = _SentinelFallback("fraudster")
        policy = LLMFraudster(fallback_policy=fallback, client=fake, retries=2)

        action = policy.act(_fraudster_obs())

        # 1 initial + 2 retries = 3 attempts, all raising.
        assert fake.call_count == 3
        assert policy.fallback_count == 1
        assert action.rationale == "sentinel fallback"

    def test_validation_error_on_unknown_action_type(self) -> None:
        payload = json.dumps({"action_type": "teleport", "ad_id": "ad_001"})
        fake = _FakeClient([payload])
        fallback = _SentinelFallback("investigator")
        policy = LLMInvestigator(fallback_policy=fallback, client=fake, retries=0)

        action = policy.act(_investigator_obs())
        assert action.action_type == "verdict"  # sentinel fallback
        assert policy.fallback_count == 1
        assert "schema" in (policy.last_error or "")

    def test_validation_error_on_missing_required_field(self) -> None:
        # propose_ad requires category + ad_copy; action_type only is invalid.
        payload = json.dumps({"action_type": "foobar"})
        fake = _FakeClient([payload])
        fallback = _SentinelFallback("fraudster")
        policy = LLMFraudster(fallback_policy=fallback, client=fake, retries=0)

        action = policy.act(_fraudster_obs())
        assert action.action_type == "end_turn"  # sentinel
        assert policy.fallback_count == 1

    def test_empty_response_falls_back(self) -> None:
        fake = _FakeClient([""])
        fallback = _SentinelFallback("fraudster")
        policy = LLMFraudster(fallback_policy=fallback, client=fake, retries=0)

        action = policy.act(_fraudster_obs())
        assert action.action_type == "end_turn"
        assert policy.fallback_count == 1

    def test_generic_api_error_is_not_retried(self) -> None:
        err = _FakeApiError("server returned 500")
        fake = _FakeClient([err, err])
        fallback = _SentinelFallback("fraudster")
        policy = LLMFraudster(fallback_policy=fallback, client=fake, retries=3)

        action = policy.act(_fraudster_obs())

        # Non-retryable class name -> stops after first call, not all 4.
        assert fake.call_count == 1
        assert policy.fallback_count == 1
        assert action.rationale == "sentinel fallback"


class TestFallbackCountAccumulation:
    def test_fallback_count_increments_across_calls(self) -> None:
        fake = _FakeClient(
            [
                "garbage",
                json.dumps(
                    {
                        "action_type": "end_turn",
                        "rationale": "good reply",
                    }
                ),
                "still garbage",
            ]
        )
        fallback = _SentinelFallback("fraudster")
        policy = LLMFraudster(fallback_policy=fallback, client=fake, retries=0)

        a1 = policy.act(_fraudster_obs())
        a2 = policy.act(_fraudster_obs())
        a3 = policy.act(_fraudster_obs())

        # 1st call: garbage -> fallback, 2nd: clean json, 3rd: garbage -> fallback.
        assert policy.fallback_count == 2
        assert policy.call_count == 3
        assert a1.rationale == "sentinel fallback"
        assert a2.rationale == "good reply"
        assert a3.rationale == "sentinel fallback"

    def test_reset_zeroes_counters_and_forwards_to_fallback(self) -> None:
        fake = _FakeClient(["not json", "also not json"])
        fallback = _SentinelFallback("fraudster")
        policy = LLMFraudster(fallback_policy=fallback, client=fake, retries=0)

        policy.act(_fraudster_obs())
        policy.act(_fraudster_obs())
        assert policy.fallback_count == 2
        assert policy.call_count == 2

        policy.reset()
        assert policy.fallback_count == 0
        assert policy.call_count == 0
        assert fallback.reset_calls == 1


# ---------------------------------------------------------------------------
# Construction / invariants
# ---------------------------------------------------------------------------


class TestConstructionInvariants:
    def test_missing_system_prompt_raises(self) -> None:
        class _Broken(LLMPolicyBase):
            # deliberately missing both system_prompt and action_model
            _log_name = "broken"

        with pytest.raises(TypeError):
            _Broken(fallback_policy=_SentinelFallback())

    def test_client_is_exposed_for_test_injection(self) -> None:
        fake = _FakeClient([])
        policy = LLMFraudster(
            fallback_policy=_SentinelFallback("fraudster"),
            client=fake,
            retries=0,
        )
        assert policy.client is fake

    def test_fraudster_user_prompt_contains_observation_slots(self) -> None:
        policy = LLMFraudster(
            fallback_policy=_SentinelFallback("fraudster"),
            client=_FakeClient([]),
            retries=0,
        )
        text = policy._build_user_prompt(_fraudster_obs())
        assert "proposals_remaining     = 5" in text
        assert "ecommerce" in text
        assert "fake_giveaway" in text

    def test_investigator_user_prompt_includes_meta_policy_line(self) -> None:
        policy = LLMInvestigator(
            fallback_policy=_SentinelFallback("investigator"),
            client=_FakeClient([]),
            retries=0,
        )
        text = policy._build_user_prompt(_investigator_obs())
        assert "Meta policy lens: FSDP-IF-03" in text
        assert "ad_001" in text


# ---------------------------------------------------------------------------
# HFInvestigator (local-transformers backend)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Minimal HF tokenizer stand-in: chat-template + decode/encode."""

    pad_token = None
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, **_):
        # We don't care about the actual encoding — the fake model returns
        # a hard-coded string regardless. Return a tiny tensor so the
        # ``encoded["input_ids"].shape[-1]`` slice still works.
        import torch  # local import: tests skip if torch missing
        return {"input_ids": torch.zeros((1, 4), dtype=torch.long)}

    def decode(self, _ids, skip_special_tokens=True):  # noqa: ARG002
        # Returns the reply string injected on the fake model.
        return self._next_reply

    def __init__(self, reply: str = ""):
        self._next_reply = reply


class _FakeHFModel:
    """Minimal HF model stand-in: device + ``.generate`` only."""

    def __init__(self, reply_ids_len: int = 8):
        self._reply_ids_len = reply_ids_len

    def parameters(self):
        # Yield one CPU param so HFInvestigator's ``next(...)`` works
        # without bringing in torch.cuda.
        import torch
        yield torch.zeros(1)

    def generate(self, **kwargs):
        import torch
        prompt_len = kwargs["input_ids"].shape[-1]
        # Append `_reply_ids_len` dummy tokens so the .decode() slice
        # returns the tokenizer's pre-loaded reply text.
        return torch.cat(
            [kwargs["input_ids"],
             torch.zeros((1, self._reply_ids_len), dtype=torch.long)],
            dim=1,
        )


class TestHFInvestigator:
    def test_clean_json_completion_validates_and_records(self) -> None:
        try:
            from counterfeint.agents.hf_investigator import HFInvestigator
        except ImportError:
            pytest.skip("transformers/torch not installed")

        payload = json.dumps(
            {
                "action_type": "investigate",
                "ad_id": "ad_001",
                "investigation_target": "payment_method",
                "rationale": "check payment trail",
            }
        )
        tok = _FakeTokenizer(reply=payload)
        policy = HFInvestigator(
            model=_FakeHFModel(),
            tokenizer=tok,
            fallback_policy=_SentinelFallback("investigator"),
        )

        action = policy.act(_investigator_obs())

        assert action.action_type == "investigate"
        assert action.investigation_target == "payment_method"
        assert policy.fallback_count == 0
        assert policy.last_completion == payload
        assert policy.last_prompt is not None
        assert "ad_001" in policy.last_prompt

    def test_alias_keys_are_coerced_before_validation(self) -> None:
        try:
            from counterfeint.agents.hf_investigator import HFInvestigator
        except ImportError:
            pytest.skip("transformers/torch not installed")

        payload = json.dumps(
            {
                "action_type": "investigate",
                "ad_id": "ad_001",
                "investigation_token": "landing_page",
                "investigation_rationale": "check copy",
            }
        )
        tok = _FakeTokenizer(reply=payload)
        policy = HFInvestigator(
            model=_FakeHFModel(),
            tokenizer=tok,
            fallback_policy=_SentinelFallback("investigator"),
        )

        action = policy.act(_investigator_obs())

        assert action.investigation_target == "landing_page"
        assert "check copy" in (action.rationale or "")
        assert policy.fallback_count == 0

    def test_garbage_completion_falls_back_and_records_error(self) -> None:
        try:
            from counterfeint.agents.hf_investigator import HFInvestigator
        except ImportError:
            pytest.skip("transformers/torch not installed")

        tok = _FakeTokenizer(reply="not json")
        sentinel = _SentinelFallback("investigator")
        policy = HFInvestigator(
            model=_FakeHFModel(),
            tokenizer=tok,
            fallback_policy=sentinel,
        )

        action = policy.act(_investigator_obs())

        assert action.rationale == "sentinel fallback"
        assert policy.fallback_count == 1
        assert policy.last_error is not None
