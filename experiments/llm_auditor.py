"""
Offline LLM Auditor (Qwen3.5-9B via vLLM) and hybrid merge policy.

The judge never participates in training. It reads saved EpisodeBundles and
returns the same typed AuditFlag schema as the deterministic Auditor.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from counterfeint.models import AuditFlag

from .episode_bundle import EpisodeBundle, load_bundle

LLM_FLAG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "track": {"type": "string", "enum": ["A", "B"]},
                    "flag_type": {
                        "type": "string",
                        "enum": [
                            "miscalibration",
                            "missing_citation",
                            "incoherent_rationale",
                            "inconsistency",
                            "bias",
                            "gibberish",
                            "parameter_mismatch",
                            "template_repetition",
                            "branding_anomaly",
                        ],
                    },
                    "target_ad_id": {"type": ["string", "null"]},
                    "severity": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["track", "flag_type", "severity"],
            },
        }
    },
    "required": ["flags"],
}

SEMANTIC_FLAG_TYPES = {
    "incoherent_rationale",
    "template_repetition",
    "parameter_mismatch",
    "missing_citation",
}
EXACT_FLAG_TYPES = {
    "miscalibration",
    "bias",
    "inconsistency",
    "gibberish",
    "branding_anomaly",
}

JUDGE_SYSTEM = """You are an independent audit judge for ad-fraud investigation episodes.
You do not see deterministic grader flags or rewards.
Return a JSON object {"flags": [...]} and nothing else.

Track A (Investigator): missing_citation, incoherent_rationale, miscalibration,
inconsistency, bias.
Track B (Fraudster): gibberish, parameter_mismatch, template_repetition, branding_anomaly.

Inspect every verdict and every ad. Prefer recall over silence: if you see a real
issue, emit a flag. Empty flags only when verdicts are well-cited AND ads are
coherent, on-category, and non-duplicative.

Examples:
{"flags":[{"track":"A","flag_type":"missing_citation","target_ad_id":"ad_001","severity":0.9,"note":"reject with no evidence"}]}
{"flags":[{"track":"B","flag_type":"parameter_mismatch","target_ad_id":"ad_002","severity":0.8,"note":"US targeting with .tk landing"}]}
{"flags":[]}
"""


def bundle_to_judge_prompt(bundle: EpisodeBundle) -> str:
    ads = (bundle.episode_record or {}).get("ads") or []
    verdicts = (bundle.episode_record or {}).get("verdicts") or []
    proposals = [
        {
            "ad_id": p.get("ad_id"),
            "category": p.get("category"),
            "ad_copy": str(p.get("ad_copy") or "")[:280],
            "targeting": str(p.get("targeting_summary") or "")[:140],
            "landing": str(p.get("landing_page_blurb") or "")[:140],
        }
        for p in bundle.fraudster_proposals
        if p.get("action_type") == "propose_ad"
    ]
    inv = [
        {
            "action": a.get("action_type"),
            "ad_id": a.get("ad_id"),
            "verdict": a.get("verdict"),
            "confidence": a.get("confidence"),
            "rationale": str(a.get("rationale") or "")[:240],
        }
        for a in bundle.investigator_actions
        if a.get("action_type") in ("verdict", "link_accounts")
    ]
    payload = {
        "task_id": bundle.task_id,
        "ads": [
            {"ad_id": a.get("ad_id"), "category": a.get("category"), "ground_truth": a.get("ground_truth")}
            for a in ads
        ],
        "verdicts": verdicts[:20],
        "investigator_verdicts": inv[:20],
        "fraudster_proposals": proposals[:8],
        "findings": {
            ad_id: str((findings or {}).get("landing_page") or "")
            for ad_id, findings in list(bundle.investigation_data_seen.items())[:8]
        },
    }
    return "EPISODE:\n" + json.dumps(payload, ensure_ascii=False)


def _loads_jsonish(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    if not text:
        return {"flags": []}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = text.find(opener), text.rfind(closer)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise


def _flag_items(data: Any) -> List[Any]:
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("flags", "audit_flags", "items"):
            if key in data:
                val = data[key]
                if isinstance(val, list):
                    return val
                if isinstance(val, dict):
                    return [val]
        if "flag_type" in data or "track" in data:
            return [data]
    return []


def parse_llm_flags(raw: Any) -> List[AuditFlag]:
    data = raw if isinstance(raw, (dict, list)) else _loads_jsonish(str(raw))
    flags: List[AuditFlag] = []
    for item in _flag_items(data):
        if not isinstance(item, dict):
            continue
        track = str(item.get("track") or "A")
        if track not in ("A", "B"):
            track = "B" if str(item.get("flag_type") or "") in EXACT_FLAG_TYPES else "A"
        try:
            sev = float(item.get("severity", 0.5) or 0.5)
        except (TypeError, ValueError):
            sev = 0.5
        flags.append(
            AuditFlag(
                track=track,  # type: ignore[arg-type]
                target_ad_id=item.get("target_ad_id"),
                flag_type=str(item.get("flag_type") or "missing_citation"),
                severity=min(1.0, max(0.0, sev)),
                note=("llm:" + str(item.get("note") or ""))[:2000],
            )
        )
    return flags


def merge_hybrid(
    det_flags: List[AuditFlag],
    llm_flags: List[AuditFlag],
) -> List[AuditFlag]:
    """Exact checks stay deterministic; semantic flags can come from either.

    Severe flags from either source are preserved, tagged in note.
    """
    merged: Dict[tuple, AuditFlag] = {}
    for f in det_flags:
        merged[(f.flag_type, f.target_ad_id, f.track)] = f
    for f in llm_flags:
        key = (f.flag_type, f.target_ad_id, f.track)
        if f.flag_type in SEMANTIC_FLAG_TYPES:
            if key not in merged or (f.severity or 0) > (merged[key].severity or 0):
                merged[key] = f
        elif f.flag_type in EXACT_FLAG_TYPES:
            # LLM may add a semantic supplement; keep deterministic if present.
            if key not in merged and (f.severity or 0) >= 0.7:
                merged[key] = f
        else:
            if key not in merged:
                merged[key] = f
    return list(merged.values())


async def judge_bundle_async(bundle: EpisodeBundle) -> tuple[List[AuditFlag], float, str]:
    """Call vLLM OpenAI-compatible API. Returns (flags, elapsed_sec)."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError("openai package required for LLM auditor") from exc

    client = AsyncOpenAI(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
    model = os.environ.get("VLLM_JUDGE_MODEL", "Qwen/Qwen3.5-9B")
    prompt = bundle_to_judge_prompt(bundle)
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.0,
        extra_body={
            "guided_json": LLM_FLAG_SCHEMA,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    elapsed = time.perf_counter() - t0
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    if not content:
        content = (getattr(msg, "reasoning_content", None) or "").strip()
    if not content:
        extra = getattr(msg, "model_extra", None) or {}
        content = str(extra.get("reasoning") or extra.get("reasoning_content") or "").strip()
    if not content:
        raise RuntimeError("empty LLM judge completion")
    return parse_llm_flags(content), elapsed, content[:1500]


def run_llm_auditor_on_dir(
    *,
    bundles_dir: Path,
    output_path: Path,
    max_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """Synchronous wrapper: judge cached bundles if a vLLM server is up."""
    import asyncio

    paths = sorted(bundles_dir.glob("*.json"))
    if max_cases:
        paths = paths[:max_cases]
    bundles = [load_bundle(p) for p in paths]
    results: Dict[str, List[Dict[str, Any]]] = {}
    latencies: List[float] = []

    async def _run() -> None:
        for b in bundles:
            key = f"{b.task_id}:{b.seed}:{b.corruption_label or 'clean'}"
            try:
                flags, elapsed, _raw = await judge_bundle_async(b)
            except Exception as exc:  # noqa: BLE001
                results[key] = [{"error": str(exc)}]
                continue
            latencies.append(elapsed)
            results[key] = [
                {
                    "track": f.track,
                    "flag_type": f.flag_type,
                    "target_ad_id": f.target_ad_id,
                    "severity": f.severity,
                    "note": f.note,
                }
                for f in flags
            ]

    asyncio.run(_run())
    payload = {
        "model": os.environ.get("VLLM_JUDGE_MODEL", "Qwen/Qwen3.5-9B"),
        "n_cases": len(bundles),
        "mean_latency_sec": sum(latencies) / len(latencies) if latencies else None,
        "flags": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def collect_judge_targets(
    *,
    cache_dir: Path,
    output_dir: Path,
    attacks: Optional[List[str]] = None,
    max_bases: Optional[int] = None,
) -> int:
    """Materialize clean + corrupted bundles for the offline LLM judge."""
    from .corruptions import FRAUDSTER_ATTACKS, INVESTIGATOR_ATTACKS, apply_corruption
    from .episode_bundle import save_bundle

    attacks = attacks or (INVESTIGATOR_ATTACKS + FRAUDSTER_ATTACKS)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(cache_dir.glob("*.json"))
    if max_bases:
        paths = paths[:max_bases]
    n = 0
    for path in paths:
        base = load_bundle(path)
        save_bundle(base, output_dir / f"{base.task_id}_seed{base.seed}_clean.json")
        n += 1
        for name in attacks:
            corrupted = apply_corruption(base, name)
            corrupted.corruption_label = name
            save_bundle(
                corrupted,
                output_dir / f"{base.task_id}_seed{base.seed}_{name}.json",
            )
            n += 1
    return n
