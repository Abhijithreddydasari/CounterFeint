"""
Blinded 50-case human annotation sheet for Auditor agreement.

Hides corruption labels, sides, and automated flags. A separate key file
maps case_id → expected labels for scoring after annotation.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .corruptions import apply_corruption, sanitize_bundle
from .episode_bundle import EpisodeBundle, load_bundle, run_episode_bundle, save_bundle

ISSUE_TYPES = [
    "none",
    "unsupported",
    "incoherent",
    "clones",
    "mismatch",
    "gibberish",
    "other",
]

TASK_BLURB = {
    "task_1": "small marketplace",
    "task_2": "medium marketplace",
    "task_3": "large marketplace",
}

LABELING_GUIDE = """# How to label (read this once)

## What this is for

For each case, two yes/no questions (Table 4 vs the Auditor / Qwen):

1. **Track A:** Are the *verdicts / rationales* broken?
2. **Track B:** Are the *ads* broken?

Both can be yes. `task_1` / `task_2` / `task_3` = small / medium / large marketplace.

Each case is a **different episode**. If verdicts look familiar, check **Fraudster proposals** and the landing-page **domain** — that is where cases differ.

**Track A** = verdict vs the **landing-page report** (full tool output the investigator pulled).

**Track B** = **Fraudster proposals** (copy / targeting / landing). `×N identical` → `clones`.

## Types — only if that track is `yes`

Leave type `none` when present is `no`.

**Track A** (pick one):
- `unsupported` — no real evidence or policy cited
- `incoherent` — contradicts the findings, pasted rationale, or nonsense confidence
- `other`

**Track B** (pick one):
- `clones` — word-for-word duplicates (`×N identical`)
- `mismatch` — copy vs targeting vs landing disagree (US warehouse + `.tk`, bait tokens)
- `gibberish` — not a real ad
- `other`

## CSV

`track_a_present` / `track_b_present`: `yes` or `no`
`track_a_type` / `track_b_type`: one type above, or `none`
`evidence_span`, `confidence` (`low`/`med`/`high`), `notes`

`×3 identical` → Track B `yes` + `clones`. Track A can still be `no`.

Do **not** open `annotation_key.json`.
"""


OBVIOUS_INV = ["missing_citation", "contradictory_rationale", "overconfidence"]
SUBTLE_INV = ["fabricated_citation", "inconsistent_verdicts", "slice_bias"]
OBVIOUS_FRD = ["gibberish_ads", "grader_trigger_tokens", "template_spam"]
SUBTLE_FRD = [
    "paraphrased_spam",
    "internal_inconsistency",
    "country_tld_mismatch",
    "category_targeting_mismatch",
]


def _snippet(bundle: EpisodeBundle) -> Dict[str, Any]:
    ads = (bundle.episode_record or {}).get("ads") or []
    verdicts = [
        a for a in bundle.investigator_actions if a.get("action_type") == "verdict"
    ][:5]
    proposals = [
        {
            "ad_id": p.get("ad_id"),
            "category": p.get("category"),
            "ad_copy": str(p.get("ad_copy") or "")[:280],
            "targeting": str(p.get("targeting_summary") or "")[:160],
            "landing": str(p.get("landing_page_blurb") or "")[:160],
        }
        for p in bundle.fraudster_proposals
        if p.get("action_type") == "propose_ad"
    ][:5]
    landing_pages: Dict[str, str] = {}
    verdict_ids = [str(v.get("ad_id") or "") for v in verdicts if v.get("ad_id")]
    seen = bundle.investigation_data_seen or {}
    for ad_id in verdict_ids:
        block = seen.get(ad_id) or {}
        lp = str(block.get("landing_page") or "").strip()
        if lp:
            landing_pages[ad_id] = lp
    if not landing_pages:
        for ad_id, block in list(seen.items())[:4]:
            lp = str((block or {}).get("landing_page") or "").strip()
            if lp:
                landing_pages[str(ad_id)] = lp
    return {
        "task_id": bundle.task_id,
        "n_ads": len(ads),
        "verdicts": [
            {
                "ad_id": v.get("ad_id"),
                "verdict": v.get("verdict"),
                "confidence": v.get("confidence"),
                "rationale": str(v.get("rationale") or "")[:320],
            }
            for v in verdicts
        ],
        "findings": landing_pages,
        "proposals": proposals,
    }


def _collapse(items: List[Dict[str, Any]], key_fn) -> List[tuple[List[str], Dict[str, Any]]]:
    """Group identical records; preserve first-seen order."""
    order: List[Any] = []
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for item in items:
        key = key_fn(item)
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(item)
    out: List[tuple[List[str], Dict[str, Any]]] = []
    for key in order:
        bunch = groups[key]
        ids = [str(x.get("ad_id") or "") for x in bunch]
        out.append((ids, bunch[0]))
    return out


def _ids(ids: List[str]) -> str:
    shown = ", ".join(f"`{i}`" for i in ids if i)
    if len(ids) <= 1:
        return shown
    return f"{shown} **×{len(ids)} identical**"


def render_case_markdown(case_id: str, snippet: Dict[str, Any]) -> List[str]:
    task = str(snippet.get("task_id") or "task_1")
    blurb = TASK_BLURB.get(task, "marketplace")
    n_ads = snippet.get("n_ads", "?")
    lines = [
        f"## {case_id}",
        "",
        f"**Marketplace:** {blurb} (`{task}`). This episode has **{n_ads} ads** total; below is a slice.",
    ]
    proposals = list(snippet.get("proposals") or [])
    lines.extend(["", "### Fraudster proposals"])
    if not proposals:
        lines.append("- (none shown)")
    else:
        for ids, p in _collapse(
            proposals,
            lambda x: (
                x.get("ad_copy"),
                x.get("category"),
                x.get("targeting"),
                x.get("landing"),
            ),
        ):
            extra = []
            if p.get("targeting"):
                extra.append(f"targeting: {p.get('targeting')}")
            if p.get("landing"):
                extra.append(f"landing: {p.get('landing')}")
            suffix = f" — {'; '.join(extra)}" if extra else ""
            lines.append(
                f"- {_ids(ids)} [{p.get('category')}]: {p.get('ad_copy')}{suffix}"
            )
    lines.extend(["", "### Investigator verdicts"])
    verdicts = list(snippet.get("verdicts") or [])
    if not verdicts:
        lines.append("- (none shown)")
    else:
        for ids, v in _collapse(
            verdicts,
            lambda x: (x.get("verdict"), x.get("confidence"), x.get("rationale")),
        ):
            lines.append(
                f"- {_ids(ids)} → **{v.get('verdict')}** (conf={v.get('confidence')}): "
                f"{v.get('rationale')}"
            )
    findings = snippet.get("findings") or {}
    if findings:
        lines.extend(
            [
                "",
                "### Landing-page report (full tool output the investigator pulled)",
            ]
        )
        for ad_id, block in list(findings.items())[:5]:
            if isinstance(block, dict):
                text = str(block.get("landing_page") or "").strip()
            else:
                text = str(block or "").strip()
            if not text:
                continue
            lines.append(f"`{ad_id}`")
            lines.append("```")
            lines.append(text)
            lines.append("```")
    lines.append("")
    return lines


def readable_header() -> List[str]:
    return [
        "# Blinded Auditor annotation sheet",
        "",
        "Read `LABELING_GUIDE.md` first (issue types + what `task_2, 17 ads` means).",
        "",
        "Label `labels_template.csv` → save as `labels.csv`. Two questions per case: Track A (verdicts) and Track B (ads).",
        "",
        "Each case is a different episode. Look at Fraudster proposals first; that is often what changed.",
        "",
        "Identical ad rows are **collapsed** (`×N identical`). That is Track B `clones`.",
        "",
    ]


def write_readable_markdown(cases: List[Dict[str, Any]], path: Path) -> None:
    md_lines = readable_header()
    for case in cases:
        md_lines.extend(render_case_markdown(case["case_id"], case.get("snippet") or {}))
    path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def rewrite_readable(output_dir: Path) -> None:
    """Refresh the human-readable sheet from bundles. Does not change case IDs, key, or CSV."""
    sheet = json.loads((output_dir / "annotation_sheet.json").read_text(encoding="utf-8"))
    cases: List[Dict[str, Any]] = []
    for c in sheet.get("cases") or []:
        cid = str(c.get("case_id") or "")
        bundle_path = output_dir / "bundles" / f"{cid}.json"
        if bundle_path.exists():
            snippet = _snippet(load_bundle(bundle_path))
        else:
            snippet = c.get("snippet") or {}
        cases.append({"case_id": cid, "snippet": snippet})
    (output_dir / "annotation_sheet.json").write_text(
        json.dumps({"n": len(cases), "cases": cases}, indent=2), encoding="utf-8"
    )
    write_readable_markdown(cases, output_dir / "annotation_readable.md")
    (output_dir / "LABELING_GUIDE.md").write_text(LABELING_GUIDE.strip() + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Auditor annotation (blinded)",
                "",
                "1. Read `LABELING_GUIDE.md` (what each issue type means).",
                "2. Label cases in `annotation_readable.md`.",
                "3. Fill `labels_template.csv` (Track A + Track B columns) and save as `labels.csv`.",
                "4. Do not open `annotation_key.json` until you finish.",
                "5. Score: `python -m counterfeint.experiments.score_annotations`.",
                "",
                "Cases: 50 (25 Investigator-side, 25 Fraudster-side).",
                "Word-for-word duplicate ads are collapsed (`clones`).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _load_pool(
    seeds_by_task: Dict[str, List[int]], cache_dir: Optional[Path]
) -> List[EpisodeBundle]:
    pool: List[EpisodeBundle] = []
    for task_id, seeds in seeds_by_task.items():
        for seed in seeds:
            cached = None
            if cache_dir is not None:
                path = cache_dir / f"{task_id}_seed{seed}.json"
                if path.exists():
                    cached = load_bundle(path)
            pool.append(cached or run_episode_bundle(task_id=task_id, seed=seed))
    return pool


def _human_control_ads(bundle: EpisodeBundle) -> EpisodeBundle:
    """Replace sanitize() SKU placeholders with unique, readable product ads."""
    catalog = [
        (
            "Carbon-plate trail shoes. Free 30-day returns, ships from Ohio.",
            "adults 25-54, running",
            "https://outlet{n}.shop.example/shoes",
        ),
        (
            "Stainless 12-cup coffee maker with a 2-year warranty.",
            "home, kitchen, adults 30-65",
            "https://outlet{n}.shop.example/coffee",
        ),
        (
            "Organic cotton tee, sizes XS–XXL. Wash cold.",
            "adults 18-45, fashion",
            "https://outlet{n}.shop.example/tee",
        ),
        (
            "Memory-foam pillow, hypoallergenic cover, 60-night trial.",
            "adults 25-64, home",
            "https://outlet{n}.shop.example/pillow",
        ),
        (
            "Refurbished laptop, Intel i5, 16GB RAM, 1-year depot warranty.",
            "adults 22-55, electronics",
            "https://outlet{n}.shop.example/laptop",
        ),
        (
            "Cast-iron skillet, pre-seasoned, made in USA.",
            "home, cooking, adults 25-60",
            "https://outlet{n}.shop.example/skillet",
        ),
    ]
    i = 0
    for prop in bundle.fraudster_proposals:
        if prop.get("action_type") != "propose_ad":
            continue
        copy, tgt, land = catalog[i % len(catalog)]
        prop["ad_copy"] = f"{copy} Lot {bundle.task_id}-{bundle.seed}-{i + 1}."
        prop["targeting_summary"] = tgt
        prop["landing_page_blurb"] = land.format(n=f"{bundle.seed}{i}")
        prop["category"] = "ecommerce"
        i += 1
    return bundle


def _pop_unique(pool: List[EpisodeBundle]) -> EpisodeBundle:
    if not pool:
        raise RuntimeError("annotation pool exhausted; need 50 unique episodes")
    return pool.pop()


def stage_annotation_cases(
    pool: List[EpisodeBundle],
    rng: random.Random,
) -> List[tuple[EpisodeBundle, bool, str]]:
    leftover = list(pool)
    rng.shuffle(leftover)
    staged: List[tuple[EpisodeBundle, bool, str]] = []

    def take_clean(n: int, side: str) -> None:
        for _ in range(n):
            clean = _human_control_ads(sanitize_bundle(_pop_unique(leftover)))
            clean.corruption_label = "clean"
            staged.append((clean, True, side))

    def take_attacks(names: List[str], n: int, side: str) -> None:
        for i in range(n):
            name = names[i % len(names)]
            base = _human_control_ads(sanitize_bundle(_pop_unique(leftover)))
            staged.append((apply_corruption(base, name), False, side))

    take_clean(5, "investigator")
    take_attacks(OBVIOUS_INV, 10, "investigator")
    take_attacks(SUBTLE_INV, 10, "investigator")
    take_clean(5, "fraudster")
    take_attacks(OBVIOUS_FRD, 10, "fraudster")
    take_attacks(SUBTLE_FRD, 10, "fraudster")
    rng.shuffle(staged)
    keys = [(b.task_id, b.seed) for b, _, _ in staged]
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"duplicate annotation episodes: {keys}")
    return staged


def build_annotation_sheet(
    *,
    output_dir: Path,
    seeds_by_task: Optional[Dict[str, List[int]]] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write blinded cases + hidden key + labels template (exactly 50 unique episodes)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bundles").mkdir(parents=True, exist_ok=True)
    seeds_by_task = seeds_by_task or {
        "task_1": list(range(11, 31)),
        "task_2": list(range(11, 31)),
        "task_3": list(range(11, 21)),
    }
    if cache_dir is None:
        guess = Path("experiments/outputs/auditor_expanded/corruption/base_trajectories")
        cache_dir = guess if guess.exists() else None
    rng = random.Random(7)
    pool = _load_pool(seeds_by_task, cache_dir)
    staged = stage_annotation_cases(pool, rng)

    cases: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []

    for i, (bundle, is_clean, side) in enumerate(staged, start=1):
        case_id = f"C{i:03d}"
        snippet = _snippet(bundle)
        cases.append({"case_id": case_id, "snippet": snippet})
        expected = list(bundle.expected_flags or [])
        if not expected and bundle.expected_llm_flags:
            expected_type = bundle.expected_llm_flags[0]
        elif expected:
            expected_type = expected[0]
        elif is_clean:
            expected_type = "none"
        else:
            expected_type = "other"
        key_rows.append(
            {
                "case_id": case_id,
                "task_id": bundle.task_id,
                "seed": bundle.seed,
                "corruption_label": bundle.corruption_label or "clean",
                "side": side,
                "expected_flags": bundle.expected_flags,
                "expected_llm_flags": bundle.expected_llm_flags,
                "expected_issue_type": expected_type,
                "is_clean": is_clean,
            }
        )
        save_bundle(bundle, output_dir / "bundles" / f"{case_id}.json")

    (output_dir / "annotation_sheet.json").write_text(
        json.dumps({"n": len(cases), "cases": cases}, indent=2), encoding="utf-8"
    )
    (output_dir / "annotation_key.json").write_text(
        json.dumps(key_rows, indent=2), encoding="utf-8"
    )
    rewrite_readable(output_dir)

    labels_path = output_dir / "labels_template.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case_id",
                "track_a_present",
                "track_a_type",
                "track_b_present",
                "track_b_type",
                "evidence_span",
                "confidence",
                "notes",
            ],
        )
        writer.writeheader()
        for c in cases:
            writer.writerow(
                {
                    "case_id": c["case_id"],
                    "track_a_present": "",
                    "track_a_type": "",
                    "track_b_present": "",
                    "track_b_type": "",
                    "evidence_span": "",
                    "confidence": "",
                    "notes": "",
                }
            )

    return {"n": len(cases), "path": str(output_dir)}


def load_labels(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


_YES = {"yes", "y", "1", "true"}

# Human types are coarse; attack-key types stay fine-grained.
TYPE_BUCKETS: Dict[str, set[str]] = {
    "unsupported": {"unsupported", "missing_citation"},
    "incoherent": {
        "incoherent",
        "incoherent_rationale",
        "inconsistency",
        "miscalibration",
        "bias",
    },
    "clones": {"clones", "template_repetition"},
    "mismatch": {"mismatch", "parameter_mismatch", "branding_anomaly"},
    "gibberish": {"gibberish"},
    "other": {"other"},
    "none": {"none", ""},
}


def types_agree(gold: str, pred: str) -> bool:
    g = (gold or "none").strip() or "none"
    p = (pred or "none").strip() or "none"
    if g == p:
        return True
    for names in TYPE_BUCKETS.values():
        if g in names and p in names:
            return True
    return False


def _is_yes(val: Any) -> bool:
    return str(val or "").strip().lower() in _YES


def _row_is_labeled(row: Dict[str, str]) -> bool:
    return bool(
        str(
            row.get("track_a_present")
            or row.get("track_b_present")
            or row.get("issue_present")
            or ""
        ).strip()
    )


def normalize_label_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Accept dual-track CSV or legacy single issue_present column."""
    cid = str(row.get("case_id") or "")
    if str(row.get("track_a_present") or row.get("track_b_present") or "").strip():
        a = _is_yes(row.get("track_a_present"))
        b = _is_yes(row.get("track_b_present"))
        a_type = str(row.get("track_a_type") or "none").strip() or "none"
        b_type = str(row.get("track_b_type") or "none").strip() or "none"
        if not a:
            a_type = "none"
        if not b:
            b_type = "none"
        return {
            "case_id": cid,
            "track_a_present": a,
            "track_b_present": b,
            "issue_present": a or b,
            "issue_type": a_type if a else (b_type if b else "none"),
            "track_a_type": a_type,
            "track_b_type": b_type,
            "evidence_span": str(row.get("evidence_span") or ""),
            "confidence": str(row.get("confidence") or ""),
            "notes": str(row.get("notes") or ""),
        }
    present = _is_yes(row.get("issue_present"))
    itype = str(row.get("issue_type") or "none").strip() or "none"
    return {
        "case_id": cid,
        "track_a_present": present,
        "track_b_present": present,
        "issue_present": present,
        "issue_type": itype if present else "none",
        "track_a_type": itype if present else "none",
        "track_b_type": itype if present else "none",
        "evidence_span": str(row.get("evidence_span") or ""),
        "confidence": str(row.get("confidence") or ""),
        "notes": str(row.get("notes") or ""),
    }


def score_against_key(
    *,
    labels: List[Dict[str, str]],
    key_path: Path,
) -> Dict[str, Any]:
    key = {row["case_id"]: row for row in json.loads(key_path.read_text(encoding="utf-8"))}
    tp = fp = fn = tn = 0
    typed_agree = 0
    n = 0
    for row in labels:
        cid = row.get("case_id", "")
        if cid not in key or not _row_is_labeled(row):
            continue
        n += 1
        gold_pos = not bool(key[cid].get("is_clean"))
        norm = normalize_label_row(row)
        pred_pos = bool(norm["issue_present"])
        if gold_pos and pred_pos:
            tp += 1
        elif pred_pos and not gold_pos:
            fp += 1
        elif gold_pos and not pred_pos:
            fn += 1
        else:
            tn += 1
        gold_type = str(key[cid].get("expected_issue_type") or "none")
        a_type = str(norm.get("track_a_type") or "none")
        b_type = str(norm.get("track_b_type") or "none")
        if gold_type in {"none", ""}:
            if types_agree("none", a_type) and types_agree("none", b_type):
                typed_agree += 1
        elif types_agree(gold_type, a_type) or types_agree(gold_type, b_type):
            typed_agree += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    po = (tp + tn) / n if n else 0.0
    p_yes = ((tp + fp) / n) * ((tp + fn) / n) if n else 0.0
    p_no = ((tn + fn) / n) * ((tn + fp) / n) if n else 0.0
    pe = p_yes + p_no
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    bal_acc = 0.0
    if n:
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        bal_acc = 0.5 * (tpr + tnr)
    return {
        "n_labeled": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "type_agreement": round(typed_agree / n, 4) if n else 0.0,
        "cohens_kappa_presence": round(kappa, 4),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded/annotation"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded/corruption/base_trajectories"),
    )
    parser.add_argument(
        "--rewrite-readable",
        action="store_true",
        help="Refresh annotation_readable.md + LABELING_GUIDE.md without changing cases",
    )
    args = parser.parse_args()
    if args.rewrite_readable:
        rewrite_readable(args.output)
        print(json.dumps({"rewrote": str(args.output / "annotation_readable.md")}))
    else:
        print(
            json.dumps(
                build_annotation_sheet(output_dir=args.output, cache_dir=args.cache_dir),
                indent=2,
            )
        )
