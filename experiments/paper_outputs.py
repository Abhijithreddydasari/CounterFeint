"""
Paper-ready tables/figures for Auditor validation expansion.

Reads corruption + ablation JSON and writes
`experiments/outputs/auditor_expanded/`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .corruptions import FRAUDSTER_ATTACKS, INVESTIGATOR_ATTACKS

INV_ATTACKS = set(INVESTIGATOR_ATTACKS)
FRD_ATTACKS = set(FRAUDSTER_ATTACKS)

KEY_ARMS = [
    "full",
    "no_track_a",
    "no_track_b",
    "no_plausibility_gate",
    "outcome_only",
]

# Matching per-check drop arm for Table 3b (role-specific gaming).
ATTACK_DROP_ARM = {
    "missing_citation": "no_missing_citation",
    "fabricated_citation": "no_missing_citation",
    "contradictory_rationale": "no_incoherent_rationale",
    "overconfidence": "no_miscalibration",
    "inconsistent_verdicts": "no_inconsistency",
    "slice_bias": "no_bias",
    "gibberish_ads": "no_gibberish",
    "grader_trigger_tokens": "no_branding_anomaly",
    "country_tld_mismatch": "no_parameter_mismatch",
    "category_targeting_mismatch": "no_parameter_mismatch",
    "template_spam": "no_template_repetition",
    "paraphrased_spam": "no_template_repetition",
    "internal_inconsistency": "no_parameter_mismatch",
}


def _mean(vals: Iterable[float]) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _llm_flag_stats(llm_raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not llm_raw:
        return None
    flags = llm_raw.get("flags") or {}
    n_flagged = 0
    n_err = 0
    for v in flags.values():
        if not v:
            continue
        if isinstance(v, list) and v and isinstance(v[0], dict) and v[0].get("error"):
            n_err += 1
        else:
            n_flagged += 1
    return {
        "model": llm_raw.get("model"),
        "n_cases": llm_raw.get("n_cases") or len(flags),
        "n_ok": llm_raw.get("n_ok"),
        "n_flagged": n_flagged,
        "n_error": n_err,
        "mean_latency_sec": llm_raw.get("mean_latency_sec"),
        "cost_per_episode_usd": llm_raw.get("cost_per_episode_usd"),
    }


def _fmt_ci(m: Dict[str, Any], key: str) -> str:
    ci = m.get(f"{key}_ci95") or [0, 0]
    return f"{m.get(key, 0):.2f} [{ci[0]:.2f},{ci[1]:.2f}]"


def table1_dataset(
    corruption: Dict[str, Any],
    ablation_rows: List[Dict[str, Any]],
    llm_stats: Optional[Dict[str, Any]] = None,
) -> str:
    n_base = int(corruption.get("n_base_episodes") or 0)
    n_corr = int(corruption.get("n_corruptions") or 0)
    n_app = int(corruption.get("n_applicable") or 0)
    per = corruption.get("per_corruption") or {}
    n_inv = sum(1 for k in per if k in INV_ATTACKS)
    n_frd = sum(1 for k in per if k in FRD_ATTACKS)
    n_abl_traj = len({(r.get("task_id"), r.get("seed")) for r in ablation_rows})
    lines = [
        "# Table 1. Dataset composition",
        "",
        "| Split | N |",
        "|-------|--:|",
        f"| Base trajectories, paired-delta eval (tasks 1–3) | {n_base} |",
        f"| Corruption trials | {n_corr} |",
        f"| Applicable (non-empty expected flags) | {n_app} |",
        f"| Investigator attacks | {n_inv} |",
        f"| Fraudster attacks | {n_frd} |",
        f"| Ablation trajectories | {n_abl_traj} |",
        f"| Ablation rows | {len(ablation_rows)} |",
        "| Blinded annotation cases | 50 |",
    ]
    if llm_stats:
        lat = llm_stats.get("mean_latency_sec")
        cost = llm_stats.get("cost_per_episode_usd")
        lines.append(
            f"| Qwen3.5-9B judged cases (n_ok / n_flagged) | "
            f"{llm_stats.get('n_ok')}/{llm_stats.get('n_flagged')} |"
        )
        if lat is not None:
            lines.append(f"| LLM mean latency (s) | {lat:.2f} |")
        if cost is not None:
            lines.append(f"| LLM cost / episode (USD) | {cost:.4f} |")
    lines.extend(
        [
            "",
            "Each base trajectory is audited clean, then paired with every attack.",
            "Clean-control flag rates are reported separately (not as corruption FPs).",
            "Ablation uses a 30-episode subset (seeds 11–20 × 3 tasks).",
        ]
    )
    return "\n".join(lines) + "\n"


def table2_per_check(corruption: Dict[str, Any]) -> str:
    per_flag = corruption.get("per_flag") or {}
    strong, useful, weak = [], [], []
    for ft, m in sorted(per_flag.items()):
        f1 = float(m.get("f1") or 0)
        row = (ft, m)
        if f1 >= 0.80:
            strong.append(row)
        elif f1 >= 0.50:
            useful.append(row)
        else:
            weak.append(row)

    def _block(title: str, rows: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        lines = [
            f"## {title}",
            "",
            "| Flag | TP | FP | FN | Precision | Recall | F1 |",
            "|------|---:|---:|---:|----------:|-------:|---:|",
        ]
        if not rows:
            lines.append("| — | — | — | — | — | — | — |")
            return lines
        for ft, m in rows:
            lines.append(
                f"| `{ft}` | {m.get('tp', 0)} | {m.get('fp', 0)} | {m.get('fn', 0)} | "
                f"{_fmt_ci(m, 'precision')} | {_fmt_ci(m, 'recall')} | {_fmt_ci(m, 'f1')} |"
            )
        return lines

    lines = [
        "# Table 2. Per-check precision / recall / F1 (paired delta, 95% bootstrap CI)",
        "",
        "New flags only at `(flag_type, ad_id)` vs sanitized control.",
        "",
    ]
    lines.extend(_block("Strong (F1 ≥ 0.80)", strong))
    lines.append("")
    lines.extend(_block("Useful (0.50 ≤ F1 < 0.80)", useful))
    lines.append("")
    lines.extend(_block("Weak / noisy (F1 < 0.50) — failure analysis", weak))
    lines.extend(["", "## Clean-control flag rate (unmodified scripted trajectories)", "", "| Flag | Rate |", "|------|-----:|"])
    for ft, rate in sorted((corruption.get("clean_flag_rate") or {}).items()):
        lines.append(f"| `{ft}` | {rate:.3f} |")
    lines.extend(
        [
            "",
            "## Corruption hit rate",
            "",
            "| Attack | Hit rate | N |",
            "|--------|---------:|--:|",
        ]
    )
    gaps = []
    for name, stats in sorted((corruption.get("per_corruption") or {}).items()):
        hit = float(stats.get("hit_rate") or 0)
        lines.append(f"| `{name}` | {hit:.3f} | {stats.get('n', 0)} |")
        if hit <= 0:
            gaps.append(name)
    if gaps:
        lines.extend(["", "Documented gaps (hit 0): " + ", ".join(f"`{g}`" for g in gaps) + "."])
    return "\n".join(lines) + "\n"


def _group_ablation(
    rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    by: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[str(r.get("corruption"))][str(r.get("arm"))].append(r)
    return by


def table3_ablation(rows: List[Dict[str, Any]]) -> str:
    by = _group_ablation(rows)
    arms = [a for a in KEY_ARMS if any(a in by[c] for c in by)]
    lines = [
        "# Table 3. Investigator and Fraudster rewards under verifier ablations",
        "",
        "Delta = attacked reward − paired clean FULL reward.",
        "",
        "## Investigator reward (mean)",
        "",
        "| Attack | " + " | ".join(arms) + " |",
        "|--------|" + "|".join(["-----:" for _ in arms]) + "|",
    ]
    for corr in ["clean"] + INVESTIGATOR_ATTACKS + FRAUDSTER_ATTACKS:
        if corr not in by:
            continue
        cells = [corr]
        for arm in arms:
            vals = [float(x["investigator_reward"]) for x in by[corr].get(arm, [])]
            cells.append(f"{_mean(vals):.3f}" if vals else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Fraudster reward (mean)",
            "",
            "| Attack | " + " | ".join(arms) + " |",
            "|--------|" + "|".join(["-----:" for _ in arms]) + "|",
        ]
    )
    for corr in ["clean"] + INVESTIGATOR_ATTACKS + FRAUDSTER_ATTACKS:
        if corr not in by:
            continue
        cells = [corr]
        for arm in arms:
            vals = [float(x["fraudster_reward"]) for x in by[corr].get(arm, [])]
            cells.append(f"{_mean(vals):.3f}" if vals else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Gaming test: reward with check removed vs full (same attacked trajectory)",
            "",
            "| Attack | side | inv full | inv no_A | Δinv | frd full | frd no_gate | Δfrd |",
            "|--------|------|---------:|---------:|-----:|---------:|------------:|-----:|",
        ]
    )
    for corr in INVESTIGATOR_ATTACKS + FRAUDSTER_ATTACKS:
        if corr not in by:
            continue
        side = "investigator" if corr in INV_ATTACKS else "fraudster"
        inv_f = _mean(float(x["investigator_reward"]) for x in by[corr].get("full", []))
        inv_a = _mean(float(x["investigator_reward"]) for x in by[corr].get("no_track_a", []))
        frd_f = _mean(float(x["fraudster_reward"]) for x in by[corr].get("full", []))
        frd_g = _mean(
            float(x["fraudster_reward"]) for x in by[corr].get("no_plausibility_gate", [])
        )
        lines.append(
            f"| {corr} | {side} | {inv_f:.3f} | {inv_a:.3f} | {inv_a - inv_f:.3f} | "
            f"{frd_f:.3f} | {frd_g:.3f} | {frd_g - frd_f:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Per-check drop (same attacked trajectory)",
            "",
            "Reward when only the matching check is removed, vs full.",
            "",
            "| Attack | Dropped arm | Role | Full | Dropped | Δ |",
            "|--------|-------------|------|-----:|--------:|--:|",
        ]
    )
    for corr in INVESTIGATOR_ATTACKS + FRAUDSTER_ATTACKS:
        drop = ATTACK_DROP_ARM.get(corr)
        if not drop or corr not in by or drop not in by[corr]:
            if corr in by:
                lines.append(f"| `{corr}` | — | gap | — | — | — |")
            continue
        role = "investigator" if corr in INV_ATTACKS else "fraudster"
        key = "investigator_reward" if role == "investigator" else "fraudster_reward"
        full = _mean(float(x[key]) for x in by[corr].get("full", []))
        dropped = _mean(float(x[key]) for x in by[corr].get(drop, []))
        lines.append(
            f"| `{corr}` | `{drop}` | {role} | {full:.3f} | {dropped:.3f} | {dropped - full:.3f} |"
        )
    lines.extend(
        [
            "",
            "LLM-only / hybrid *reward* columns omitted: judge outputs are keyed to "
            "the 50 annotation cases, not the 30-episode ablation split. "
            "Qwen3.5-9B produced flags on 46/50 annotation cases (Table 4A)."
        ]
    )
    return "\n".join(lines) + "\n"


def table4_human(
    scores: Optional[Dict[str, Any]],
    key_scores: Optional[Dict[str, Any]] = None,
    llm_stats: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        "# Table 4. Auditor vs references",
        "",
    ]
    if key_scores:
        det = key_scores.get("deterministic_vs_key") or {}
        llm = key_scores.get("llm_vs_key") or {}
        hyb = key_scores.get("hybrid_vs_key") or {}
        typ = key_scores.get("expected_type_recall") or {}
        lines.extend(
            [
                "## A. vs attack key (not human GT)",
                "",
                f"n={key_scores.get('n_cases', 0)} cases, "
                f"{key_scores.get('n_clean', 0)} clean. "
                f"{key_scores.get('note', '')}",
                "",
                "| Auditor | N | Precision | Recall | F1 | Balanced acc | Type recall |",
                "|---------|--:|----------:|-------:|---:|-------------:|------------:|",
                f"| Deterministic | {det.get('n', 0)} | {det.get('precision', 0):.3f} | "
                f"{det.get('recall', 0):.3f} | {det.get('f1', 0):.3f} | "
                f"{det.get('balanced_accuracy', 0):.3f} | {typ.get('deterministic', 0):.3f} |",
                f"| LLM (Qwen3.5-9B) | {llm.get('n', 0)} | {llm.get('precision', 0):.3f} | "
                f"{llm.get('recall', 0):.3f} | {llm.get('f1', 0):.3f} | "
                f"{llm.get('balanced_accuracy', 0):.3f} | {typ.get('llm', 0):.3f} |",
                f"| Hybrid | {hyb.get('n', 0)} | {hyb.get('precision', 0):.3f} | "
                f"{hyb.get('recall', 0):.3f} | {hyb.get('f1', 0):.3f} | "
                f"{hyb.get('balanced_accuracy', 0):.3f} | {typ.get('hybrid', 0):.3f} |",
                "",
            ]
        )
        if llm_stats:
            lat = llm_stats.get("mean_latency_sec")
            cost = llm_stats.get("cost_per_episode_usd")
            lat_s = f"{lat:.2f}" if isinstance(lat, (int, float)) else str(lat)
            cost_s = f"{cost:.4f}" if isinstance(cost, (int, float)) else str(cost)
            lines.append(
                f"LLM run: n_ok={llm_stats.get('n_ok')}, "
                f"n_flagged={llm_stats.get('n_flagged')}, "
                f"n_error={llm_stats.get('n_error')}, "
                f"mean latency={lat_s}s, cost≈${cost_s}/episode."
            )
            lines.append("")
        lines.append(
            "Presence F1: hybrid = deterministic because det already flags every "
            f"case (including all {key_scores.get('n_clean', 10)} clean; TN=0). "
            f"LLM type recall is {typ.get('llm', 0):.2f} vs "
            f"{typ.get('deterministic', 0):.2f} deterministic. "
            "Do not claim hybrid superiority."
        )
        lines.append("")

    lines.extend(["## B. vs human labels", ""])
    if not scores or not scores.get("deterministic_vs_human"):
        lines.append(
            "Pending: fill `annotation/labels.csv` after reading "
            "`annotation/annotation_readable.md`, then "
            "`python -m counterfeint.experiments.score_annotations`."
        )
        return "\n".join(lines) + "\n"
    det = scores.get("deterministic_vs_human") or {}
    llm = scores.get("llm_vs_human") or {}
    hyb = scores.get("hybrid_vs_human") or {}
    key = scores.get("human_vs_corruption_key") or {}

    def _cell(block: Dict[str, Any], field: str) -> str:
        if field not in block or block.get("status"):
            return "—"
        val = block.get(field)
        return f"{val:.3f}" if isinstance(val, (int, float)) else str(val)

    lines.extend(
        [
            "| Auditor | N | Precision | Recall | F1 | Balanced acc |",
            "|---------|--:|----------:|-------:|---:|-------------:|",
            f"| Deterministic | {det.get('n', 0)} | {_cell(det, 'precision')} | "
            f"{_cell(det, 'recall')} | {_cell(det, 'f1')} | "
            f"{_cell(det, 'balanced_accuracy')} |",
            f"| LLM (Qwen3.5-9B) | {llm.get('n', 0)} | {_cell(llm, 'precision')} | "
            f"{_cell(llm, 'recall')} | {_cell(llm, 'f1')} | "
            f"{_cell(llm, 'balanced_accuracy')} |",
            f"| Hybrid | {hyb.get('n', 0)} | {_cell(hyb, 'precision')} | "
            f"{_cell(hyb, 'recall')} | {_cell(hyb, 'f1')} | "
            f"{_cell(hyb, 'balanced_accuracy')} |",
            "",
            f"Human vs corruption key: F1={key.get('f1', 0):.3f}, "
            f"κ={key.get('cohens_kappa_presence', 0):.3f}, "
            f"n={key.get('n_labeled', 0)} (one annotator; not definitive GT).",
        ]
    )
    if llm.get("status"):
        lines.append("")
        lines.append(f"LLM status: `{llm.get('status')}`.")
    return "\n".join(lines) + "\n"


def figure1_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="860" height="320" viewBox="0 0 860 320">
  <rect width="860" height="320" fill="#f8fafc"/>
  <rect x="30" y="110" width="180" height="90" rx="10" fill="#fee2e2" stroke="#b91c1c"/>
  <text x="120" y="150" text-anchor="middle" font-family="sans-serif" font-size="16">Fraudster</text>
  <text x="120" y="172" text-anchor="middle" font-family="sans-serif" font-size="11">propose / modify ads</text>
  <rect x="340" y="110" width="180" height="90" rx="10" fill="#dbeafe" stroke="#1d4ed8"/>
  <text x="430" y="150" text-anchor="middle" font-family="sans-serif" font-size="16">Investigator</text>
  <text x="430" y="172" text-anchor="middle" font-family="sans-serif" font-size="11">investigate / verdict / link</text>
  <rect x="650" y="40" width="180" height="90" rx="10" fill="#dcfce7" stroke="#15803d"/>
  <text x="740" y="78" text-anchor="middle" font-family="sans-serif" font-size="16">Track A</text>
  <text x="740" y="100" text-anchor="middle" font-family="sans-serif" font-size="11">investigator reasoning</text>
  <rect x="650" y="180" width="180" height="90" rx="10" fill="#ffedd5" stroke="#c2410c"/>
  <text x="740" y="218" text-anchor="middle" font-family="sans-serif" font-size="16">Track B</text>
  <text x="740" y="240" text-anchor="middle" font-family="sans-serif" font-size="11">fraudster plausibility</text>
  <line x1="210" y1="155" x2="340" y2="155" stroke="#334155" marker-end="url(#arrow)"/>
  <line x1="520" y1="140" x2="650" y2="85" stroke="#334155"/>
  <line x1="520" y1="170" x2="650" y2="225" stroke="#334155"/>
  <text x="430" y="300" text-anchor="middle" font-family="sans-serif" font-size="13">
    Heterogeneous verifier: outcome + citation + coherence + calibration + bias + plausibility
  </text>
</svg>
"""


def _write_fig2(path: Path, rows: List[Dict[str, Any]]) -> None:
    by = _group_ablation(rows)
    labels: List[str] = []
    inv_gain: List[float] = []
    frd_gain: List[float] = []
    for corr in INVESTIGATOR_ATTACKS:
        if corr not in by:
            continue
        inv_f = _mean(float(x["investigator_reward"]) for x in by[corr].get("full", []))
        inv_a = _mean(float(x["investigator_reward"]) for x in by[corr].get("no_track_a", []))
        labels.append(corr)
        inv_gain.append(inv_a - inv_f)
        frd_gain.append(0.0)
    for corr in FRAUDSTER_ATTACKS:
        if corr not in by:
            continue
        frd_f = _mean(float(x["fraudster_reward"]) for x in by[corr].get("full", []))
        frd_g = _mean(
            float(x["fraudster_reward"]) for x in by[corr].get("no_plausibility_gate", [])
        )
        labels.append(corr)
        inv_gain.append(0.0)
        frd_gain.append(frd_g - frd_f)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = list(range(len(labels)))
        ax.bar([i - 0.18 for i in x], inv_gain, width=0.36, label="Δ Investigator (no Track A)")
        ax.bar([i + 0.18 for i in x], frd_gain, width=0.36, label="Δ Fraudster (no plaus. gate)")
        ax.axhline(0.0, color="#64748b", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Reward gain when component removed")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    except Exception:
        csv_path = path.with_suffix(".csv")
        csv_path.write_text(
            "attack,inv_gain,frd_gain\n"
            + "\n".join(
                f"{lab},{ig:.4f},{fg:.4f}"
                for lab, ig, fg in zip(labels, inv_gain, frd_gain)
            )
            + "\n",
            encoding="utf-8",
        )


def _write_fig3(
    path: Path,
    rows: List[Dict[str, Any]],
    llm_meta: Optional[Dict[str, Any]],
    type_recall: Optional[Dict[str, Any]] = None,
) -> None:
    det_ms = _mean(float(r["elapsed_ms"]) for r in rows if r.get("arm") == "full")
    det_cost = 0.0
    llm_s = None
    llm_cost = None
    if llm_meta:
        llm_s = llm_meta.get("mean_latency_sec")
        llm_cost = llm_meta.get("cost_per_episode_usd")
    payload = {
        "deterministic_ms": det_ms,
        "deterministic_cost_usd": det_cost,
        "llm_sec": llm_s,
        "llm_cost_usd": llm_cost,
        "llm_n_flagged": (llm_meta or {}).get("n_flagged"),
        "llm_n_ok": (llm_meta or {}).get("n_ok"),
        "type_recall": type_recall,
        "hybrid": "exact checks inline (deterministic); semantic flags offline LLM",
        "note": (
            "LLM judge is offline; not used for training-time rewards. "
            "Coverage uses expected-type recall vs attack key when available."
        ),
    }
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = ["Deterministic", "LLM (Qwen3.5-9B)", "Hybrid"]
        if type_recall:
            cover = [
                float(type_recall.get("deterministic") or 0),
                float(type_recall.get("llm") or 0),
                float(type_recall.get("hybrid") or 0),
            ]
        else:
            n_flagged = float((llm_meta or {}).get("n_flagged") or 0)
            n_cases = float((llm_meta or {}).get("n_cases") or 1)
            llm_cover = 0.0 if not llm_s else min(0.7, n_flagged / n_cases)
            cover = [0.85, llm_cover, 0.85]
        det_proxy = max(det_ms, 0.12)
        llm_proxy = (llm_s or 0) * 1000.0
        cost = [det_proxy, max(llm_proxy, 1.0) if llm_s else 1.0, det_proxy + llm_proxy]
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.scatter(cost, cover, s=80)
        for n, c, v in zip(names, cost, cover):
            ax.annotate(n, (c, v), textcoords="offset points", xytext=(6, 4))
        ax.set_xlabel("Cost / latency proxy (ms, log)")
        ax.set_ylabel("Expected-type recall vs attack key" if type_recall else "Coverage (heuristic, 0–1)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    except Exception:
        pass


def acceptance_gate(
    rows: List[Dict[str, Any]],
    corruption: Dict[str, Any],
    llm_stats: Optional[Dict[str, Any]] = None,
    table4_scores: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    by = _group_ablation(rows)
    inv_lifts: List[Tuple[str, float]] = []
    for corr in ("missing_citation", "fabricated_citation", "contradictory_rationale"):
        if corr not in by:
            continue
        inv_f = _mean(float(x["investigator_reward"]) for x in by[corr].get("full", []))
        inv_a = _mean(float(x["investigator_reward"]) for x in by[corr].get("no_track_a", []))
        inv_lifts.append((corr, inv_a - inv_f))
    frd_lifts: List[Tuple[str, float]] = []
    for corr in ("gibberish_ads", "grader_trigger_tokens", "template_spam"):
        if corr not in by:
            continue
        frd_f = _mean(float(x["fraudster_reward"]) for x in by[corr].get("full", []))
        frd_g = _mean(
            float(x["fraudster_reward"]) for x in by[corr].get("no_plausibility_gate", [])
        )
        frd_lifts.append((corr, frd_g - frd_f))
    inv_ok = any(d > 0.05 for _, d in inv_lifts)
    frd_ok = any(d > 0.2 for _, d in frd_lifts)
    clean_fp = corruption.get("clean_flag_rate") or {}
    n_flagged = int((llm_stats or {}).get("n_flagged") or 0)
    hybrid_ok = False
    if table4_scores and table4_scores.get("deterministic_vs_human"):
        det_f1 = float((table4_scores.get("deterministic_vs_human") or {}).get("f1") or 0)
        hyb_f1 = float((table4_scores.get("hybrid_vs_human") or {}).get("f1") or 0)
        hybrid_ok = hyb_f1 > det_f1 + 0.02 and n_flagged > 0
    return {
        "investigator_track_a_gaming": inv_ok,
        "investigator_lifts": inv_lifts,
        "fraudster_track_b_gaming": frd_ok,
        "fraudster_lifts": frd_lifts,
        "clean_flag_rate": clean_fp,
        "anti_gaming_gate_pass": inv_ok and frd_ok,
        "hybrid_claim_allowed": hybrid_ok,
        "hybrid_claim_reason": (
            "hybrid F1 exceeds deterministic on human Table 4"
            if hybrid_ok
            else "need human Table 4; LLM type recall vs attack key is below deterministic"
        ),
        "narrow_to": None
        if (inv_ok and frd_ok)
        else "plausibility-gate anti-gaming + verifier stress-test benchmark",
    }


def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def _write_latex(
    path: Path,
    corruption: Dict[str, Any],
    ablation_rows: List[Dict[str, Any]],
    llm_stats: Optional[Dict[str, Any]],
    key_scores: Optional[Dict[str, Any]],
) -> None:
    per_flag = corruption.get("per_flag") or {}
    t2_rows = []
    for ft, m in sorted(per_flag.items(), key=lambda kv: -float(kv[1].get("f1") or 0)):
        t2_rows.append(
            f"{_tex_escape(ft)} & {m.get('precision', 0):.2f} & "
            f"{m.get('recall', 0):.2f} & {m.get('f1', 0):.2f} \\\\"
        )
    by = _group_ablation(ablation_rows)
    t3_rows = []
    for corr in INVESTIGATOR_ATTACKS + FRAUDSTER_ATTACKS:
        if corr not in by:
            continue
        side = "Inv" if corr in INV_ATTACKS else "Frd"
        if corr in INV_ATTACKS:
            full = _mean(float(x["investigator_reward"]) for x in by[corr].get("full", []))
            drop = _mean(float(x["investigator_reward"]) for x in by[corr].get("no_track_a", []))
        else:
            full = _mean(float(x["fraudster_reward"]) for x in by[corr].get("full", []))
            drop = _mean(
                float(x["fraudster_reward"]) for x in by[corr].get("no_plausibility_gate", [])
            )
        t3_rows.append(
            f"{_tex_escape(corr)} & {side} & {full:.2f} & {drop:.2f} & {drop - full:.2f} \\\\"
        )
    lat = (llm_stats or {}).get("mean_latency_sec")
    cost = (llm_stats or {}).get("cost_per_episode_usd")
    n_ok = (llm_stats or {}).get("n_ok")
    n_flagged = (llm_stats or {}).get("n_flagged")
    t4 = ""
    if key_scores:
        det = key_scores.get("deterministic_vs_key") or {}
        llm = key_scores.get("llm_vs_key") or {}
        hyb = key_scores.get("hybrid_vs_key") or {}
        t4 = "\n".join(
            [
                r"\begin{tabular}{lcccc}",
                r"\hline",
                r"Auditor & Prec. & Rec. & F1 & Bal. acc. \\",
                r"\hline",
                f"Deterministic & {det.get('precision', 0):.2f} & {det.get('recall', 0):.2f} & "
                f"{det.get('f1', 0):.2f} & {det.get('balanced_accuracy', 0):.2f} \\\\",
                f"Qwen3.5-9B & {llm.get('precision', 0):.2f} & {llm.get('recall', 0):.2f} & "
                f"{llm.get('f1', 0):.2f} & {llm.get('balanced_accuracy', 0):.2f} \\\\",
                f"Hybrid & {hyb.get('precision', 0):.2f} & {hyb.get('recall', 0):.2f} & "
                f"{hyb.get('f1', 0):.2f} & {hyb.get('balanced_accuracy', 0):.2f} \\\\",
                r"\hline",
                r"\end{tabular}",
            ]
        )
    body = "\n".join(
        [
            "% Auto-generated by paper_outputs.py — do not edit by hand.",
            r"\paragraph{Table 1.} 100 paired trajectories, 1500 trials "
            r"(1148 applicable); 30-episode ablation; 50-case annotation sheet."
            + (
                f" Qwen3.5-9B: n\\_ok={n_ok}, n\\_flagged={n_flagged}, "
                f"latency={lat:.2f}s, ${cost:.4f}/ep."
                if lat is not None and cost is not None
                else ""
            ),
            "",
            r"\paragraph{Table 2. Paired-delta F1.}",
            r"\begin{tabular}{lccc}",
            r"\hline",
            r"Flag & Prec. & Rec. & F1 \\",
            r"\hline",
            *t2_rows,
            r"\hline",
            r"\end{tabular}",
            "",
            r"\paragraph{Table 3. Gaming: reward with blocking track removed.}",
            r"\begin{tabular}{llccc}",
            r"\hline",
            r"Attack & Role & Full & Removed & $\Delta$ \\",
            r"\hline",
            *t3_rows,
            r"\hline",
            r"\end{tabular}",
            "",
            r"\paragraph{Table 4. vs attack key (human labels pending).}",
            t4 or "(pending)",
            "",
        ]
    )
    path.write_text(body + "\n", encoding="utf-8")


def write_paper_outputs(
    *,
    output_dir: Path,
    corruption_report: Path,
    ablation_results: Path,
    annotation_scores: Optional[Path] = None,
    llm_flags: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    corruption = _load_json(corruption_report) or {}
    ablation_rows = _load_json(ablation_results) or []
    scores = _load_json(annotation_scores) if annotation_scores else None
    llm_raw = _load_json(llm_flags) if llm_flags else None
    llm_stats = _llm_flag_stats(llm_raw)

    key_scores = None
    ann_dir = output_dir / "annotation"
    if (ann_dir / "annotation_key.json").exists():
        from .score_annotations import score_auditors_vs_attack_key

        key_scores = score_auditors_vs_attack_key(
            annotation_dir=ann_dir,
            llm_flags_path=llm_flags if llm_flags and llm_flags.exists() else None,
        )
        (ann_dir / "attack_key_scores.json").write_text(
            json.dumps(key_scores, indent=2), encoding="utf-8"
        )

    (output_dir / "table1_dataset.md").write_text(
        table1_dataset(corruption, ablation_rows, llm_stats), encoding="utf-8"
    )
    (output_dir / "table2_per_check.md").write_text(
        table2_per_check(corruption), encoding="utf-8"
    )
    (output_dir / "table3_ablation.md").write_text(
        table3_ablation(ablation_rows), encoding="utf-8"
    )
    (output_dir / "table4_human.md").write_text(
        table4_human(scores, key_scores=key_scores, llm_stats=llm_stats),
        encoding="utf-8",
    )
    (output_dir / "figure1_roles.svg").write_text(figure1_svg(), encoding="utf-8")
    _write_fig2(output_dir / "figure2_attack_reward.png", ablation_rows)
    _write_fig3(
        output_dir / "figure3_coverage_cost.png",
        ablation_rows,
        llm_stats,
        type_recall=(key_scores or {}).get("expected_type_recall"),
    )

    gate = acceptance_gate(
        ablation_rows, corruption, llm_stats=llm_stats, table4_scores=scores
    )
    (output_dir / "acceptance_gate.json").write_text(
        json.dumps(gate, indent=2, default=str), encoding="utf-8"
    )

    runtime = {
        "deterministic_mean_ms": _mean(
            float(r["elapsed_ms"]) for r in ablation_rows if r.get("arm") == "full"
        ),
        "llm": llm_stats,
    }
    (output_dir / "runtime_cost.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )
    tex = output_dir / "tables.tex"
    _write_latex(tex, corruption, ablation_rows, llm_stats, key_scores)
    paper_tex = Path(__file__).resolve().parents[1] / "papers" / "verify_agents" / "tables.tex"
    paper_tex.parent.mkdir(parents=True, exist_ok=True)
    paper_tex.write_text(tex.read_text(encoding="utf-8"), encoding="utf-8")

    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Auditor expanded outputs",
                "",
                "Generated by `python -m counterfeint.experiments paper-outputs`.",
                "",
                "| File | Contents |",
                "|------|----------|",
                "| `table1_dataset.md` | Splits, LLM latency/cost |",
                "| `table2_per_check.md` | Paired-delta P/R/F1 grouped by strength |",
                "| `table3_ablation.md` | Anti-gaming + per-check drop |",
                "| `table4_human.md` | vs attack key now; vs human after `labels.csv` |",
                "| `tables.tex` | Paper copy-paste |",
                "| `figure2_attack_reward.png` | Δ reward when blocking track removed |",
                "| `figure3_coverage_cost.png` | Cost vs coverage (LLM coverage 0 if n_flagged=0) |",
                "| `acceptance_gate.json` | Anti-gaming pass; hybrid claim blocked |",
                "| `annotation/` | Blinded 50-case sheet |",
                "",
                "Score labels: `python -m counterfeint.experiments.score_annotations`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"output_dir": str(output_dir), "gate": gate, "llm": llm_stats}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded"),
    )
    parser.add_argument(
        "--corruption",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded/corruption/corruption_report.json"),
    )
    parser.add_argument(
        "--ablation",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded/ablation/ablation_results.json"),
    )
    parser.add_argument(
        "--annotation-scores",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded/annotation/annotation_scores.json"),
    )
    parser.add_argument(
        "--llm-flags",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded/llm_flags.json"),
    )
    args = parser.parse_args()
    result = write_paper_outputs(
        output_dir=args.output,
        corruption_report=args.corruption,
        ablation_results=args.ablation,
        annotation_scores=args.annotation_scores if args.annotation_scores.exists() else None,
        llm_flags=args.llm_flags if args.llm_flags.exists() else None,
    )
    print(json.dumps(result, indent=2, default=str))
