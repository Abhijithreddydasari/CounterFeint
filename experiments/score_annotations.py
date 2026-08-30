"""Score human labels against the hidden annotation key and auditor flags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from counterfeint.models import AuditFlag

from .annotation_sheet import (
    load_labels,
    normalize_label_row,
    score_against_key,
    _row_is_labeled,
)
from .episode_bundle import load_bundle
from .llm_auditor import merge_hybrid
from .reward_arms import run_audit


def _flags_to_presence(flags: List[AuditFlag]) -> bool:
    return any((f.severity or 0) >= 0.4 for f in flags)


def _items_to_flags(raw: Any) -> List[AuditFlag]:
    flags: List[AuditFlag] = []
    if not isinstance(raw, list):
        return flags
    for item in raw:
        if not isinstance(item, dict) or "error" in item:
            continue
        flags.append(
            AuditFlag(
                track=item.get("track", "A"),
                target_ad_id=item.get("target_ad_id"),
                flag_type=item.get("flag_type", "missing_citation"),
                severity=float(item.get("severity", 0.5) or 0.5),
                note=str(item.get("note") or "")[:2000],
            )
        )
    return flags


def _presence_metrics(gold: Dict[str, bool], pred: Dict[str, bool]) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for cid, g in gold.items():
        p = bool(pred.get(cid, False))
        if g and p:
            tp += 1
        elif p and not g:
            fp += 1
        elif g and not p:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(0.5 * (tpr + tnr), 4),
    }


def score_auditors_vs_attack_key(
    *,
    annotation_dir: Path,
    llm_flags_path: Path | None = None,
) -> Dict[str, Any]:
    """Score auditors against the hidden corruption key (not human GT)."""
    key_rows = json.loads((annotation_dir / "annotation_key.json").read_text(encoding="utf-8"))
    llm_payload: Dict[str, Any] = {}
    if llm_flags_path and llm_flags_path.exists():
        raw = json.loads(llm_flags_path.read_text(encoding="utf-8"))
        llm_payload = raw.get("flags") or raw

    gold: Dict[str, bool] = {}
    det_pred: Dict[str, bool] = {}
    llm_pred: Dict[str, bool] = {}
    hyb_pred: Dict[str, bool] = {}
    type_hits = {"det": 0, "llm": 0, "hybrid": 0, "n_pos": 0}

    for row in key_rows:
        cid = row["case_id"]
        gold[cid] = not bool(row.get("is_clean"))
        bundle_path = annotation_dir / "bundles" / f"{cid}.json"
        if not bundle_path.exists():
            continue
        bundle = load_bundle(bundle_path)
        audit = run_audit(bundle)
        det_flags = list(audit.track_a_flags) + list(audit.track_b_flags)
        llm_flags = _items_to_flags(llm_payload.get(cid) or [])
        hyb_flags = merge_hybrid(det_flags, llm_flags)
        det_pred[cid] = _flags_to_presence(det_flags)
        llm_pred[cid] = _flags_to_presence(llm_flags)
        hyb_pred[cid] = _flags_to_presence(hyb_flags)
        expected = str(row.get("expected_issue_type") or "none")
        if expected != "none":
            type_hits["n_pos"] += 1
            if any(f.flag_type == expected for f in det_flags):
                type_hits["det"] += 1
            if any(f.flag_type == expected for f in llm_flags):
                type_hits["llm"] += 1
            if any(f.flag_type == expected for f in hyb_flags):
                type_hits["hybrid"] += 1

    n_pos = type_hits["n_pos"] or 1
    return {
        "reference": "attack_key",
        "n_cases": len(gold),
        "n_clean": sum(1 for v in gold.values() if not v),
        "deterministic_vs_key": _presence_metrics(gold, det_pred),
        "llm_vs_key": _presence_metrics(gold, llm_pred),
        "hybrid_vs_key": _presence_metrics(gold, hyb_pred),
        "expected_type_recall": {
            "deterministic": round(type_hits["det"] / n_pos, 4),
            "llm": round(type_hits["llm"] / n_pos, 4),
            "hybrid": round(type_hits["hybrid"] / n_pos, 4),
            "n_positive": type_hits["n_pos"],
        },
        "note": (
            "Attack-key presence, not human GT. Scripted clean cases still fire "
            "many deterministic flags (see Table 2 clean-control rates)."
        ),
    }


def score_auditors_vs_human(
    *,
    labels: List[Dict[str, str]],
    annotation_dir: Path,
    llm_flags_path: Path | None = None,
) -> Dict[str, Any]:
    key_rows = json.loads((annotation_dir / "annotation_key.json").read_text(encoding="utf-8"))
    key = {row["case_id"]: row for row in key_rows}
    llm_payload: Dict[str, Any] = {}
    if llm_flags_path and llm_flags_path.exists():
        raw = json.loads(llm_flags_path.read_text(encoding="utf-8"))
        llm_payload = raw.get("flags") or raw

    human_any: Dict[str, bool] = {}
    human_a: Dict[str, bool] = {}
    human_b: Dict[str, bool] = {}
    for row in labels:
        cid = row.get("case_id", "")
        if not _row_is_labeled(row):
            continue
        norm = normalize_label_row(row)
        human_any[cid] = bool(norm["issue_present"])
        human_a[cid] = bool(norm["track_a_present"])
        human_b[cid] = bool(norm["track_b_present"])

    det_any: Dict[str, bool] = {}
    det_a: Dict[str, bool] = {}
    det_b: Dict[str, bool] = {}
    llm_any: Dict[str, bool] = {}
    llm_a: Dict[str, bool] = {}
    llm_b: Dict[str, bool] = {}
    hyb_any: Dict[str, bool] = {}
    hyb_a: Dict[str, bool] = {}
    hyb_b: Dict[str, bool] = {}
    for cid in human_any:
        bundle_path = annotation_dir / "bundles" / f"{cid}.json"
        if not bundle_path.exists():
            continue
        bundle = load_bundle(bundle_path)
        audit = run_audit(bundle)
        det_flags = list(audit.track_a_flags) + list(audit.track_b_flags)
        det_a[cid] = _flags_to_presence([f for f in det_flags if f.track == "A"])
        det_b[cid] = _flags_to_presence([f for f in det_flags if f.track == "B"])
        det_any[cid] = det_a[cid] or det_b[cid]
        llm_raw = llm_payload.get(cid) or llm_payload.get(
            f"{bundle.task_id}:{bundle.seed}:{bundle.corruption_label or 'clean'}"
        ) or []
        llm_flags = _items_to_flags(llm_raw)
        llm_a[cid] = _flags_to_presence([f for f in llm_flags if f.track == "A"])
        llm_b[cid] = _flags_to_presence([f for f in llm_flags if f.track == "B"])
        llm_any[cid] = llm_a[cid] or llm_b[cid]
        hyb = merge_hybrid(det_flags, llm_flags)
        hyb_a[cid] = _flags_to_presence([f for f in hyb if f.track == "A"])
        hyb_b[cid] = _flags_to_presence([f for f in hyb if f.track == "B"])
        hyb_any[cid] = hyb_a[cid] or hyb_b[cid]

    return {
        "human_vs_corruption_key": score_against_key(
            labels=labels, key_path=annotation_dir / "annotation_key.json"
        ),
        "deterministic_vs_human": _presence_metrics(human_any, det_any),
        "llm_vs_human": _presence_metrics(human_any, llm_any) if llm_flags_path else {
            "n": 0,
            "status": "pending_llm_run",
        },
        "hybrid_vs_human": _presence_metrics(human_any, hyb_any),
        "track_a": {
            "deterministic_vs_human": _presence_metrics(human_a, det_a),
            "llm_vs_human": _presence_metrics(human_a, llm_a),
            "hybrid_vs_human": _presence_metrics(human_a, hyb_a),
        },
        "track_b": {
            "deterministic_vs_human": _presence_metrics(human_b, det_b),
            "llm_vs_human": _presence_metrics(human_b, llm_b),
            "hybrid_vs_human": _presence_metrics(human_b, hyb_b),
        },
        "n_key_cases": len(key),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Score Auditor annotation labels")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=Path("experiments/outputs/auditor_expanded/annotation"),
    )
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--llm-flags", type=Path, default=None)
    args = parser.parse_args()
    labels_path = args.labels or (args.annotation_dir / "labels.csv")
    if not labels_path.exists():
        print(json.dumps({"status": "pending_human_labels", "path": str(labels_path)}))
        return
    labels = load_labels(labels_path)
    payload = score_auditors_vs_human(
        labels=labels,
        annotation_dir=args.annotation_dir,
        llm_flags_path=args.llm_flags,
    )
    out = args.annotation_dir / "annotation_scores.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
