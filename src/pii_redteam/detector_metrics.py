"""Pure (torch-free) logic for detector fine-tuning (`train_detector`): data, selection, metrics.

"d0" is the starting detector and "d1" the fine-tuned one (M07b names, kept in the reports).

The checkpoint is selected on dev_calibration + a name-disjoint hold-out of
d1_aug_v1. short_eval and final-eval are only scored after selection (M07b.1).
"""

from __future__ import annotations

import json
import random

from pii_redteam.probe_metrics import outcome_for_sample

OUTCOMES = ("exact_hit", "partial_overlap", "boundary_error", "complete_miss")


def load_synthetic_rows(path: str) -> list[dict]:
    """Authored rows -> detector rows. Raises if a stored span is not the entity value."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            entities = rec["request"]["entities"]
            if len(entities) != 1:
                raise ValueError(f"{rec['id']}: expected one entity, got {len(entities)}")
            value = entities[0]["value"]
            text = rec["target"]["text"]
            start, end = rec["target"]["span"]
            if text[start:end] != value:
                raise ValueError(f"{rec['id']}: span {start}:{end} is not {value!r}")
            rows.append(
                {
                    "example_id": rec["id"],
                    "text": text,
                    "spans": [{"label": "PERSON", "start": start, "end": end}],
                    "name": value,
                }
            )
    return rows


def load_negative_rows(path: str) -> list[dict]:
    """Name-free short messages -> detector rows with no gold span.

    Same schema as the short_eval files, but `request.entities == []`,
    `target.span is None`, and a top-level `distractor` type (place, org_product, ...). Any row that names an entity or carries a span is rejected:
    a negative set with a labeled person in it would count true hits as false positives.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            entities = rec["request"]["entities"]
            if entities != []:
                raise ValueError(f"{rec['id']}: a negative row must have no entities")
            if rec["target"].get("span") is not None:
                raise ValueError(f"{rec['id']}: a negative row must have target.span == null")
            text = rec["target"]["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{rec['id']}: empty text")
            distractor = rec.get("distractor")
            if not isinstance(distractor, str) or not distractor:
                raise ValueError(f"{rec['id']}: a negative row needs a 'distractor' type")
            rows.append(
                {
                    "example_id": rec["id"],
                    "text": text,
                    "spans": [],
                    "name": None,
                    "distractor": distractor,
                }
            )
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def negative_fp_metrics(rows: list[dict], preds: list[list[tuple[int, int]]]) -> dict:
    """False positives on name-free rows: rows with any predicted span, and span count."""
    if len(rows) != len(preds):
        raise ValueError(f"{len(rows)} rows but {len(preds)} predictions")
    if any(row["spans"] for row in rows):
        raise ValueError("negative_fp_metrics takes name-free rows only")
    flagged = [
        {
            "id": row["example_id"],
            "text": row["text"],
            "spans": [[s, e, row["text"][s:e]] for s, e in sorted(set(pred))],
        }
        for row, pred in zip(rows, preds, strict=True)
        if pred
    ]
    return {
        "rows": len(rows),
        "rows_with_pred": len(flagged),
        "pred_spans": sum(len(set(pred)) for pred in preds),
        "fp_row_rate": round(len(flagged) / len(rows), 4) if rows else None,
        "flagged": sorted(flagged, key=lambda f: f["id"]),
    }


def split_by_name(rows: list[dict], dev_fraction: float, seed: int) -> tuple[list, list]:
    """Seeded train/dev split with no exact name (case-folded) in both parts."""
    by_name: dict[str, list[dict]] = {}
    for row in rows:
        by_name.setdefault(row["name"].casefold(), []).append(row)
    names = sorted(by_name)
    random.Random(seed).shuffle(names)
    target = round(len(rows) * dev_fraction)
    dev_names: set[str] = set()
    dev_count = 0
    for name in names:
        if dev_count >= target:
            break
        dev_names.add(name)
        dev_count += len(by_name[name])
    train = [r for r in rows if r["name"].casefold() not in dev_names]
    dev = [r for r in rows if r["name"].casefold() in dev_names]
    return train, dev


def span_metrics(rows: list[dict], preds: list[list[tuple[int, int]]]) -> dict:
    """Exact-span P/R/F1, per-target outcomes, and predictions touching no gold span.

    Outcome names and rules match the probe (`outcome_for_sample`), so complete_miss
    here is the same quantity as the M07a go/no-go miss rate.
    """
    if len(rows) != len(preds):
        raise ValueError(f"{len(rows)} rows but {len(preds)} predictions")
    tp = fp = fn = neg_rows = neg_fp = neg_rows_with_pred = extra_spans = rows_with_extra = 0
    outcomes = dict.fromkeys(OUTCOMES, 0)
    per_row: dict[str, str] = {}
    for row, pred in zip(rows, preds, strict=True):
        gold = [(sp["start"], sp["end"]) for sp in row["spans"]]
        pred_set, gold_set = set(pred), set(gold)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
        if not gold:
            neg_rows += 1
            neg_fp += len(pred_set)
            neg_rows_with_pred += bool(pred_set)
        extra = [p for p in pred_set if not any(p[0] < g[1] and p[1] > g[0] for g in gold_set)]
        extra_spans += len(extra)
        rows_with_extra += bool(extra)
        for span in gold:
            outcome = outcome_for_sample(span, list(pred_set))
            outcomes[outcome] += 1
            per_row[row["example_id"]] = outcome
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n_targets = sum(outcomes.values())
    return {
        "rows": len(rows),
        "span_precision": round(precision, 4),
        "span_recall": round(recall, 4),
        "span_f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "outcomes": outcomes,
        "complete_miss_rate": round(outcomes["complete_miss"] / n_targets, 4)
        if n_targets
        else None,
        "neg_rows": neg_rows,
        "neg_false_positives": neg_fp,
        "neg_rows_with_pred": neg_rows_with_pred,
        "extra_pred_spans": extra_spans,
        "rows_with_extra_pred": rows_with_extra,
        "per_row_outcome": per_row,
    }


def strip_per_row(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "per_row_outcome"}


def is_better_candidate(
    dev_f1: float,
    aug_dev_f1: float,
    best_aug_dev_f1: float | None,
    d0_dev_f1: float,
    max_drop: float,
) -> bool:
    """Eligible only within max_drop of D0's dev F1; then the highest aug-dev F1 wins.

    Ties keep the earlier epoch.
    """
    if dev_f1 < d0_dev_f1 - max_drop:
        return False
    return best_aug_dev_f1 is None or aug_dev_f1 > best_aug_dev_f1


def paired_changes(d0_metrics: dict, d1_metrics: dict) -> dict:
    """Same rows under both detectors: exact hits gained and lost by D1."""
    d0, d1 = d0_metrics["per_row_outcome"], d1_metrics["per_row_outcome"]
    if d0.keys() != d1.keys():
        raise ValueError("paired comparison needs the same rows for both detectors")
    gained = sorted(k for k in d0 if d0[k] != "exact_hit" and d1[k] == "exact_hit")
    lost = sorted(k for k in d0 if d0[k] == "exact_hit" and d1[k] != "exact_hit")
    return {"gained_exact": len(gained), "lost_exact": len(lost), "lost_exact_ids": lost}


def short_eval_gate(d0_metrics: dict, d1_metrics: dict, min_gain: float) -> dict:
    gain = round(d1_metrics["span_recall"] - d0_metrics["span_recall"], 4)
    return {"recall_gain": gain, "min_gain": min_gain, "ok": gain >= min_gain}


def no_regression_gate(d0_metrics: dict, d1_metrics: dict, max_drop: float) -> dict:
    precision_drop = round(d0_metrics["span_precision"] - d1_metrics["span_precision"], 4)
    recall_drop = round(d0_metrics["span_recall"] - d1_metrics["span_recall"], 4)
    return {
        "precision_drop": precision_drop,
        "recall_drop": recall_drop,
        "max_drop": max_drop,
        "ok": precision_drop <= max_drop and recall_drop <= max_drop,
    }


FREEZE_RULE = "fewest_dev_neg_fp_rows"


def select_freeze_epoch(candidates: list[dict], d0_dev_f1: float, max_drop: float) -> dict:
    """Pre-registered freeze rule over saved epochs, from DEV (dev_calibration) data only.

    Each candidate is {"epoch", "dev_span_f1", "dev_neg_rows_with_pred"}. Eligible: dev
    span-F1 >= D0's - max_drop. Among eligible, fewest name-free dev rows with a predicted
    span wins; tie -> higher dev span-F1; tie -> earlier epoch. Any other field in a
    candidate is ignored, so nothing scored later (final-eval, probe) can change the pick.
    """
    keyed = [
        {
            "epoch": int(c["epoch"]),
            "dev_span_f1": c["dev_span_f1"],
            "dev_neg_rows_with_pred": c["dev_neg_rows_with_pred"],
        }
        for c in candidates
    ]
    if len({c["epoch"] for c in keyed}) != len(keyed):
        raise ValueError("duplicate epochs among candidates")
    eligible = [c for c in keyed if round(d0_dev_f1 - c["dev_span_f1"], 4) <= max_drop]
    ranked = sorted(
        eligible, key=lambda c: (c["dev_neg_rows_with_pred"], -c["dev_span_f1"], c["epoch"])
    )
    return {
        "rule": FREEZE_RULE,
        "d0_dev_f1": d0_dev_f1,
        "dev_f1_max_drop": max_drop,
        "candidates": sorted(keyed, key=lambda c: c["epoch"]),
        "eligible_epochs": sorted(c["epoch"] for c in eligible),
        "chosen_epoch": ranked[0]["epoch"] if ranked else None,
    }
