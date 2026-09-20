"""Span metrics on fixtures (M04 stub: exact-span P/R/F1; overlap-based later)."""

from __future__ import annotations

from collections import Counter


def _key(span: dict) -> tuple:
    return (span.get("label"), span.get("start"), span.get("end"))


def exact_span_prf(pred: list[dict], gold: list[dict]) -> dict:
    """Exact (label, start, end) precision/recall/F1 with multiset counting."""
    pred_counts = Counter(_key(s) for s in pred)
    gold_counts = Counter(_key(s) for s in gold)
    tp = sum((pred_counts & gold_counts).values())
    precision = tp / sum(pred_counts.values()) if pred_counts else 0.0
    recall = tp / sum(gold_counts.values()) if gold_counts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "precision": precision, "recall": recall, "f1": f1}
