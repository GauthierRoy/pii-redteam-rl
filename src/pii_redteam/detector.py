"""Detector interface (plan section 7) with a deterministic CPU fake for M01 smoke.

`predict_spans` / `score_target` are the stable seams the real ModernBERT-based D0
(M03) will plug into. The fake predicts nothing; all its outputs are labeled MOCK
in run reports and must never appear in scientific results.

The alignment helpers below are tokenizer-agnostic (M03.2): they operate on token
offset sequences in Python character indices, end-exclusive — the interface HF fast
tokenizers expose via ``return_offsets_mapping=True``. The real ModernBERT
tokenizer plugs in at the Colab gate (M03.1); mBERT token/BIO arrays are never
reused (plan §4.2).
"""

from __future__ import annotations

PERSON = "PERSON"


def bio_labels_from_spans(offsets: list[tuple[int, int]], spans: list[dict]) -> list[str]:
    """Token-level BIO labels for the given char spans.

    ``offsets[i]`` is the (start, end) char range of token ``i``; zero-width
    offsets (e.g. special tokens) are always labeled ``O``. A token whose range
    overlaps a span is labeled ``B-<label>`` for the first overlapping token of
    that span and ``I-<label>`` for the rest. Spans are assumed non-overlapping
    and sorted-agnostic; boundaries landing mid-token label the whole token.
    """
    labels = ["O"] * len(offsets)
    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        first = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue
            if tok_start < span["end"] and tok_end > span["start"]:
                labels[i] = ("B-" if first else "I-") + span["label"]
                first = False
    return labels


def spans_from_bio_labels(offsets: list[tuple[int, int]], labels: list[str]) -> list[dict]:
    """Decode BIO labels back to char spans (conlleval-style).

    A ``B-`` or orphan ``I-`` starts a span; ``I-`` with the same label extends
    it. Char recovery is token-granular: the span covers the full char range of
    its tokens. Zero-width tokens never open or extend a span.
    """
    if len(offsets) != len(labels):
        raise ValueError(f"{len(offsets)} offsets vs {len(labels)} labels")
    spans: list[dict] = []
    current: list | None = None  # [label, start, end]
    for (tok_start, tok_end), label in zip(offsets, labels, strict=True):
        if label == "O" or tok_start == tok_end:
            if current is not None:
                spans.append({"label": current[0], "start": current[1], "end": current[2]})
                current = None
            continue
        prefix, _, label_name = label.partition("-")
        if prefix not in ("B", "I") or not label_name:
            raise ValueError(f"invalid BIO label {label!r}")
        if prefix == "B" or current is None or current[0] != label_name:
            if current is not None:
                spans.append({"label": current[0], "start": current[1], "end": current[2]})
            current = [label_name, tok_start, tok_end]
        else:
            current[2] = tok_end
    if current is not None:
        spans.append({"label": current[0], "start": current[1], "end": current[2]})
    return spans


def truncate_labels(labels: list[str], max_length: int) -> list[str]:
    """Drop labels beyond ``max_length`` tokens (truncation handling, M03.2)."""
    if max_length < 0:
        raise ValueError("max_length must be >= 0")
    return labels[:max_length]


class FakeDetector:
    """Frozen mock detector: finds no spans, scores 0.0 correctness."""

    def __init__(self, model_id: str = "fake-detector-001") -> None:
        self.model_id = model_id
        self.frozen = True

    def predict_spans(self, text: str) -> list[dict]:
        """Return predicted PERSON spans as [{label, start, end}]. Mock: always []."""
        _ = text
        return []

    def score_target(self, text: str, start: int, end: int) -> float:
        """Return P(correct labels) over the target span. Mock: always 0.0."""
        _, _, _ = text, start, end
        return 0.0
