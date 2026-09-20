"""Detector interface (plan section 7) with a deterministic CPU fake for M01 smoke.

`predict_spans` / `score_target` are the stable seams the real ModernBERT-based D0
(M03) will plug into. The fake predicts nothing; all its outputs are labeled MOCK
in run reports and must never appear in scientific results.
"""

from __future__ import annotations


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
