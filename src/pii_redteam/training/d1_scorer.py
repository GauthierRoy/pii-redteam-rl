"""Frozen-detector scorer for the RL reward (M08.2): the probe's D0/D1 scoring path.

Wraps `probe._score_with_d0` unchanged so RL scoring equals the validated M07a scoring.
The detector is switched to eval mode and every parameter is frozen at construction.
"""

from __future__ import annotations

from ..probe import _score_with_d0


class D1Scorer:
    def __init__(self, model, tokenizer, max_length: int = 512) -> None:
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def score(self, samples: list[dict]) -> None:
        _score_with_d0(self.model, self.tokenizer, samples, self.max_length)
