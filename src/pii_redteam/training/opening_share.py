"""Collapse guard for the RL reward: how many other requests' texts start with the same word.

The first long run (long-20261114-004948) drove the attacker to open every text with "Dear",
whatever the request. Measured on its samples, the share of texts written for OTHER requests
that start with the same word was at most 0.15 before RL and 0.99 after. A text above
`threshold` is flagged; texts at the base attacker's level are never flagged.
"""

from __future__ import annotations

import re
from collections import Counter, deque

from ..probe_metrics import name_spans

_WORD = re.compile(r"[a-z']+")


def opening_word(text: str, name: str) -> str:
    """First word of `text` once the requested name is removed, lower-cased ("" if none)."""
    for start, end in reversed(name_spans(text, name)):
        text = text[:start] + " " + text[end:]
    match = _WORD.search(text.lower())
    return match.group(0) if match else ""


class OpeningShare:
    """Rolling window of (request, opening word) over the most recent valid texts."""

    def __init__(self, threshold: float, window: int, min_other_texts: int) -> None:
        self.threshold = threshold
        self.min_other_texts = min_other_texts
        self.recent: deque[tuple[str, str]] = deque(maxlen=window)

    def shares(self, requests: list[str], openings: list[str]) -> list[float | None]:
        """Per text: share of the window's texts for other requests with the same opening word.

        The window includes this batch. None while fewer than `min_other_texts` texts from
        other requests are available, so a handful of texts cannot trigger the guard.
        """
        self.recent.extend(zip(requests, openings, strict=True))
        per_request = Counter(request for request, _ in self.recent)
        per_opening = Counter(opening for _, opening in self.recent)
        per_pair = Counter(self.recent)
        shares: list[float | None] = []
        for request, opening in zip(requests, openings, strict=True):
            others = len(self.recent) - per_request[request]
            if others < self.min_other_texts:
                shares.append(None)
            else:
                shares.append((per_opening[opening] - per_pair[(request, opening)]) / others)
        return shares

    def is_repeated(self, share: float | None) -> bool:
        return share is not None and share > self.threshold
