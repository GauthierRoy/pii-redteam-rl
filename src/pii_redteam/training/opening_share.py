"""Collapse guard for the RL reward: how many other requests' texts start with the same word.

The first long run (long-20261114-004948) drove the attacker to open every text with "Dear",
whatever the request. Measured on its samples, the share of texts written for OTHER requests
that start with the same word was at most 0.15 before RL and 0.99 after.

A text keeps its full reward up to a share of `full_reward_until` and none from
`zero_reward_from`, with a straight line in between. A hard cut at 0.20 was tried first
(pilot-opening-20261114-103332): in training the base attacker's share swings around 0.15
from window to window, so 14-29% of valid texts lost their reward and nothing was learned.
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

    def __init__(
        self, full_reward_until: float, zero_reward_from: float, window: int, min_other_texts: int
    ) -> None:
        if not 0.0 <= full_reward_until < zero_reward_from <= 1.0:
            raise ValueError("need 0 <= full_reward_until < zero_reward_from <= 1")
        self.full_reward_until = full_reward_until
        self.zero_reward_from = zero_reward_from
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

    def reward_factor(self, share: float | None) -> float:
        """1.0 up to `full_reward_until`, falling linearly to 0.0 at `zero_reward_from`."""
        if share is None:
            return 1.0
        span = self.zero_reward_from - self.full_reward_until
        return min(1.0, max(0.0, (self.zero_reward_from - share) / span))
