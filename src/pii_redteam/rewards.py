"""Reward interface, M01 stub (full spec in M07, `docs/REWARD_SPEC.md` later).

Contract honored from the start: invalid samples score below the valid range and
every component (validity / difficulty / repetition) is logged separately instead
of being folded into one opaque number.
"""

from __future__ import annotations

INVALID_TOTAL = -1.0
VALID_MIN, VALID_MAX = 0.0, 1.0


def repetition_penalty(text: str) -> float:
    """Tiny type-token-ratio penalty. 0.0 for normal text, -0.2 for boilerplate loops."""
    tokens = text.split()
    if len(tokens) < 8:
        return 0.0
    ttr = len(set(tokens)) / len(tokens)
    return -0.2 if ttr < 0.3 else 0.0


def score_candidate(*, text: str, supplied_name: str, detector, max_len: int = 500) -> dict:
    """Score one generated positive. `detector` provides `score_target` (0..1)."""
    from .validation import validate_candidate

    valid, reason = validate_candidate(text=text, supplied_name=supplied_name, max_len=max_len)
    if not valid:
        return {
            "valid": False,
            "reason": reason,
            "validity": 0.0,
            "difficulty": 0.0,
            "repetition": 0.0,
            "total": INVALID_TOTAL,
        }
    start = text.index(supplied_name)
    end = start + len(supplied_name)
    difficulty = min(
        VALID_MAX, max(VALID_MIN, 1.0 - float(detector.score_target(text, start, end)))
    )
    return {
        "valid": True,
        "reason": reason,
        "validity": 1.0,
        "difficulty": difficulty,
        "repetition": repetition_penalty(text),
        "total": difficulty,
    }
