"""M07b prompt variants (P0/P1/P2) and pure comparison metrics.

Variants, all on the SAME v2 request bank and the SAME JSON output contract
("Return only a JSON object with a 'text' field."):

  P0  the current (v1) prompt — baseline; v2 request fields are ignored
  P1  P0 + context_type + target_length (+ name_position hint when present)
  P2  P1 + two short few-shot examples of different context types, with fixed
      fictional names that are in NO name pool, name never at the start

The comparison metrics below are pure functions (no GPU): v2 validity rate +
failure breakdown, share of outputs starting with the name, name-position
distribution, top-10 opening-pattern share (first 5 words), distinct-1/2,
median unique valid outputs per group of n, and length distribution.
"""

from __future__ import annotations

import re
from collections import Counter

from pii_redteam.probe_metrics import distinct_n, normalize
from pii_redteam.requests import (
    CONTEXT_TYPE_LABELS,
    NAME_POSITION_LABELS,
    render_prompt,
)

PROMPT_VARIANTS = ("P0", "P1", "P2")

# Fixed fictional few-shot names for P2. Verified against every name pool:
# not in train_side_pool, not held-out, not in final-eval rows (M07b rule).
FEWSHOT_NAMES = ("Dorotea Lunvik", "Keferis Onda")

# Two short examples, different context types, name NOT at the start.
FEWSHOT_EXAMPLES = (
    (
        "chat_message",
        '{"text": "Quick heads-up - rehearsal moved to Thursday 7pm at the studio, and '
        'Dorotea Lunvik asked everyone to bring the new scores."}',
    ),
    (
        "support_ticket",
        '{"text": "Order #48213 has not shipped after ten days and the tracking page '
        "errors out; customer requests a refund to the original card, filed by "
        'Keferis Onda on behalf of the account holder."}',
    ),
)


def render_prompt_v2(record: dict, variant: str) -> str:
    """Render the user prompt for a v2 request under a given variant."""
    if variant == "P0":
        return render_prompt(record)
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")

    task = record["task"]
    lines = [render_prompt(record)]
    lines.append(f"Context type: {CONTEXT_TYPE_LABELS[task['context_type']]}")
    length = task["target_length"]
    lines.append(f"Target length: {length['min_words']}-{length['max_words']} words.")
    name_position = task.get("name_position")
    if name_position:
        lines.append(f"Place the name {NAME_POSITION_LABELS[name_position]}.")
    if variant == "P2":
        lines.append("Two examples of the style (different context types):")
        for context_type, example in FEWSHOT_EXAMPLES:
            lines.append(f"- {CONTEXT_TYPE_LABELS[context_type]}: {example}")
    return "\n".join(lines)


# ---------------------------------------------------------------- metrics


def opens_with_name(text: str, target_span: tuple[int, int]) -> bool:
    """True when nothing but whitespace precedes the requested name."""
    return not text[: target_span[0]].strip()


def opening_pattern(text: str, n_words: int = 5) -> str:
    """Normalized first-n-words pattern used for template-collapse counting."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    return " ".join(words[:n_words])


def name_position_bin(span: tuple[int, int], text: str) -> str:
    """start / middle / end by the name's midpoint fraction of the text."""
    if not text:
        return "start"
    frac = (span[0] + span[1]) / 2 / len(text)
    if frac < 0.25:
        return "start"
    if frac < 0.75:
        return "middle"
    return "end"


def comparison_metrics(samples: list[dict], group_size: int = 8) -> dict:
    """One (variant, temperature) cell of the M07b comparison.

    Each sample dict needs: status (v2), text, target_span (v2), group_id,
    request_id, name. Returns the metric set of the M07b spec.
    """
    valid = [s for s in samples if s["status"] == "valid"]
    out: dict = {
        "total": len(samples),
        "valid": len(valid),
        "validity_rate": round(len(valid) / len(samples), 4) if samples else 0.0,
    }
    failures = Counter(s["status"] for s in samples if s["status"] != "valid")
    out["failure_breakdown"] = dict(sorted(failures.items()))

    if not valid:
        return out | {
            "name_at_start_share": 0.0,
            "name_position": {},
            "top10_opening_share": 0.0,
            "top10_openings": [],
            "distinct1": 0.0,
            "distinct2": 0.0,
            "unique_per_group_median": 0,
            "length_words": {},
        }

    n = len(valid)
    out["name_at_start_share"] = round(
        sum(1 for s in valid if opens_with_name(s["text"], s["target_span"])) / n, 4
    )
    positions = Counter(name_position_bin(s["target_span"], s["text"]) for s in valid)
    out["name_position"] = {
        k: round(positions.get(k, 0) / n, 4) for k in ("start", "middle", "end")
    }

    openings = Counter(opening_pattern(s["text"]) for s in valid)
    top10 = openings.most_common(10)
    out["top10_opening_share"] = round(sum(c for _, c in top10) / n, 4)
    out["top10_openings"] = top10

    word_lists = [s["text"].split() for s in valid]
    out["distinct1"] = round(distinct_n(word_lists, 1), 4)
    out["distinct2"] = round(distinct_n(word_lists, 2), 4)

    groups: dict[str, set[str]] = {}
    for s in valid:
        groups.setdefault(s["group_id"], set()).add(normalize(s["text"]))
    uniq = sorted(len(g) for g in groups.values())
    out["unique_per_group_median"] = uniq[len(uniq) // 2]
    out["unique_per_group_min"] = uniq[0]
    out["groups"] = len(groups)

    words = sorted(len(w) for w in word_lists)
    out["length_words"] = {
        "p10": words[int(n * 0.1)],
        "p50": words[n // 2],
        "p90": words[min(n - 1, int(n * 0.9))],
        "min": words[0],
        "max": words[-1],
    }
    return out


def variant_threshold_report(metrics: dict, thresholds: dict) -> dict:
    """M07b pass criteria (pre-committed before the run)."""

    def check(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"ok": bool(ok), "detail": detail}

    checks: dict[str, dict] = {}
    check(
        "validity",
        metrics.get("validity_rate", 0.0) >= thresholds["validity_min"],
        f"validity_rate={metrics.get('validity_rate')} (min {thresholds['validity_min']})",
    )
    check(
        "name_at_start",
        metrics.get("name_at_start_share", 1.0) <= thresholds["name_at_start_max"],
        f"name_at_start_share={metrics.get('name_at_start_share')} (max "
        f"{thresholds['name_at_start_max']})",
    )
    check(
        "top10_openings",
        metrics.get("top10_opening_share", 1.0) <= thresholds["top10_openings_max"],
        f"top10_opening_share={metrics.get('top10_opening_share')} (max "
        f"{thresholds['top10_openings_max']})",
    )
    check(
        "unique_per_group",
        metrics.get("unique_per_group_median", 0) >= thresholds["median_unique_valid_min"],
        f"median={metrics.get('unique_per_group_median')} (min "
        f"{thresholds['median_unique_valid_min']})",
    )
    return checks
