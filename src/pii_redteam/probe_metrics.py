"""Pure probe-metric functions (M07a): validation, diversity, reward signal, thresholds.

Everything here is GPU-free and unit-tested; `probe.py` does the torch work and
calls into this module. Definitions used for outcome classification:
  exact_hit      a predicted PERSON span equals the target span
  partial_overlap a predicted span strictly contains / is contained in the target
  boundary_error a predicted span crosses a target boundary (overlap, no containment)
  complete_miss  no predicted span overlaps the target
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import combinations

PROMPT_MARKER = "Write one short realistic message"


# ---------------------------------------------------------------- validation


def parse_json_strict(text: str) -> str | None:
    """Strict contract parse: a single JSON object with a string 'text' field."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("text"), str):
        return parsed["text"]
    return None


def parse_json_tolerant(text: str) -> str | None:
    """Tolerant parse: allow markdown code fences around the JSON object."""
    stripped = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    return parse_json_strict(stripped)


def name_spans(text: str, name: str) -> list[tuple[int, int]]:
    """Word-boundary character spans of the exact name in the text."""
    return [(m.start(), m.end()) for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text)]


def classify_sample(raw_text: str, name: str, truncated: bool) -> dict:
    """Per-sample validation. Status: valid or invalid:{json|name_missing|name_dup|truncated}.

    Truncation is checked first: a truncated generation can never close the JSON,
    so reporting invalid:json for it would hide the root cause.
    """
    if truncated:
        return {"status": "invalid:truncated", "text": None, "target_span": None}
    message = parse_json_strict(raw_text)
    if message is None:
        return {"status": "invalid:json", "text": None, "target_span": None}
    spans = name_spans(message, name)
    if not spans:
        return {"status": "invalid:name_missing", "text": message, "target_span": None}
    if len(spans) > 1:
        return {"status": "invalid:name_dup", "text": message, "target_span": None}
    return {"status": "valid", "text": message, "target_span": spans[0]}


def name_position_fraction(span: tuple[int, int], text: str) -> float:
    """0.0 = name at start, 1.0 = at end of the text."""
    if not text:
        return 0.0
    return (span[0] + span[1]) / 2 / len(text)


# ---------------------------------------------------------------- diversity


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def distinct_n(word_lists: list[list[str]], n: int) -> float:
    grams = [tuple(ws[i : i + n]) for ws in word_lists for i in range(len(ws) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def jaccard(a: set, b: set) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def diversity_metrics(samples: list[dict]) -> dict:
    """Contract rate, duplicates, distinct-n, within-group diversity, template collapse.

    `samples` are per-sample dicts with at least: group_id, status, text, raw_text,
    temperature, n_chars, n_tokens, target_span.
    """
    valid = [s for s in samples if s["status"] == "valid"]
    out: dict = {
        "total": len(samples),
        "valid": len(valid),
        "contract_rate": round(len(valid) / len(samples), 4) if samples else 0.0,
    }
    failures = Counter(s["status"] for s in samples if s["status"] != "valid")
    out["failure_breakdown"] = dict(sorted(failures.items()))

    texts = [s["text"] for s in valid]
    normed = [normalize(t) for t in texts]
    n = len(texts)
    if n < 2:
        return out | {
            "exact_dup_rate": 0.0,
            "normalized_dup_rate": 0.0,
            "distinct1": 0.0,
            "distinct2": 0.0,
        }

    out["exact_dup_rate"] = round(1 - len(set(texts)) / n, 4)
    out["normalized_dup_rate"] = round(1 - len(set(normed)) / n, 4)
    word_lists = [t.split() for t in texts]
    out["distinct1"] = round(distinct_n(word_lists, 1), 4)
    out["distinct2"] = round(distinct_n(word_lists, 2), 4)

    # within-group unique valid outputs
    groups: dict[str, list[str]] = {}
    for s in valid:
        groups.setdefault(s["group_id"], []).append(normalize(s["text"]))
    uniq = sorted(len(set(g)) for g in groups.values())
    if uniq:
        out["unique_per_group_median"] = uniq[len(uniq) // 2]
        out["unique_per_group_p10"] = uniq[max(0, int(len(uniq) * 0.1))]
    # pairwise Jaccard of word 3-grams within groups
    sims = []
    for g in groups.values():
        gram_sets = [
            {tuple(ws[i : i + 3]) for i in range(len(ws) - 2)} for ws in [t.split() for t in g]
        ]
        for a, b in combinations(gram_sets, 2):
            sims.append(jaccard(a, b))
    out["within_group_jaccard_mean"] = round(sum(sims) / len(sims), 4) if sims else 0.0

    # template collapse: opening patterns and name position
    openings = Counter(" ".join(t.split()[:5]) for t in texts)
    out["top10_opening_share"] = round(sum(c for _, c in openings.most_common(10)) / n, 4)
    pos = [name_position_fraction(s["target_span"], s["text"]) for s in valid]
    out["name_position"] = {
        "start(<25%)": sum(1 for p in pos if p < 0.25) / len(pos) if pos else 0.0,
        "middle": sum(1 for p in pos if 0.25 <= p < 0.75) / len(pos) if pos else 0.0,
        "end(>=75%)": sum(1 for p in pos if p >= 0.75) / len(pos) if pos else 0.0,
    }
    out["length_chars"] = {
        "p10": sorted(s["n_chars"] for s in valid)[int(len(valid) * 0.1)] if valid else 0,
        "p50": sorted(s["n_chars"] for s in valid)[len(valid) // 2] if valid else 0,
        "p90": sorted(s["n_chars"] for s in valid)[int(len(valid) * 0.9)] if valid else 0,
    }
    return out


# ---------------------------------------------------------------- rewards


def outcome_for_sample(target_span: tuple[int, int], pred_spans: list[tuple[int, int]]) -> str:
    """Classify one sample against D0's decoded PERSON spans."""
    ts, te = target_span
    overlap = [p for p in pred_spans if p[0] < te and p[1] > ts]
    if not overlap:
        return "complete_miss"
    for ps, pe in overlap:
        if (ps, pe) == (ts, te):
            return "exact_hit"
    for ps, pe in overlap:
        if (ps <= ts and pe >= te) or (ps >= ts and pe <= te):
            return "partial_overlap"
    return "boundary_error"


def logit(p: float, clip: float = 10.0) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    val = -math.log(p / (1 - p))
    return max(-clip, min(clip, val))


def reward_variants(p_target_mean: float) -> dict:
    return {"r_prob": 1.0 - p_target_mean, "r_logit": logit(p_target_mean)}


def reward_metrics(samples: list[dict], clip: float = 10.0) -> dict:
    """Aggregate reward-signal metrics over VALID samples (invalids excluded, reported).

    Each sample needs: group_id, status, p_target_mean, p_target_min, outcome,
    n_chars, name_position, extra_person_flags.
    """
    valid = [s for s in samples if s["status"] == "valid"]
    invalid = [s for s in samples if s["status"] != "valid"]
    out: dict = {"valid_scored": len(valid), "invalid_excluded": len(invalid)}
    if not valid:
        return out

    outcomes = Counter(s["outcome"] for s in valid)
    n = len(valid)
    out["outcome_rates"] = {
        k: round(outcomes.get(k, 0) / n, 4)
        for k in ("exact_hit", "partial_overlap", "boundary_error", "complete_miss")
    }

    ps = sorted(s["p_target_mean"] for s in valid)
    out["p_target_mean"] = {
        "p1": ps[max(0, int(n * 0.01))],
        "p5": ps[max(0, int(n * 0.05))],
        "p25": ps[int(n * 0.25)],
        "p50": ps[n // 2],
    }
    out["p_target_min_p5"] = sorted(s["p_target_min"] for s in valid)[max(0, int(n * 0.05))]

    # group-level signal for each reward variant
    groups: dict[str, list[dict]] = {}
    for s in valid:
        groups.setdefault(s["group_id"], []).append(s)
    out["dead_group_share"] = round(
        sum(1 for g in groups.values() if all(s["p_target_mean"] > 0.99 for s in g)) / len(groups),
        4,
    )
    for variant in ("r_prob", "r_logit"):
        stds = []
        for g in groups.values():
            vals = [reward_variants(s["p_target_mean"])[variant] for s in g]
            mean = sum(vals) / len(vals)
            stds.append((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5)
        out[f"live_group_share_{variant}"] = round(
            sum(1 for s_ in stds if s_ >= 0.05) / len(stds), 4
        )
        out[f"within_group_std_mean_{variant}"] = round(sum(stds) / len(stds), 4)

    # exploit checks: correlation of reward with length / position / extra flags
    def corr(xs: list[float], ys: list[float]) -> float:
        n_ = len(xs)
        if n_ < 3:
            return 0.0
        mx, my = sum(xs) / n_, sum(ys) / n_
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        return cov / (vx * vy) if vx * vy else 0.0

    rewards = [s["r_prob"] for s in valid]
    out["exploit_correlations"] = {
        "length": round(corr(rewards, [float(s["n_chars"]) for s in valid]), 4),
        "name_position": round(corr(rewards, [s["name_position"] for s in valid]), 4),
        "extra_person_flag": round(
            corr(rewards, [float(s["extra_person_flags"] > 0) for s in valid]), 4
        ),
    }
    return out


# ---------------------------------------------------------------- thresholds


def threshold_report(metrics: dict, thresholds: dict) -> dict:
    """Evaluate the go/no-go table. Every check gets a pass/fail + the measured value."""
    miss_rate = metrics.get("outcome_rates", {}).get("complete_miss")
    live = max(
        metrics.get("live_group_share_r_prob", 0.0), metrics.get("live_group_share_r_logit", 0.0)
    )
    checks = {
        "format_valid": (
            metrics.get("contract_rate", 0.0) >= thresholds["format_valid_min"],
            f"contract_rate={metrics.get('contract_rate')}",
        ),
        "duplicates": (
            metrics.get("exact_dup_rate", 1.0) < thresholds["exact_dup_max"],
            f"exact_dup_rate={metrics.get('exact_dup_rate')}",
        ),
        "within_group_diversity": (
            metrics.get("unique_per_group_median", 0) >= thresholds["unique_per_group_median_min"],
            f"median={metrics.get('unique_per_group_median')}",
        ),
        "reward_signal": (
            (live >= thresholds["live_group_share_min"])
            or (
                miss_rate is not None
                and thresholds["miss_rate_range"][0]
                <= miss_rate
                <= thresholds["miss_rate_range"][1]
            ),
            f"live_group_share={live}, miss_rate={miss_rate}",
        ),
        "truncation": (
            metrics.get("target_spans_lost_to_d0_truncation", 0)
            <= thresholds["target_spans_lost_max"],
            f"lost={metrics.get('target_spans_lost_to_d0_truncation', 0)}",
        ),
    }
    return {name: {"ok": bool(ok), "detail": detail} for name, (ok, detail) in checks.items()}
