"""Strict sample-validity rules v2 (M07b prompting experiment).

Applies to every prompt variant and re-labels M07a pilot samples. Adds to the v1
output contract (JSON `text` field, name exactly once):

  - at least MIN_WORDS words AND at least one clause of real content besides
    the name (pure greetings/small-talk do not count as content)
  - the name is not the whole text; the label form "Name:" is rejected
  - refusals / meta-answers are rejected
  - a hard length cap of MAX_WORDS words

The v1 classifier in `probe_metrics.classify_sample` is untouched (the M07a
scorer path keeps its own definitions); v2 labels live alongside, not instead.
"""

from __future__ import annotations

import json
import re

from pii_redteam.probe_metrics import name_spans, parse_json_strict

MIN_WORDS = 8
MAX_WORDS = 80

# Case-insensitive substrings; a hit means the output is a refusal/meta-answer.
REFUSAL_PATTERNS = (
    r"i cannot",
    r"i can[' ]?t",
    r"as an ai",
    r"as a language model",
    r"i do not know what",
    r"i don[' ]?t know what",
    r"please provide",
    r"could you provide",
    r"i[' a]*m sorry",
    r"i apologize",
    r"i am unable",
    r"unable to comply",
    r"cannot comply",
    r"can[' ]?t comply",
    r"cannot fulfill",
    r"cannot generate",
    r"can[' ]?t generate",
    r"cannot create",
    r"against my",
    r"ethical guidelines",
    r"that request",
)
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

# Words a pure greeting / small-talk opener may consist of. A text whose words
# (minus the name) all come from this set is a greeting, not content.
_GREETING_WORDS = frozenset(
    """
    a am and afternoon are best catch chat day dear doing evening fine finds for friend
    good greetings have hey hello hi hope how i is it its lets lovely meet meeting morning my
    name nice regards see seeing soon talk thank thanks the this to today tonight
    touches up warm well wishes you your
    """.split()
)


def word_count(text: str) -> int:
    return len(text.split())


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text))


def is_greeting_only(text_wo_name: str, name: str) -> bool:
    """True when, without the name, the text is only greeting/small-talk words."""
    stripped = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", " ", text_wo_name, flags=re.IGNORECASE)
    words = [re.sub(r"[^a-z']", "", w.lower()) for w in stripped.split()]
    words = [w for w in words if w]
    if not words:
        return True
    return all(w in _GREETING_WORDS for w in words)


def has_label_form(text: str, name: str) -> bool:
    """The name used as a label ("Name: ..."), not as message content."""
    return re.match(rf"^\s*{re.escape(name)}\s*:", text) is not None


def classify_sample_v2(raw_text: str, name: str, truncated: bool) -> dict:
    """Per-sample v2 validation. Statuses:

    valid                      all v1 contract checks + all v2 content rules
    invalid:truncated / :json / :name_missing / :name_dup   (v1, unchanged)
    invalid:name_only          the name is (nearly) the whole text / label form
    invalid:refusal             refusal or meta-answer detected
    invalid:too_short           fewer than MIN_WORDS words
    invalid:no_content          words >= MIN_WORDS but nothing but greeting
    invalid:too_long            more than MAX_WORDS words
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

    stripped = message.strip()
    if stripped == name or has_label_form(stripped, name):
        return {"status": "invalid:name_only", "text": message, "target_span": spans[0]}
    if is_refusal(message):
        return {"status": "invalid:refusal", "text": message, "target_span": spans[0]}

    words = word_count(message)
    if words < MIN_WORDS:
        return {"status": "invalid:too_short", "text": message, "target_span": spans[0]}
    if is_greeting_only(message, name):
        return {"status": "invalid:no_content", "text": message, "target_span": spans[0]}
    if words > MAX_WORDS:
        return {"status": "invalid:too_long", "text": message, "target_span": spans[0]}
    return {"status": "valid", "text": message, "target_span": spans[0]}


def relabel_samples(samples: list[dict], *, p_target_field: str = "p_target_mean") -> dict:
    """Re-label an M07a/M07b samples.jsonl under v2 rules.

    Each sample dict needs ``raw_text``, ``name``, ``truncated``. Old v1
    ``status``/``outcome``/p-target values are kept for comparison. Returns a
    report with per-sample flips and the hard-sample subset (lowest p_target).
    """
    relabeled = []
    for s in samples:
        v2 = classify_sample_v2(s.get("raw_text", ""), s["name"], bool(s.get("truncated")))
        out = dict(s)
        out["status_v1"] = s.get("status")
        out["outcome"] = s.get("outcome")
        out["status_v2"] = v2["status"]
        out["text_v2"] = v2["text"]
        out["target_span_v2"] = v2["target_span"]
        relabeled.append(out)

    def p_target(s: dict) -> float:
        val = s.get(p_target_field)
        return val if isinstance(val, (int, float)) else 2.0  # unscored sorts last

    sorted_by_p = sorted(relabeled, key=p_target)
    hard_n = min(20, max(1, len(sorted_by_p) // 10 or 1))
    hard = sorted_by_p[:hard_n]
    return {
        "n": len(relabeled),
        "v1_valid": sum(1 for s in relabeled if s["status_v1"] == "valid"),
        "v2_valid": sum(1 for s in relabeled if s["status_v2"] == "valid"),
        "flips_valid_to_invalid": [
            {
                "group_id": s.get("group_id"),
                "sample_idx": s.get("sample_idx"),
                "p_target_mean": s.get(p_target_field),
                "outcome": s.get("outcome"),
                "status_v2": s["status_v2"],
                "text": s.get("text_v2") or s.get("text"),
            }
            for s in relabeled
            if s["status_v1"] == "valid" and s["status_v2"] != "valid"
        ],
        "hard_samples": [
            {
                "group_id": s.get("group_id"),
                "sample_idx": s.get("sample_idx"),
                "p_target_mean": s.get(p_target_field),
                "outcome": s.get("outcome"),
                "status_v1": s["status_v1"],
                "status_v2": s["status_v2"],
                "text": s.get("text_v2") or s.get("text"),
            }
            for s in hard
        ],
        "samples": relabeled,
    }


def raw_text_for(text: str | None) -> str:
    """Reconstruct the raw completion from a parsed text (fixture helper)."""
    return json.dumps({"text": text or ""})
