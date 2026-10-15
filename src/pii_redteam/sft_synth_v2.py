"""Synthetic SFT dataset v2 toolkit (template_paraphrase).

Additive module for the ``feat/spark-sft-data`` branch. Frozen code — the
scorer (``detector.py``, ``rewards.py``) and the prompt-variant code
(``probe.py``, ``probe_metrics.py``, ``requests.py``, ``generator.py``) — is
never modified here. The only import from frozen code is the read-only R07
pool filter (``is_requestable_name``); span helpers used by the *separate*
D0-flag script are imported there, not here, so this module stays stdlib-only.

Row format (request schema v2)::

    {"id": ..., "split": "sft_train" | "sft_dev" | "short_eval",
     "request": {"language": "en", "context_type": ...,
                 "target_length": [min, max],
                 "name_position": "start|middle|end|signature",
                 "entities": [{"type": "PERSON", "value": "<exact name>"}]},
     "target": {"text": "...", "span": [start, end]},
     "provenance": "template_paraphrase|llm_authored_direct|human_authored", "name_form": "first|full|surname"}

``name_form`` is a derived coverage attribute, not a model input: ``full``
means the entity value itself is multi-token; ``first``/``surname`` describe
the usage slot of a single-token value (casual vs. titled/formal reference).
Character offsets are Python str indices, end-exclusive (R03).
"""

from __future__ import annotations

import re
from collections import Counter

CONTEXT_TYPES = (
    "support ticket",
    "email with signature",
    "forwarded email thread",
    "meeting notes",
    "chat message",
    "appointment reminder",
    "form/record line",
    "list/table row",
    "informal text with typos",
    "third-person report",
)

NAME_POSITIONS = ("start", "middle", "end", "signature")

# Word-count buckets covering the 20-60 range (owner-confirmed).
LENGTH_BUCKETS = ((20, 30), (31, 45), (46, 60))

NAME_FORMS = ("first", "full", "surname")

# Row provenance labels (dataset relabel D16 + short_eval_v2 STEP 2).
PROVENANCES = ("template_paraphrase", "llm_authored_direct", "human_authored")

SPLITS = ("sft_train", "sft_dev", "short_eval")

MIN_WORDS, MAX_WORDS = 20, 60

# Refusal / assistant-meta framings that must never appear in authored texts.
# Deliberately tight: ordinary message openers ("Please note", "Here is the
# file") are natural and allowed; only assistant-role framings are banned.
REFUSAL_RES = [
    re.compile(r"(?i)\bas an ai\b"),
    re.compile(r"(?i)\bas a language model\b"),
    re.compile(r"(?i)\bi (can'?t|cannot|am unable|won'?t) (help|comply|do that|assist)\b"),
    re.compile(r"(?i)\bunable to comply\b"),
]

META_RES = [
    re.compile(r"(?i)\bhere is (a|an|your|the) (synthetic|example|requested|generated)\b"),
    re.compile(r"(?i)\bi('ve| have) written\b"),
    re.compile(r"(?i)\blet me know if you('d| would) like"),
    re.compile(r"(?i)\bsure,? here\b"),
    re.compile(r"(?i)\bthis (text|message|example) contains the name\b"),
]

# Tokens inside which the supplied name must never hide (no lucas.martin@...).
_CREDENTIAL_TOKEN_RES = [
    re.compile(r"[\w.+-]+@[\w.-]+\.\w+"),  # emails
    re.compile(r"(?<!\w)@\w+"),  # handles
    re.compile(r"https?://\S+"),  # urls
    re.compile(r"(?<!\w)[\w.-]+\.(com|org|net|io|fr|de|eu|co\.uk)\b"),  # bare domains
]

_SIGNOFF_CUE = re.compile(
    r"(?i)(thanks|thank you|best regards|kind regards|best|regards|cheers|sincerely|signed|--)"
    r"\s*$"
    r"|(thanks|thank you|best regards|kind regards|best|regards|cheers|sincerely)\s*,?\s*\n"
)


def word_count(text: str) -> int:
    """Whitespace-token word count (the target_length contract unit)."""
    return len(text.split())


def bucket_of(n: int) -> int | None:
    """Index into LENGTH_BUCKETS holding n words, or None when out of range."""
    for i, (lo, hi) in enumerate(LENGTH_BUCKETS):
        if lo <= n <= hi:
            return i
    return None


def find_spans(text: str, name: str) -> list[tuple[int, int]]:
    """Word-boundary character spans of the exact name (same convention as probe_metrics)."""
    if not name:
        return []
    return [(m.start(), m.end()) for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", text)]


def compute_span(text: str, name: str) -> tuple[int, int] | None:
    """The single span when the name occurs exactly once, else None."""
    spans = find_spans(text, name)
    return spans[0] if len(spans) == 1 else None


def position_fraction(span: tuple[int, int], text: str) -> float:
    """Span-center fraction: 0.0 = text start, 1.0 = text end."""
    if not text:
        return 0.0
    return (span[0] + span[1]) / 2 / len(text)


def measured_position(span: tuple[int, int], text: str) -> str:
    """Band of the span center: start (<25%), end (>=75%), else middle."""
    frac = position_fraction(span, text)
    if frac < 0.25:
        return "start"
    if frac >= 0.75:
        return "end"
    return "middle"


def is_signature(text: str, span: tuple[int, int]) -> bool:
    """True when the span sits in a trailing sign-off (last line region)."""
    tail = text[span[1] :]
    if len(tail) > 80:
        return False
    head = text[max(0, span[0] - 80) : span[0]]
    last_line_start = text.rfind("\n", 0, span[0]) + 1
    in_last_line = span[0] >= last_line_start and "\n" not in text[span[0] : span[1]]
    return in_last_line and bool(_SIGNOFF_CUE.search(head) or head.strip().endswith("--"))


def check_position(text: str, span: tuple[int, int], claimed: str) -> bool:
    """True when the span placement satisfies the claimed name_position."""
    if claimed == "signature":
        return is_signature(text, span)
    if is_signature(text, span):
        # A trailing sign-off also counts as end-adjacent, nothing else.
        return claimed == "end"
    return measured_position(span, text) == claimed


def _alnum_lower(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.lower())


def name_hidden_in_other_pii(text: str, name: str) -> str | None:
    """Return the offending token when the name hides inside an email/handle/url/domain."""
    norm = _alnum_lower(name)
    parts = [_alnum_lower(p) for p in name.split() if len(_alnum_lower(p)) >= 4]
    needles = [n for n in [norm, *parts] if n]
    for rx in _CREDENTIAL_TOKEN_RES:
        for m in rx.finditer(text):
            tok_norm = _alnum_lower(m.group(0))
            for needle in needles:
                if needle and needle in tok_norm:
                    return m.group(0)
    return None


def refusal_or_meta_hit(text: str) -> str | None:
    """Return the matched pattern string for refusal/meta-text, else None."""
    for rx in (*REFUSAL_RES, *META_RES):
        m = rx.search(text)
        if m:
            return m.group(0)
    return None


def validate_row_schema(row: dict) -> list[str]:
    """JSON-schema-level errors for a v2 row (no text-content checks)."""
    errors: list[str] = []
    if row.get("split") not in SPLITS:
        errors.append(f"split must be one of {SPLITS}, got {row.get('split')!r}")
    request = row.get("request")
    if not isinstance(request, dict):
        return errors + ["row needs a mapping 'request'"]
    if request.get("language") != "en":
        errors.append(f"request.language must be 'en', got {request.get('language')!r}")
    if request.get("context_type") not in CONTEXT_TYPES:
        errors.append(f"unknown context_type {request.get('context_type')!r}")
    bounds = request.get("target_length")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or not all(isinstance(b, int) for b in bounds)
        or not (MIN_WORDS <= bounds[0] <= bounds[1] <= MAX_WORDS)
    ):
        errors.append(f"target_length must be [min,max] ints within [{MIN_WORDS},{MAX_WORDS}]")
    elif tuple(bounds) not in LENGTH_BUCKETS:
        errors.append(f"target_length {bounds} is not one of the coverage buckets {LENGTH_BUCKETS}")
    if request.get("name_position") not in NAME_POSITIONS:
        errors.append(f"unknown name_position {request.get('name_position')!r}")
    entities = request.get("entities")
    if not isinstance(entities, list) or len(entities) != 1:
        errors.append("request.entities must hold exactly one entity")
    else:
        ent = entities[0]
        if not isinstance(ent, dict) or ent.get("type") != "PERSON" or not ent.get("value"):
            errors.append("entity must be {type: PERSON, value: <non-empty name>}")
    if row.get("provenance") not in PROVENANCES:
        errors.append(f"provenance must be one of {PROVENANCES}, got {row.get('provenance')!r}")
    if row.get("name_form") not in NAME_FORMS:
        errors.append(f"unknown name_form {row.get('name_form')!r}")
    return errors


def validate_row_content(row: dict) -> list[str]:
    """Hard content-contract errors: exact-once match, spans, length, position, hygiene.

    Hygiene (post-audit, 2026-10-15): repeated sentences, >= 3 sentences under
    4 words (filler padding), and any 3-gram repeated within the row.
    """
    errors: list[str] = []
    try:
        name = row["request"]["entities"][0]["value"]
        text = row["target"]["text"]
    except (KeyError, TypeError, IndexError):
        return ["row missing request.entities[0].value or target.text"]
    if not isinstance(text, str) or not text.strip():
        return ["target.text must be a non-empty string"]
    rep = repeated_sentence(text)
    if rep:
        errors.append(f"filler: repeated sentence {rep!r}")
    if short_sentence_excess(text):
        errors.append("filler: >= 3 sentences under 4 words")
    tg = repeated_trigram(text)
    if tg:
        errors.append(f"filler: repeated 3-gram {' '.join(tg)!r}")
    spans = find_spans(text, name)
    if len(spans) == 0:
        errors.append(f"name {name!r} missing (no word-boundary match)")
    elif len(spans) > 1:
        errors.append(f"name {name!r} occurs {len(spans)}x, must occur exactly once")
    else:
        span = spans[0]
        if text[span[0] : span[1]] != name:
            errors.append("computed span does not slice back to the name")
        if row.get("target", {}).get("span") != [span[0], span[1]]:
            errors.append("recorded target.span disagrees with the computed span")
        if not check_position(text, span, row["request"].get("name_position", "")):
            errors.append(
                f"span placement fails claimed name_position "
                f"{row['request'].get('name_position')!r}"
            )
    bounds = row["request"].get("target_length", [MIN_WORDS, MAX_WORDS])
    n = word_count(text)
    if isinstance(bounds, list) and len(bounds) == 2 and not (bounds[0] <= n <= bounds[1]):
        errors.append(f"word count {n} outside target_length {bounds}")
    bad_token = name_hidden_in_other_pii(text, name)
    if bad_token:
        errors.append(f"name hides inside other PII token {bad_token!r}")
    hit = refusal_or_meta_hit(text)
    if hit:
        errors.append(f"refusal/meta-text pattern {hit!r}")
    return errors


def validate_row(row: dict) -> list[str]:
    """All hard errors for a v2 row (schema + content). Empty means shippable."""
    return validate_row_schema(row) + validate_row_content(row)


def normalize(text: str) -> str:
    """Lowercase whitespace-collapsed form for exact-dup detection."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def word_trigrams(text: str) -> set[tuple[str, str, str]]:
    """Word 3-gram set for near-duplicate detection."""
    words = text.split()
    return {(words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)}


def jaccard3(a: set, b: set) -> float:
    """Jaccard similarity over word 3-gram sets (1.0 when both empty)."""
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def find_duplicates(rows: list[dict], threshold: float = 0.7) -> dict:
    """Exact + near-duplicate detection over row texts.

    Returns ``{"exact": [(i, j)], "near": [(i, j, sim)]}`` index pairs with
    j > i. Exact uses the normalized form; near uses 3-gram Jaccard over the
    raw text and skips pairs already exact-duplicate.
    """
    normed = [normalize(r["target"]["text"]) for r in rows]
    seen: dict[str, int] = {}
    exact: list[tuple[int, int]] = []
    for j, key in enumerate(normed):
        if key in seen:
            exact.append((seen[key], j))
        else:
            seen[key] = j
    exact_set = set(exact)
    grams = [word_trigrams(r["target"]["text"]) for r in rows]
    near: list[tuple[int, int, float]] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if (i, j) in exact_set:
                continue
            sim = jaccard3(grams[i], grams[j])
            if sim > threshold:
                near.append((i, j, round(sim, 4)))
    return {"exact": exact, "near": near}


def opening(text: str, n_words: int = 5) -> str:
    """Opening key (first n words, lowercased) for template-collapse monitoring."""
    return " ".join(text.split()[:n_words]).lower()


# ---- post-audit hygiene checks (2026-10-15, spark-sft-v1 STEP 1) ----

# Fail a row when it contains this many or more sentences of < 4 words.
FILLER_SHORT_SENTENCE_LIMIT = 3
# Template reuse: runs of >= this many consecutive normalized shared words.
TEMPLATE_REUSE_WORDS = 8

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _sentence_key(sentence: str) -> str:
    """Normalized sentence key: lowercase, punctuation stripped."""
    return " ".join(re.findall(r"[a-z0-9']+", sentence.lower()))


def repeated_sentence(text: str) -> str | None:
    """Return the first sentence that appears more than once in the row, else None."""
    seen: set[str] = set()
    for s in _sentences(text):
        key = _sentence_key(s)
        if not key:
            continue
        if key in seen:
            return s
        seen.add(key)
    return None


def short_sentence_excess(text: str, limit: int = FILLER_SHORT_SENTENCE_LIMIT) -> bool:
    """True when the row has >= limit sentences of fewer than 4 words."""
    n = sum(1 for s in _sentences(text) if 0 < len(s.split()) < 4)
    return n >= limit


def repeated_trigram(text: str) -> tuple[str, str, str] | None:
    """Return a word 3-gram that occurs more than once within the row, else None."""
    words = [w for w in re.findall(r"[A-Za-z0-9']+", text.lower()) if w]
    seen: set[tuple[str, str, str]] = set()
    for i in range(len(words) - 2):
        g = (words[i], words[i + 1], words[i + 2])
        if g in seen:
            return g
        seen.add(g)
    return None


def _norm_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w]


def _shingles(text: str, n: int) -> set[tuple[str, ...]]:
    words = _norm_words(text)
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def find_template_reuse(rows: list[dict], n_words: int = TEMPLATE_REUSE_WORDS) -> dict[str, object]:
    """Template-reuse detection across rows: >= n_words consecutive shared words.

    Returns ``{"pairs": [(i, j)], "rows_involved": {i, ...}}`` over input order.
    Within-split rates should be computed by the caller (this is split-agnostic).
    """
    shingles: dict[tuple[str, ...], set[int]] = {}
    for i, r in enumerate(rows):
        for s in _shingles(r["target"]["text"], n_words):
            shingles.setdefault(s, set()).add(i)
    pairs: set[tuple[int, int]] = set()
    for owners in shingles.values():
        if len(owners) > 1:
            o = sorted(owners)
            for a in range(len(o)):
                for b in range(a + 1, len(o)):
                    pairs.add((o[a], o[b]))
    return {
        "pairs": sorted(pairs),
        "rows_involved": sorted({x for p in pairs for x in p}),
    }


def cross_split_template_overlap(
    rows: list[dict], eval_splits: tuple[str, ...] = ("short_eval",)
) -> list[tuple[str, str, int]]:
    """Template overlap between non-eval rows and eval-split rows (hard-fail check).

    Returns ``[(non_eval_id, eval_id, n_shared_shingles)]`` for every row pair
    sharing at least one 8-word run. Must be empty for any usable dataset.
    """
    n = TEMPLATE_REUSE_WORDS
    eval_shingles: dict[tuple[str, ...], set[str]] = {}
    for r in rows:
        if r["split"] in eval_splits:
            for s in _shingles(r["target"]["text"], n):
                eval_shingles.setdefault(s, set()).add(r["id"])
    overlaps: list[tuple[str, str, int]] = []
    seen_pairs: dict[tuple[str, str], int] = {}
    for r in rows:
        if r["split"] in eval_splits:
            continue
        for s in _shingles(r["target"]["text"], n):
            for eval_id in eval_shingles.get(s, ()):
                key = (r["id"], eval_id)
                seen_pairs[key] = seen_pairs.get(key, 0) + 1
    overlaps = [(a, b, c) for (a, b), c in sorted(seen_pairs.items())]
    return overlaps


def coverage_counts(rows: list[dict]) -> dict:
    """Nested counts: context_type x name_position x bucket x name_form."""
    counts: dict[str, dict] = {}
    for row in rows:
        req = row["request"]
        bucket = f"{req['target_length'][0]}-{req['target_length'][1]}"
        counts.setdefault(req["context_type"], {}).setdefault(req["name_position"], {}).setdefault(
            bucket, Counter()
        )[row["name_form"]] += 1
    return counts
