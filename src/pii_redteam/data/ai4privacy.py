"""AI4Privacy source adapter (M02).

Converts ai4privacy/pii-masking-300k rows into the canonical annotated-example
schema. Only stdlib: rows arrive as parsed dicts, everything else is pure code.
mBERT token/label arrays are deliberately ignored (plan §4.2).
"""

from __future__ import annotations

import json

REVISION = "c8c77895a005822682b66ab547fc0422579bc1d3"
SOURCE = "ai4privacy/pii-masking-300k"

PERSON_LABELS = frozenset({"GIVENNAME1", "GIVENNAME2", "LASTNAME1", "LASTNAME2", "LASTNAME3"})


class QuarantinedRecord(Exception):
    """Raised when a source row fails verification and must be excluded."""


def parse_span_labels(raw: str) -> list[tuple[int, int, str]]:
    """Parse the serialized `span_labels` field. Safe `json.loads`, never `eval`."""
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"span_labels is not valid JSON: {e}") from e
    if not isinstance(items, list):
        raise ValueError("span_labels must be a JSON list")
    out = []
    for i, item in enumerate(items):
        if (
            not isinstance(item, list)
            or len(item) != 3
            or not isinstance(item[0], int)
            or not isinstance(item[1], int)
            or not isinstance(item[2], str)
        ):
            raise ValueError(f"span_labels[{i}] must be [start:int, end:int, label:str]")
        out.append((item[0], item[1], item[2]))
    return out


def verify_masks(text: str, masks: list[dict]) -> list[str]:
    """Check every mask satisfies text[start:end] == value. Empty means verified."""
    errors = []
    for i, m in enumerate(masks):
        try:
            start, end, value = m["start"], m["end"], m["value"]
        except (KeyError, TypeError):
            errors.append(f"mask {i} is missing start/end/value")
            continue
        if text[start:end] != value:
            errors.append(f"mask {i} ({m.get('label')}): text[{start}:{end}] != {value!r}")
    return errors


def person_spans(masks: list[dict]) -> list[dict]:
    """Map GIVENNAME*/LASTNAME* masks to PERSON spans, merging whitespace-adjacent parts."""
    parts = sorted((m["start"], m["end"]) for m in masks if m.get("label") in PERSON_LABELS)
    merged: list[list[int]] = []
    for start, end in parts:
        if merged and start >= merged[-1][1] and start - merged[-1][1] <= 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"label": "PERSON", "start": s, "end": e} for s, e in merged]


def to_canonical(record: dict) -> dict:
    """Convert one source row to the canonical example. Raises QuarantinedRecord."""
    for key in ("id", "source_text", "privacy_mask", "language"):
        if key not in record:
            raise QuarantinedRecord(f"missing required field {key!r}")
    text = record["source_text"]
    errors = verify_masks(text, record["privacy_mask"])
    if errors:
        raise QuarantinedRecord("; ".join(errors[:3]))
    return {
        "example_id": f"ai4privacy:{record['id']}",
        "text": text,
        "language": record["language"],
        "spans": person_spans(record["privacy_mask"]),
        "provenance": {"source": SOURCE, "revision": REVISION, "record_id": record["id"]},
    }


def is_single_person_once(canonical: dict) -> bool:
    """True when the text holds exactly one PERSON span occurring exactly once."""
    spans = canonical["spans"]
    if len(spans) != 1:
        return False
    name = canonical["text"][spans[0]["start"] : spans[0]["end"]]
    return canonical["text"].count(name) == 1


def is_short_text(canonical: dict, max_words: int = 60) -> bool:
    """True when the text is short enough to be an SFT target (plan §4.2).

    M05 lesson (2026-10-10): training on full AI4Privacy records taught the generator
    to emit long PII-stuffed documents; generation then could not close the JSON
    output contract. SFT targets must be short records (or extracted snippets).
    """
    return len(canonical["text"].split()) <= max_words


def build_sft_pair(canonical: dict, *, request_id: str) -> tuple[dict, str]:
    """Build a (generation request, assistant target) SFT pair (plan §4.1–4.2).

    The target is the existing source text, never a new model paragraph. Raises
    ValueError when the target does not satisfy the single-name-once contract.
    """
    if not is_single_person_once(canonical):
        raise ValueError("SFT pilot needs exactly one PERSON span occurring once")
    span = canonical["spans"][0]
    name = canonical["text"][span["start"] : span["end"]]
    request = {
        "request_id": request_id,
        "request_source": "dataset_derived",
        "task": {
            "language": "en" if canonical["language"] == "English" else canonical["language"],
            "entities": [{"type": "PERSON", "value": name}],
            "constraints": {
                "occurrences_per_entity": 1,
                "no_additional_person_names": True,
            },
        },
    }
    return request, canonical["text"]
