"""Schema and span validation (M01.3).

Canonical boundary: Python str character indices, end-exclusive. Exact string
presence of a name is a necessary check, never proof of semantic validity —
that limitation is recorded here and enforced by audit in M07/M09.
"""

from __future__ import annotations

PERSON = "PERSON"


def validate_annotated_example(ex: dict) -> list[str]:
    """Return a list of schema errors; empty means valid."""
    errors: list[str] = []
    text = ex.get("text")
    if not isinstance(text, str) or not text:
        return ["example needs a non-empty string 'text'"]
    spans = ex.get("spans")
    if not isinstance(spans, list):
        return ["example needs a list 'spans'"]
    parsed: list[tuple[int, int, str]] = []
    for i, span in enumerate(spans):
        if not isinstance(span, dict):
            errors.append(f"span {i} is not a mapping")
            continue
        label, start, end = span.get("label"), span.get("start"), span.get("end")
        if label != PERSON:
            errors.append(f"span {i} label must be {PERSON!r}, got {label!r}")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"span {i} start/end must be ints")
            continue
        if not 0 <= start < end <= len(text):
            errors.append(f"span {i} offsets [{start}, {end}) out of bounds for len {len(text)}")
            continue
        parsed.append((start, end, label))
    parsed.sort()
    for (s1, e1, _), (s2, e2, _) in zip(parsed, parsed[1:], strict=False):
        if s2 < e1:
            errors.append(f"overlapping spans [{s1}, {e1}) and [{s2}, {e2})")
    return errors


def validate_candidate(*, text: str, supplied_name: str, max_len: int = 500) -> tuple[bool, str]:
    """Hard validity gate for a generated positive (M07.1 subset at M01)."""
    if not isinstance(text, str) or not text:
        return False, "empty output"
    if len(text) > max_len:
        return False, f"exceeds max_len {max_len}"
    if not supplied_name:
        return False, "no supplied name"
    count = text.count(supplied_name)
    if count == 0:
        return False, "missing supplied name"
    if count > 1:
        return False, "duplicate supplied name"
    return True, "ok"
