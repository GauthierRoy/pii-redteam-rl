"""Dataset source adapters (M02). Start with `ai4privacy` for pii-masking-300k."""

from pii_redteam.data.ai4privacy import (
    PERSON_LABELS,
    REVISION,
    SOURCE,
    QuarantinedRecord,
    build_sft_pair,
    is_single_person_once,
    parse_span_labels,
    person_spans,
    to_canonical,
    verify_masks,
)

__all__ = [
    "PERSON_LABELS",
    "REVISION",
    "SOURCE",
    "QuarantinedRecord",
    "build_sft_pair",
    "is_single_person_once",
    "parse_span_labels",
    "person_spans",
    "to_canonical",
    "verify_masks",
]
