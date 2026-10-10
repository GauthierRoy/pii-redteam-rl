"""Dataset source adapters (M02). Start with `ai4privacy` for pii-masking-300k."""

from pii_redteam.data.ai4privacy import (
    PERSON_LABELS,
    REVISION,
    SOURCE,
    QuarantinedRecord,
    build_sft_pair,
    is_short_text,
    is_single_person_once,
    parse_span_labels,
    person_spans,
    to_canonical,
    verify_masks,
)
from pii_redteam.data.splits import (
    carve_final_eval,
    carve_heldout_names,
    dedup_by_text,
    person_name_counts,
    template_skeleton,
    text_sha256,
)

__all__ = [
    "PERSON_LABELS",
    "REVISION",
    "SOURCE",
    "QuarantinedRecord",
    "build_sft_pair",
    "carve_final_eval",
    "carve_heldout_names",
    "dedup_by_text",
    "is_short_text",
    "is_single_person_once",
    "parse_span_labels",
    "person_name_counts",
    "person_spans",
    "template_skeleton",
    "text_sha256",
    "to_canonical",
    "verify_masks",
]
