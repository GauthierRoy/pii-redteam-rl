"""Tokenizer alignment tests (M03.2): BIO labeling, decoding, truncation, Unicode.

The alignment helpers are tokenizer-agnostic; these tests use stub offset
sequences in Python character indices — the interface HF fast tokenizers expose
via ``return_offsets_mapping=True``. The real ModernBERT tokenizer plugs in at
the Colab gate (M03.1); mBERT arrays are never reused.
"""

import unittest

from pii_redteam.detector import (
    bio_labels_from_spans,
    spans_from_bio_labels,
    trim_span_whitespace,
    truncate_labels,
)


def word_offsets(text: str) -> list[tuple[int, int]]:
    """Space-split token offsets in Python character indices (end-exclusive)."""
    offsets = []
    i = 0
    for part in text.split(" "):
        if part:
            offsets.append((i, i + len(part)))
        i += len(part) + 1
    return offsets


def spans(*ranges: tuple[int, int]) -> list[dict]:
    return [{"label": "PERSON", "start": s, "end": e} for s, e in ranges]


class TestAlignment(unittest.TestCase):
    def test_word_level_roundtrip(self):
        text = "Ana Ruiz met Jean Dupont"
        offsets = word_offsets(text)
        original = spans((0, 8), (13, 24))
        labels = bio_labels_from_spans(offsets, original)
        self.assertEqual(labels, ["B-PERSON", "I-PERSON", "O", "B-PERSON", "I-PERSON"])
        self.assertEqual(spans_from_bio_labels(offsets, labels), original)

    def test_unicode_roundtrip(self):
        text = "Veuillez contacter Élodie Martin demain 🙏"
        offsets = word_offsets(text)
        original = spans((19, 32))
        labels = bio_labels_from_spans(offsets, original)
        self.assertEqual(spans_from_bio_labels(offsets, labels), original)
        self.assertEqual(text[19:32], "Élodie Martin")

    def test_subword_tokens_still_roundtrip(self):
        offsets = [(0, 0), (19, 22), (22, 25), (26, 32), (32, 32)]
        original = spans((19, 32))
        labels = bio_labels_from_spans(offsets, original)
        self.assertEqual(labels, ["O", "B-PERSON", "I-PERSON", "I-PERSON", "O"])
        self.assertEqual(spans_from_bio_labels(offsets, labels), original)

    def test_special_tokens_labeled_O(self):
        offsets = [(0, 0), (0, 3), (4, 8), (0, 0)]
        labels = bio_labels_from_spans(offsets, spans((0, 8)))
        self.assertEqual(labels, ["O", "B-PERSON", "I-PERSON", "O"])
        self.assertEqual(spans_from_bio_labels(offsets, labels), spans((0, 8)))

    def test_boundary_mid_token_is_token_granular(self):
        offsets = [(0, 9)]
        labels = bio_labels_from_spans(offsets, spans((2, 8)))
        self.assertEqual(labels, ["B-PERSON"])
        self.assertEqual(spans_from_bio_labels(offsets, labels), spans((0, 9)))

    def test_orphan_i_starts_new_span(self):
        self.assertEqual(spans_from_bio_labels([(0, 3)], ["I-PERSON"]), spans((0, 3)))

    def test_adjacent_b_tokens_are_distinct_spans(self):
        offsets = [(0, 3), (3, 6)]
        decoded = spans_from_bio_labels(offsets, ["B-PERSON", "B-PERSON"])
        self.assertEqual(decoded, spans((0, 3), (3, 6)))

    def test_label_switch_splits_span(self):
        offsets = [(0, 3), (3, 6)]
        decoded = spans_from_bio_labels(offsets, ["B-PERSON", "I-LOC"])
        self.assertEqual(
            decoded,
            [{"label": "PERSON", "start": 0, "end": 3}, {"label": "LOC", "start": 3, "end": 6}],
        )

    def test_mismatched_lengths_and_bad_labels_rejected(self):
        with self.assertRaises(ValueError):
            spans_from_bio_labels([(0, 1)], ["B-PERSON", "I-PERSON"])
        with self.assertRaises(ValueError):
            spans_from_bio_labels([(0, 1)], ["X-PERSON"])
        with self.assertRaises(ValueError):
            spans_from_bio_labels([(0, 1)], ["B-"])

    def test_truncation(self):
        labels = ["O", "B-PERSON", "I-PERSON", "O"]
        self.assertEqual(truncate_labels(labels, 2), ["O", "B-PERSON"])
        self.assertEqual(truncate_labels(labels, 99), labels)
        with self.assertRaises(ValueError):
            truncate_labels(labels, -1)

    def test_bpe_leading_space_trimmed_on_recovery(self):
        # ModernBERT-style BPE: the token starting a word includes the leading space.
        text = "Contact Ana"
        offsets = [(0, 7), (7, 11)]  # tokens "Contact" | " Ana" (space-prefixed)
        gold = spans((8, 11))
        labels = bio_labels_from_spans(offsets, gold)
        self.assertEqual(labels, ["O", "B-PERSON"])
        raw = spans_from_bio_labels(offsets, labels)
        self.assertEqual(raw, spans((7, 11)))  # token-granular: includes the space
        self.assertEqual(trim_span_whitespace(text, raw), gold)

    def test_trim_drops_whitespace_only_and_trims_trailing(self):
        self.assertEqual(trim_span_whitespace("  ", spans((0, 2))), [])
        text = "Ana here"
        self.assertEqual(trim_span_whitespace(text, spans((0, 4))), spans((0, 3)))


if __name__ == "__main__":
    unittest.main()
