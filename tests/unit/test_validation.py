"""Validation tests: span schema and the exact-name-once candidate gate."""

import json
import os
import unittest

from pii_redteam import validation

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_fixtures():
    with open(os.path.join(ROOT, "tests", "fixtures", "examples.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestValidation(unittest.TestCase):
    def test_fixtures_are_valid_and_deterministic(self):
        examples = load_fixtures()
        self.assertEqual(len(examples), 3)
        for ex in examples:
            self.assertEqual(validation.validate_annotated_example(ex), [])
        self.assertEqual(load_fixtures(), examples)

    def test_span_slice_matches_name(self):
        ex = load_fixtures()[0]
        span = ex["spans"][0]
        self.assertEqual(ex["text"][span["start"] : span["end"]], "Ana Ruiz")

    def test_bad_offsets_rejected(self):
        ex = {
            "example_id": "bad",
            "text": "Hi Ana",
            "spans": [{"label": "PERSON", "start": 3, "end": 99}],
        }
        self.assertTrue(validation.validate_annotated_example(ex))

    def test_overlapping_spans_rejected(self):
        ex = {
            "example_id": "overlap",
            "text": "Ana Ruiz",
            "spans": [
                {"label": "PERSON", "start": 0, "end": 8},
                {"label": "PERSON", "start": 4, "end": 8},
            ],
        }
        self.assertTrue(any("overlap" in e for e in validation.validate_annotated_example(ex)))

    def test_candidate_exact_once(self):
        ok, _ = validation.validate_candidate(
            text="Hello Ana Ruiz, welcome", supplied_name="Ana Ruiz"
        )
        self.assertTrue(ok)
        self.assertFalse(
            validation.validate_candidate(text="Hello there", supplied_name="Ana Ruiz")[0]
        )
        self.assertFalse(
            validation.validate_candidate(text="Ana Ruiz meets Ana Ruiz", supplied_name="Ana Ruiz")[
                0
            ]
        )
        self.assertFalse(validation.validate_candidate(text="", supplied_name="Ana Ruiz")[0])


if __name__ == "__main__":
    unittest.main()
