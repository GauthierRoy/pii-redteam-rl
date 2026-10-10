"""AI4Privacy adapter tests: parse, verify, PERSON mapping, canonical, SFT pairs."""

import json
import os
import unittest

from pii_redteam.data import ai4privacy
from pii_redteam.data.ai4privacy import QuarantinedRecord

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_rows():
    with open(os.path.join(ROOT, "tests", "fixtures", "ai4privacy_rows.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestAi4privacy(unittest.TestCase):
    def test_span_labels_parse_and_reject(self):
        self.assertEqual(ai4privacy.parse_span_labels('[[0, 5, "X"]]'), [(0, 5, "X")])
        with self.assertRaises(ValueError):
            ai4privacy.parse_span_labels("not json")
        with self.assertRaises(ValueError):
            ai4privacy.parse_span_labels('{"a": 1}')

    def test_masks_verify_clean(self):
        rows = load_rows()
        self.assertEqual(
            ai4privacy.verify_masks(rows[0]["source_text"], rows[0]["privacy_mask"]), []
        )

    def test_template_leak_quarantined(self):
        rows = load_rows()
        with self.assertRaises(QuarantinedRecord):
            ai4privacy.to_canonical(rows[1])

    def test_person_merge_and_title_excluded(self):
        rows = load_rows()
        canonical = ai4privacy.to_canonical(rows[0])
        self.assertEqual(len(canonical["spans"]), 1)
        span = canonical["spans"][0]
        self.assertEqual(canonical["text"][span["start"] : span["end"]], "Ana Ruiz")
        self.assertNotIn("ana@example.com", canonical["text"][span["start"] : span["end"]])

    def test_accented_slice_preserved(self):
        rows = load_rows()
        canonical = ai4privacy.to_canonical(rows[2])
        span = canonical["spans"][0]
        self.assertEqual(canonical["text"][span["start"] : span["end"]], "Élodie Martin")

    def test_two_names_not_single_once(self):
        rows = load_rows()
        canonical = ai4privacy.to_canonical(rows[3])
        self.assertEqual(len(canonical["spans"]), 2)
        self.assertFalse(ai4privacy.is_single_person_once(canonical))
        with self.assertRaises(ValueError):
            ai4privacy.build_sft_pair(canonical, request_id="x")

    def test_sft_pair_target_satisfies_prompt(self):
        rows = load_rows()
        canonical = ai4privacy.to_canonical(rows[0])
        request, target = ai4privacy.build_sft_pair(canonical, request_id="sft-0000")
        name = request["task"]["entities"][0]["value"]
        self.assertEqual(target.count(name), 1)
        self.assertEqual(request["request_source"], "dataset_derived")

    def test_is_short_text_m05_lesson(self):
        rows = load_rows()
        short = ai4privacy.to_canonical(rows[0])  # "Please send the appointment..."
        self.assertTrue(ai4privacy.is_short_text(short, max_words=60))
        long = ai4privacy.to_canonical(rows[3])
        self.assertFalse(ai4privacy.is_short_text(long, max_words=5))
        self.assertTrue(ai4privacy.is_short_text(long, max_words=200))


if __name__ == "__main__":
    unittest.main()
