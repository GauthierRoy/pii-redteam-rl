"""Tests for name-free eval rows (short_eval_neg_v1) and the shipped dataset."""

import copy
import hashlib
import json
import os
import unittest

from pii_redteam import sft_synth_v2 as v2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data/synthetic/short_eval_neg_v1.jsonl")
MANIFEST = os.path.join(ROOT, "data/synthetic/short_eval_neg_v1_manifest.json")

TEXT = (
    "Reminder: your annual check-up is confirmed for Thursday at 3 PM at Bayview Clinic, "
    "40 Harbor Road, Portland. Please arrive ten minutes early. Front desk team"
)


def neg_row(**over):
    base = {
        "id": "short-eval-neg-v1-0000",
        "split": "short_eval",
        "request": {
            "language": "en",
            "context_type": "appointment reminder",
            "target_length": [20, 30],
            "name_position": None,
            "entities": [],
        },
        "target": {"text": TEXT, "span": None},
        "provenance": "llm_authored_direct:glm-5.3",
        "name_form": None,
        "distractor": "place",
    }
    base.update(over)
    return base


class TestValidateNegativeRow(unittest.TestCase):
    def test_good_row_passes(self):
        self.assertEqual(v2.validate_negative_row(neg_row()), [])

    def test_entities_must_be_empty(self):
        r = neg_row()
        r["request"]["entities"] = [{"type": "PERSON", "value": "Ana"}]
        self.assertTrue(any("entities" in e for e in v2.validate_negative_row(r)))

    def test_span_must_be_null(self):
        r = neg_row(target={"text": TEXT, "span": [0, 3]})
        self.assertTrue(any("span" in e for e in v2.validate_negative_row(r)))

    def test_name_position_and_form_must_be_null(self):
        r = neg_row(name_form="first")
        r["request"]["name_position"] = "start"
        errs = v2.validate_negative_row(r)
        self.assertTrue(any("name_position" in e for e in errs))
        self.assertTrue(any("name_form" in e for e in errs))

    def test_unknown_distractor_rejected(self):
        self.assertTrue(v2.validate_negative_row(neg_row(distractor="pet")))

    def test_word_count_outside_bucket_rejected(self):
        r = neg_row(target={"text": "Closed Thursday at Harbor Road.", "span": None})
        self.assertTrue(any("word count" in e for e in v2.validate_negative_row(r)))

    def test_placeholder_rejected(self):
        r = neg_row(target={"text": TEXT.replace("Bayview Clinic", "[Clinic]"), "span": None})
        self.assertTrue(any("placeholder" in e for e in v2.validate_negative_row(r)))

    def test_subject_echo_is_not_filler(self):
        text = (
            "Subject: Kestrel app crashes on launch\n\nThe Kestrel app crashes whenever I open "
            "the billing module. Reinstalling it did not help. Running the latest build.\n\n"
            "— Accounts Team"
        )
        r = neg_row(target={"text": text, "span": None})
        r["request"]["context_type"] = "email with signature"
        self.assertEqual(v2.validate_negative_row(r), [])

    def test_body_repeat_is_still_filler(self):
        text = (
            "Raised by the duty manager at noon, then checked by the duty manager again after "
            "lunch because the boiler in the basement kept tripping. Logged for the night crew."
        )
        r = neg_row(target={"text": text, "span": None})
        self.assertTrue(any("3-gram" in e for e in v2.validate_negative_row(r)))

    def test_does_not_mutate_input(self):
        r = neg_row()
        before = copy.deepcopy(r)
        v2.validate_negative_row(r)
        self.assertEqual(r, before)

    def test_homograph_is_not_gated(self):
        self.assertIn("homograph", v2.DISTRACTORS)
        self.assertNotIn("homograph", v2.GATED_DISTRACTORS)
        self.assertEqual(set(v2.DISTRACTORS) - set(v2.GATED_DISTRACTORS), {"homograph"})


class TestShippedDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(DATA, encoding="utf-8") as f:
            cls.rows = [json.loads(line) for line in f if line.strip()]
        with open(MANIFEST, encoding="utf-8") as f:
            cls.manifest = json.load(f)

    def test_sha_matches_manifest(self):
        with open(DATA, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(sha, self.manifest["files"]["short_eval_neg_v1"]["sha256"])

    def test_every_row_valid_and_name_free(self):
        for r in self.rows:
            self.assertEqual(v2.validate_negative_row(r), [], r["id"])

    def test_counts_match_manifest(self):
        self.assertEqual(len(self.rows), self.manifest["counts"]["rows"])
        gated = sum(r["distractor"] in v2.GATED_DISTRACTORS for r in self.rows)
        self.assertEqual(gated, self.manifest["counts"]["gated_rows"])

    def test_ids_unique(self):
        ids = [r["id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))

    def test_hard_capitalized_cases_kept(self):
        ids = {r["id"] for r in self.rows}
        self.assertIn("short-eval-neg-v1-0128", ids)
        self.assertIn("short-eval-neg-v1-0133", ids)


if __name__ == "__main__":
    unittest.main()
