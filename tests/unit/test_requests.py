"""Generation-request schema tests: validation, bank import, sampler determinism."""

import os
import unittest

from pii_redteam import requests as req

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BANK = os.path.join(ROOT, "tests", "fixtures", "spark_bank.jsonl")
PROVENANCE = {"brief": "test brief", "created_by": "test", "created_utc": "2026-09-20"}


class TestRequests(unittest.TestCase):
    def test_valid_records_pass(self):
        bank = req.load_bank(BANK, provenance=PROVENANCE)
        self.assertEqual(bank["manifest"]["count"], 2)
        self.assertEqual(bank["manifest"]["by_source"], ["spark_authored"])
        self.assertEqual(len(bank["manifest"]["sha256"]), 64)

    def test_bank_hash_stable(self):
        self.assertEqual(
            req.load_bank(BANK, provenance=PROVENANCE)["manifest"]["sha256"],
            req.load_bank(BANK, provenance=PROVENANCE)["manifest"]["sha256"],
        )

    def test_provenance_required(self):
        with self.assertRaises(ValueError):
            req.load_bank(BANK, provenance={"brief": "x"})

    def test_bad_source_rejected(self):
        rec = {"request_id": "r", "request_source": "guessed", "task": {"entities": []}}
        self.assertTrue(req.validate_request(rec))

    def test_contradictory_prompt_rejected(self):
        rec = {
            "request_id": "r",
            "request_source": "spark_authored",
            "task": {"entities": [{"type": "PERSON", "value": "Ana Ruiz"}]},
            "prompt": "Write about Paris with no names.",
        }
        errors = req.validate_request(rec)
        self.assertTrue(any("contradicts" in e for e in errors))

    def test_render_prompt_contains_values_once(self):
        bank = req.load_bank(BANK, provenance=PROVENANCE)
        prompt = req.render_prompt(bank["records"][0])
        self.assertEqual(prompt.count("Élodie Martin"), 1)

    def test_sampler_deterministic_and_seeded(self):
        names = ["Ana Ruiz", "Jean Dupont", "Élodie Martin"]
        first = req.seeded_sampler(names, seed=7)
        again = req.seeded_sampler(names, seed=7)
        other = req.seeded_sampler(names, seed=8)
        self.assertEqual(first, again)
        self.assertNotEqual([r["request_id"] for r in first], [r["request_id"] for r in other])
        for r in first:
            self.assertEqual(req.validate_request(r), [])


if __name__ == "__main__":
    unittest.main()
