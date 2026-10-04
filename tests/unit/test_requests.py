"""Generation-request schema tests: validation, bank import, sampler determinism."""

import json
import os
import tempfile
import unittest

from pii_redteam import requests as req

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BANK = os.path.join(ROOT, "tests", "fixtures", "spark_bank.jsonl")
PROVENANCE = {"brief": "test brief", "created_by": "test", "created_utc": "2026-09-20"}


def write_pool(names: list[str]) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(
            {
                "version": 1,
                "seed": 0,
                "train_side_pool": [{"value": n, "count": 1} for n in names],
                "heldout_names": [{"value": "Held Out", "count": 1, "origin": "final_eval_unique"}],
                "heldout_rule": "test",
            },
            f,
        )
    return path


def request(value: str) -> dict:
    return {
        "request_id": f"r-{value[:6]}",
        "request_source": "spark_authored",
        "task": {"entities": [{"type": "PERSON", "value": value}]},
    }


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

    def test_person_name_pool_enforced(self):
        pool_path = write_pool(["Ana Ruiz", "Jean Dupont"])
        try:
            allowed, pool_prov = req.load_person_name_pool(pool_path)
            self.assertEqual(allowed, frozenset({"Ana Ruiz", "Jean Dupont"}))
            self.assertEqual(pool_prov["size"], 2)
            self.assertEqual(len(pool_prov["sha256"]), 64)
            self.assertEqual(req.validate_request(request("Ana Ruiz"), allowed), [])
            errors = req.validate_request(request("Held Out"), allowed)
            self.assertTrue(any("train-side name pool" in e for e in errors))
        finally:
            os.unlink(pool_path)

    def test_bank_rejects_out_of_pool_name(self):
        pool_path = write_pool(["Ana Ruiz"])
        bank_fd, bank_path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(bank_fd, "w") as f:
            f.write(json.dumps(request("Ana Ruiz")) + "\n")
            f.write(json.dumps(request("Held Out")) + "\n")
        try:
            pool = req.load_person_name_pool(pool_path)
            with self.assertRaises(ValueError) as ctx:
                req.load_bank(bank_path, provenance=PROVENANCE, person_name_pool=pool)
            self.assertIn("train-side name pool", str(ctx.exception))
            ok = req.load_bank(bank_path, provenance=PROVENANCE)
            self.assertIsNone(ok["manifest"]["person_name_pool_provenance"])
        finally:
            os.unlink(pool_path)
            os.unlink(bank_path)

    def test_bank_records_pool_provenance(self):
        pool_path = write_pool(["Ana Ruiz"])
        bank_fd, bank_path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(bank_fd, "w") as f:
            f.write(json.dumps(request("Ana Ruiz")) + "\n")
        try:
            pool = req.load_person_name_pool(pool_path)
            bank = req.load_bank(bank_path, provenance=PROVENANCE, person_name_pool=pool)
            self.assertEqual(bank["manifest"]["person_name_pool_provenance"]["size"], 1)
        finally:
            os.unlink(pool_path)
            os.unlink(bank_path)

    def test_sampler_rejects_out_of_pool_name(self):
        allowed = frozenset({"Ana Ruiz"})
        req.seeded_sampler(["Ana Ruiz"], seed=7, allowed_names=allowed)
        with self.assertRaises(ValueError) as ctx:
            req.seeded_sampler(["Held Out"], seed=7, allowed_names=allowed)
        self.assertIn("train-side pool", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
