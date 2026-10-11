"""EXPERIMENTS.yaml registry tests: schema, required fields, uniqueness, lock presence."""

import os
import unittest

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED = {
    "run_id",
    "kind",
    "milestone",
    "executed",
    "where",
    "repo_commit",
    "artifacts",
    "checks",
    "metrics",
    "outcome",
}


class TestExperimentsRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "EXPERIMENTS.yaml"), encoding="utf-8") as f:
            cls.registry = yaml.safe_load(f)

    def test_loads_and_has_locks(self):
        self.assertEqual(self.registry["schema"], 1)
        for key in ("final_eval_sha256", "request_bank_sha256", "d0_model_sha256"):
            self.assertEqual(len(self.registry["locks"][key]), 64, key)

    def test_runs_exist_with_required_fields(self):
        runs = self.registry["runs"]
        self.assertGreaterEqual(len(runs), 7)
        for entry in runs:
            missing = REQUIRED - set(entry)
            self.assertFalse(missing, f"{entry.get('run_id')} missing {missing}")

    def test_run_ids_unique(self):
        ids = [e["run_id"] for e in self.registry["runs"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_checks_wellformed(self):
        for entry in self.registry["runs"]:
            checks = entry["checks"]
            self.assertIsInstance(checks.get("pass"), int, entry["run_id"])
            self.assertIsInstance(checks.get("fail"), int, entry["run_id"])
            self.assertGreaterEqual(checks["pass"] + checks["fail"], 1, entry["run_id"])

    def test_negative_results_not_deleted(self):
        outcomes = " ".join(e["outcome"] for e in self.registry["runs"])
        self.assertIn("NEGATIVE RESULT", outcomes)


if __name__ == "__main__":
    unittest.main()
