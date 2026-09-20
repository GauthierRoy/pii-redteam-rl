"""Smoke integration test: isolated dirs, deterministic report, resume guard end to end."""

import json
import os
import tempfile
import unittest

from pii_redteam import config, experiments
from pii_redteam.manifest import ResumeError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_smoke_inputs():
    cfg = config.resolve_config(
        config.load_config(os.path.join(ROOT, "configs", "smoke", "smoke.yaml"))
    )
    with open(os.path.join(ROOT, "tests", "fixtures", "examples.jsonl")) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    return cfg, examples


class TestSmoke(unittest.TestCase):
    def test_end_to_end_writes_manifest_and_report(self):
        cfg, examples = load_smoke_inputs()
        with tempfile.TemporaryDirectory() as d:
            report = experiments.run_smoke(cfg, os.path.join(d, "run"), examples)
            run = os.path.join(d, "run")
            self.assertTrue(os.path.exists(os.path.join(run, "manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(run, "resolved_config.yaml")))
            self.assertTrue(os.path.exists(os.path.join(run, "report.json")))
            self.assertTrue(report["mock"])
            self.assertGreater(report["counts"]["candidates"], 0)
            self.assertEqual(
                report["counts"]["valid"] + report["counts"]["invalid"],
                report["counts"]["candidates"],
            )

    def test_report_deterministic_across_dirs(self):
        cfg, examples = load_smoke_inputs()
        with tempfile.TemporaryDirectory() as d:
            r1 = experiments.run_smoke(cfg, os.path.join(d, "a"), examples)
            r2 = experiments.run_smoke(cfg, os.path.join(d, "b"), examples)
            self.assertEqual(r1, r2)

    def test_rerun_into_same_dir_without_resume_fails(self):
        cfg, examples = load_smoke_inputs()
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "run")
            experiments.run_smoke(cfg, run, examples)
            with self.assertRaises(ResumeError):
                experiments.run_smoke(cfg, run, examples)

    def test_bad_fixture_fails_loudly(self):
        cfg, _ = load_smoke_inputs()
        bad = [{"example_id": "bad", "text": "Hi", "spans": "not-a-list"}]
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                experiments.run_smoke(cfg, os.path.join(d, "run"), bad)


if __name__ == "__main__":
    unittest.main()
