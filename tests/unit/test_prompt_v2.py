"""Schema v2 + prompt-variant + comparison-metric tests (M07b, pure functions)."""

import json
import os
import unittest

from pii_redteam import requests as req
from pii_redteam.prompt_v2 import (
    FEWSHOT_EXAMPLES,
    FEWSHOT_NAMES,
    comparison_metrics,
    name_position_bin,
    opens_with_name,
    render_prompt_v2,
    variant_threshold_report,
)
from pii_redteam.validity_v2 import classify_sample_v2, raw_text_for

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
POOLS = os.path.join(ROOT, "artifacts", "splits", "name_pools.json")
BANK_V2 = os.path.join(ROOT, "banks", "v2_prompting", "spark_seed20260920_v2_n20.jsonl")


def v2_request(**overrides) -> dict:
    task = {
        "language": "en",
        "entities": [{"type": "PERSON", "value": "Ana Ruiz"}],
        "constraints": {"occurrences_per_entity": 1, "no_additional_person_names": True},
        "context_type": "chat_message",
        "target_length": {"min_words": 20, "max_words": 60},
    }
    task.update(overrides.pop("task", {}))
    record = {
        "request_id": "r-v2",
        "request_source": "spark_authored",
        "schema_version": 2,
        "task": task,
    }
    record.update(overrides)
    return record


class TestSchemaV2(unittest.TestCase):
    def test_valid_v2_record(self):
        self.assertEqual(req.validate_request(v2_request()), [])

    def test_bad_context_type_rejected(self):
        rec = v2_request(task={"context_type": "carrier_pigeon"})
        self.assertTrue(any("context_type" in e for e in req.validate_request(rec)))

    def test_bad_target_length_rejected(self):
        rec = v2_request(task={"target_length": {"min_words": 60, "max_words": 20}})
        self.assertTrue(any("target_length" in e for e in req.validate_request(rec)))

    def test_bad_name_position_rejected(self):
        rec = v2_request(task={"name_position": "upside_down"})
        self.assertTrue(any("name_position" in e for e in req.validate_request(rec)))

    def test_missing_context_type_rejected(self):
        rec = v2_request()
        del rec["task"]["context_type"]
        self.assertTrue(any("context_type" in e for e in req.validate_request(rec)))

    def test_unknown_schema_version_rejected(self):
        rec = v2_request()
        rec["schema_version"] = 3
        self.assertTrue(any("schema_version" in e for e in req.validate_request(rec)))

    def test_v1_records_unchanged(self):
        rec = {
            "request_id": "r",
            "request_source": "spark_authored",
            "task": {"entities": [{"type": "PERSON", "value": "Ana Ruiz"}]},
        }
        self.assertEqual(req.validate_request(rec), [])

    def test_sampler_v2_path(self):
        names = ["Ana Ruiz", "Jean Dupont"]
        extras = [
            {
                "context_type": "chat_message",
                "target_length": {"min_words": 20, "max_words": 60},
            },
            {
                "context_type": "meeting_notes",
                "target_length": {"min_words": 20, "max_words": 60},
                "name_position": "end",
            },
        ]
        reqs = req.seeded_sampler(names, seed=7, task_extras=extras)
        self.assertEqual(reqs[0]["schema_version"], 2)
        self.assertEqual(req.validate_request(reqs[0]), [])
        self.assertEqual(reqs[1]["task"]["name_position"], "end")
        with self.assertRaises(ValueError):
            req.seeded_sampler(names, seed=7, task_extras=extras[:1])

    def test_bank_v2_imports_and_locks(self):
        if not os.path.exists(POOLS):
            self.skipTest("split artifacts not built")
        allowed, prov = req.load_person_name_pool(POOLS)
        bank = req.load_bank(
            BANK_V2,
            provenance={
                "brief": "test",
                "created_by": "test",
                "created_utc": "2026-10-12",
            },
            person_name_pool=(allowed, prov),
        )
        self.assertEqual(bank["manifest"]["count"], 20)
        self.assertEqual(len(bank["manifest"]["sha256"]), 64)
        types = [r["task"]["context_type"] for r in bank["records"]]
        self.assertEqual(len(set(types)), 10)
        self.assertTrue(all(r["schema_version"] == 2 for r in bank["records"]))


class TestPromptVariants(unittest.TestCase):
    def test_p0_is_baseline(self):
        rec = v2_request()
        self.assertEqual(render_prompt_v2(rec, "P0"), req.render_prompt(rec))

    def test_p1_adds_context_and_length(self):
        prompt = render_prompt_v2(v2_request(), "P1")
        self.assertIn("a casual chat message", prompt)
        self.assertIn("Target length: 20-60 words.", prompt)
        self.assertNotIn("example", prompt.lower())

    def test_p1_renders_name_position_hint(self):
        rec = v2_request(task={"name_position": "end"})
        self.assertIn("Place the name near the end.", render_prompt_v2(rec, "P1"))

    def test_p2_adds_fewshot(self):
        prompt = render_prompt_v2(v2_request(), "P2")
        self.assertIn("Two examples of the style", prompt)
        for name in FEWSHOT_NAMES:
            self.assertIn(name, prompt)
        # few-shot examples are single JSON objects with a text field
        for _, example in FEWSHOT_EXAMPLES:
            parsed = json.loads(example)
            self.assertIn("text", parsed)

    def test_fewshot_names_not_at_start(self):
        for _, example in FEWSHOT_EXAMPLES:
            text = json.loads(example)["text"]
            for name in FEWSHOT_NAMES:
                if name in text:
                    self.assertFalse(
                        opens_with_name(text, (text.index(name), text.index(name) + len(name)))
                    )

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            render_prompt_v2(v2_request(), "P3")


def sample(group_id, text, name="Ana Ruiz", status="valid"):
    v2 = classify_sample_v2(raw_text_for(text), name, False)
    return {
        "group_id": group_id,
        "request_id": group_id,
        "name": name,
        "status": v2["status"],
        "text": v2["text"],
        "target_span": v2["target_span"],
    }


class TestComparisonMetrics(unittest.TestCase):
    def test_all_valid_cell(self):
        samples = [
            sample(
                "g1",
                f"Ana Ruiz noted the shipment of record {i} arrives on Tuesday "
                "morning at the dock office.",
            )
            for i in range(8)
        ]
        metrics = comparison_metrics(samples)
        self.assertEqual(metrics["validity_rate"], 1.0)
        self.assertEqual(metrics["name_at_start_share"], 1.0)
        self.assertEqual(metrics["unique_per_group_median"], 8)

    def test_mixed_failure_breakdown(self):
        samples = [sample("g1", "Ana Ruiz mentioned the weather is perfect.")] * 8
        samples += [sample("g2", "Ana Ruiz noted the shipment arrives Tuesday at the dock.")]
        metrics = comparison_metrics(samples)
        self.assertEqual(metrics["failure_breakdown"], {"invalid:too_short": 8})
        self.assertAlmostEqual(metrics["validity_rate"], 1 / 9, places=3)

    def test_opening_patterns_collapse(self):
        texts = [
            "Ana Ruiz noted the shipment of record arrives on Tuesday morning." for _ in range(8)
        ] + [f"Please note that Ana Ruiz confirmed the update number {i} today." for i in range(2)]
        samples = [sample(f"g{i}", t) for i, t in enumerate(texts)]
        metrics = comparison_metrics(samples)
        self.assertEqual(metrics["top10_opening_share"], 1.0)
        self.assertEqual(len(metrics["top10_openings"]), 2)

    def test_name_position_bins(self):
        self.assertEqual(name_position_bin((0, 3), "abc def ghi jkl"), "start")
        self.assertEqual(name_position_bin((4, 7), "abc def ghi jkl abc"), "middle")
        self.assertEqual(name_position_bin((11, 14), "abcdefghijk lmn"), "end")

    def test_threshold_report(self):
        # 8 too-short samples + 1 valid -> validity fails, name-at-start fails
        samples = [sample("g1", "Ana Ruiz mentioned the weather is perfect.")] * 8
        samples.append(sample("g1", "Ana Ruiz noted the shipment arrives Tuesday morning."))
        metrics = comparison_metrics(samples)
        thresholds = {
            "validity_min": 0.90,
            "name_at_start_max": 0.40,
            "top10_openings_max": 0.30,
            "median_unique_valid_min": 6,
        }
        report = variant_threshold_report(metrics, thresholds)
        self.assertFalse(report["validity"]["ok"])
        self.assertFalse(report["name_at_start"]["ok"])
        self.assertFalse(report["top10_openings"]["ok"])
        self.assertFalse(report["unique_per_group"]["ok"])


if __name__ == "__main__":
    unittest.main()
