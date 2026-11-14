"""RL dataset rows: same prompt as the validated P1 comparison, one name per row."""

import copy
import json
import os
import unittest

from pii_redteam.prompt_v2 import render_prompt_v2
from pii_redteam.training.rl_dataset import rl_rows

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BANK = os.path.join(ROOT, "banks", "v2_prompting", "spark_seed20260920_v2_n20.jsonl")


def load_records():
    with open(BANK, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestRlRows(unittest.TestCase):
    def test_one_row_per_request_with_the_requested_name(self):
        records = load_records()
        rows = rl_rows(records)
        self.assertEqual(len(rows), len(records))
        for record, row in zip(records, rows, strict=True):
            self.assertEqual(row["name"], record["task"]["entities"][0]["value"])
            self.assertEqual(row["request_id"], record["request_id"])

    def test_prompt_is_the_validated_p1_text_as_one_user_message(self):
        record = load_records()[0]
        (row,) = rl_rows([record])
        self.assertEqual(
            row["prompt"], [{"role": "user", "content": render_prompt_v2(record, "P1")}]
        )
        self.assertIn(f"Entity PERSON: {row['name']}", row["prompt"][0]["content"])

    def test_variant_changes_the_prompt(self):
        record = load_records()[0]
        p0 = rl_rows([record], variant="P0")[0]["prompt"][0]["content"]
        p1 = rl_rows([record], variant="P1")[0]["prompt"][0]["content"]
        self.assertNotEqual(p0, p1)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            rl_rows(load_records(), variant="P9")

    def test_schema_v1_request_is_rejected(self):
        record = copy.deepcopy(load_records()[0])
        del record["schema_version"]
        with self.assertRaises(ValueError):
            rl_rows([record])

    def test_request_with_several_entities_is_rejected(self):
        record = copy.deepcopy(load_records()[0])
        record["task"]["entities"].append({"type": "PERSON", "value": "Other"})
        with self.assertRaises(ValueError):
            rl_rows([record])

    def test_duplicate_request_ids_are_rejected(self):
        record = load_records()[0]
        with self.assertRaises(ValueError):
            rl_rows([record, record])


if __name__ == "__main__":
    unittest.main()
