"""RL request bank: seeded, train-side names only, balanced contexts, disjoint from banks."""

import json
import os
import unittest

from pii_redteam.requests import CONTEXT_TYPES, validate_request
from pii_redteam.training.rl_bank import bank_names, build_rl_requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2_BANK = os.path.join(ROOT, "banks", "v2_prompting", "spark_seed20260920_v2_n20.jsonl")
POOL = frozenset(f"Firstname{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(300))


class TestBuildRlRequests(unittest.TestCase):
    def test_requests_are_valid_v2_with_unique_pool_names(self):
        requests = build_rl_requests(POOL, n=40, seed=1)
        self.assertEqual(len(requests), 40)
        names = [r["task"]["entities"][0]["value"] for r in requests]
        self.assertEqual(len(set(names)), 40)
        self.assertTrue(set(names) <= POOL)
        for request in requests:
            self.assertEqual(request["schema_version"], 2)
            self.assertEqual(validate_request(request, POOL), [])

    def test_context_types_are_balanced(self):
        requests = build_rl_requests(POOL, n=40, seed=1)
        counts = {c: 0 for c in CONTEXT_TYPES}
        for request in requests:
            counts[request["task"]["context_type"]] += 1
        self.assertEqual(set(counts.values()), {4})

    def test_same_seed_gives_the_same_bank_and_other_seed_differs(self):
        self.assertEqual(
            build_rl_requests(POOL, n=20, seed=1), build_rl_requests(POOL, n=20, seed=1)
        )
        self.assertNotEqual(
            build_rl_requests(POOL, n=20, seed=1), build_rl_requests(POOL, n=20, seed=2)
        )

    def test_excluded_names_never_appear(self):
        first = build_rl_requests(POOL, n=50, seed=1)
        taken = frozenset(r["task"]["entities"][0]["value"] for r in first)
        second = build_rl_requests(POOL, n=50, seed=2, exclude=taken)
        self.assertFalse(taken & {r["task"]["entities"][0]["value"] for r in second})

    def test_signature_position_only_for_contexts_with_a_signature(self):
        for request in build_rl_requests(POOL, n=200, seed=3):
            task = request["task"]
            if task.get("name_position") == "signature":
                self.assertIn(task["context_type"], ("email_with_signature", "form_record_line"))

    def test_more_requests_than_available_names_is_rejected(self):
        with self.assertRaises(ValueError):
            build_rl_requests(POOL, n=len(POOL) + 1, seed=1)

    def test_bank_names_reads_every_requested_person(self):
        with open(V2_BANK, encoding="utf-8") as f:
            expected = {
                json.loads(line)["task"]["entities"][0]["value"] for line in f if line.strip()
            }
        self.assertEqual(bank_names(V2_BANK), expected)


if __name__ == "__main__":
    unittest.main()
