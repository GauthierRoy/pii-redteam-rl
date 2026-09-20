"""Reward tests: invalid below the valid range, components logged separately."""

import unittest

from pii_redteam import rewards
from pii_redteam.detector import FakeDetector


class TestRewards(unittest.TestCase):
    def test_valid_candidate_scores_in_range(self):
        r = rewards.score_candidate(
            text="Hello, I am Ana Ruiz today.", supplied_name="Ana Ruiz", detector=FakeDetector()
        )
        self.assertTrue(r["valid"])
        self.assertGreaterEqual(r["total"], rewards.VALID_MIN)
        self.assertLessEqual(r["total"], rewards.VALID_MAX)
        for key in ("validity", "difficulty", "repetition"):
            self.assertIn(key, r)

    def test_invalid_cannot_outrank_valid(self):
        bad = rewards.score_candidate(
            text="Hello there", supplied_name="Ana Ruiz", detector=FakeDetector()
        )
        good = rewards.score_candidate(
            text="Hello, I am Ana Ruiz today.", supplied_name="Ana Ruiz", detector=FakeDetector()
        )
        self.assertFalse(bad["valid"])
        self.assertLess(bad["total"], rewards.VALID_MIN)
        self.assertGreater(good["total"], bad["total"])

    def test_repetition_penalty_triggers_on_loops(self):
        self.assertEqual(rewards.repetition_penalty("short text here"), 0.0)
        loopy = " ".join(["bill"] * 20)
        self.assertLess(rewards.repetition_penalty(loopy), 0.0)


if __name__ == "__main__":
    unittest.main()
