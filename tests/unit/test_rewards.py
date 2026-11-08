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

    def test_name_hidden_in_longer_word_gets_invalid_reward(self):
        r = rewards.score_candidate(
            text="The Annual review is on Monday", supplied_name="Ann", detector=FakeDetector()
        )
        self.assertFalse(r["valid"])
        self.assertEqual(r["total"], rewards.INVALID_TOTAL)

    def test_detector_scores_the_real_mention_not_an_earlier_substring(self):
        class RecordingDetector:
            def score_target(self, text, start, end):
                self.target = text[start:end], start
                return 0.5

        detector = RecordingDetector()
        rewards.score_candidate(
            text="The Annual note went to Ann today", supplied_name="Ann", detector=detector
        )
        self.assertEqual(detector.target, ("Ann", 24))


if __name__ == "__main__":
    unittest.main()
