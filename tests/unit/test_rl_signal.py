"""RL signal summary: group statistics over scored samples, valid and invalid mixed."""

import json
import unittest

from pii_redteam.training.rl_signal import signal_summary


def sample(request_id, text, reward, status="valid"):
    raw = json.dumps({"text": text}) if text is not None else "not json"
    return {"request_id": request_id, "raw_text": raw, "reward": reward, "status": status}


class TestSignalSummary(unittest.TestCase):
    def setUp(self):
        self.samples = [
            sample("a", "one two three", 0.9),
            sample("a", "one two three four five", 0.1),
            sample("b", "same words here", 0.2),
            sample("b", None, -1.0, "invalid:json"),
        ]
        self.summary = signal_summary(self.samples, {"valid_rate": 0.75})

    def test_counts_and_passthrough_metrics(self):
        self.assertEqual(self.summary["samples"], 4)
        self.assertEqual(self.summary["groups"], 2)
        self.assertEqual(self.summary["valid_rate"], 0.75)
        self.assertEqual(self.summary["status_counts"], {"valid": 3, "invalid:json": 1})

    def test_group_with_a_high_reward_counts_as_a_detector_miss(self):
        self.assertEqual(self.summary["groups_with_detector_miss"], 0.5)

    def test_reward_std_is_the_mean_of_per_group_population_std(self):
        self.assertAlmostEqual(self.summary["mean_group_reward_std"], (0.4 + 0.6) / 2)

    def test_valid_statistics_ignore_invalid_samples(self):
        self.assertAlmostEqual(self.summary["mean_valid_reward"], (0.9 + 0.1 + 0.2) / 3)
        self.assertAlmostEqual(self.summary["mean_valid_words"], (3 + 5 + 3) / 3)

    def test_all_invalid_gives_no_valid_statistics(self):
        summary = signal_summary([sample("a", None, -1.0, "invalid:json")], {})
        self.assertIsNone(summary["mean_valid_reward"])
        self.assertIsNone(summary["mean_valid_words"])

    def test_unique_text_share_per_group(self):
        samples = [sample("a", "x y z w v u t s", 0.3), sample("a", "x y z w v u t s", 0.3)]
        self.assertEqual(signal_summary(samples, {})["mean_unique_text_share_per_group"], 0.5)


if __name__ == "__main__":
    unittest.main()
