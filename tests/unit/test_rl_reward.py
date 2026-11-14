"""GRPO reward adapter: contract statuses, reward ordering, truncation, logging, groups."""

import json
import unittest

from pii_redteam.training.opening_share import OpeningShare
from pii_redteam.training.rl_reward import (
    INVALID_REWARD,
    REPEATED_OPENING_REWARD,
    TargetDetectorReward,
)

EOS = 7
GOOD_TEXT = "Hi Lucas, could you please call me back about the invoice we discussed yesterday?"
OTHER_TEXT = "Yesterday Lucas wrote a long note about the delayed shipment and the missing invoice."
AMARA_TEXT = "Please tell the front desk that Amara has arrived for the appointment this morning."


class FakeScorer:
    def __init__(self, p_by_text: dict[str, float | None]) -> None:
        self.p_by_text = p_by_text
        self.scored_texts: list[str] = []

    def score(self, samples: list[dict]) -> None:
        for sample in samples:
            if sample["status"] != "valid":
                continue
            self.scored_texts.append(sample["text"])
            sample["p_target_mean"] = self.p_by_text[sample["text"]]
            p_target = sample["p_target_mean"]
            if p_target is not None:
                sample["outcome"] = "complete_miss" if p_target < 0.5 else "exact_hit"


def message(text: str) -> str:
    return json.dumps({"text": text})


def reward_for(completions, names, p_by_text, completion_ids=None, **kwargs):
    reward = TargetDetectorReward(FakeScorer(p_by_text), EOS)
    return reward(completions, names, completion_ids=completion_ids, **kwargs)


class TestTargetDetectorReward(unittest.TestCase):
    def test_valid_reward_is_one_minus_detector_probability(self):
        text = GOOD_TEXT
        rewards = reward_for([message(text)], ["Lucas"], {text: 0.25})
        self.assertAlmostEqual(rewards[0], 0.75)

    def test_invalid_statuses_get_the_invalid_reward_below_every_valid_reward(self):
        good = GOOD_TEXT
        completions = [
            message(good),
            "not json at all",
            message("Hi there, call me."),
            message("Lucas and Lucas again."),
            message("Hi Lucas"),
        ]
        names = ["Lucas"] * 5
        completion_ids = [[1, EOS], [1, EOS], [1, EOS], [1, EOS], [1, 2]]
        rewards = reward_for(
            completions, names, {good: 0.999, "Hi Lucas": 0.0}, completion_ids=completion_ids
        )
        self.assertAlmostEqual(rewards[0], 0.001)
        self.assertEqual(rewards[1:], [INVALID_REWARD] * 4)
        self.assertLess(INVALID_REWARD, min(rewards[0], 0.0))

    def test_short_text_that_v1_accepted_is_invalid_and_never_scored(self):
        scorer = FakeScorer({})
        rewards = TargetDetectorReward(scorer, EOS)(
            [message("Hi Lucas")], ["Lucas"], completion_ids=[[1, EOS]]
        )
        self.assertEqual(rewards, [INVALID_REWARD])
        self.assertEqual(scorer.scored_texts, [])

    def test_truncated_completion_is_invalid_even_when_its_json_is_complete(self):
        text = GOOD_TEXT
        rewards = reward_for([message(text)], ["Lucas"], {text: 0.0}, completion_ids=[[1, 2, 3]])
        self.assertEqual(rewards, [INVALID_REWARD])

    def test_missing_completion_ids_do_not_flag_truncation(self):
        text = GOOD_TEXT
        rewards = reward_for([message(text)], ["Lucas"], {text: 0.0})
        self.assertEqual(rewards, [1.0])

    def test_invalid_samples_are_never_sent_to_the_scorer(self):
        scorer = FakeScorer({})
        TargetDetectorReward(scorer, EOS)(["garbage", message("no name here")], ["Lucas", "Lucas"])
        self.assertEqual(scorer.scored_texts, [])

    def test_scorer_returning_no_probability_is_invalid(self):
        text = GOOD_TEXT
        rewards = reward_for([message(text)], ["Lucas"], {text: None})
        self.assertEqual(rewards, [INVALID_REWARD])

    def test_conversational_completions_are_unwrapped(self):
        text = GOOD_TEXT
        completion = [{"role": "assistant", "content": message(text)}]
        rewards = reward_for([completion], ["Lucas"], {text: 0.5})
        self.assertAlmostEqual(rewards[0], 0.5)

    def test_all_invalid_group_has_zero_reward_variance(self):
        rewards = reward_for(["bad"] * 8, ["Lucas"] * 8, {})
        self.assertEqual(set(rewards), {INVALID_REWARD})

    def test_metrics_count_invalid_completions_in_the_denominator(self):
        a, b = GOOD_TEXT, OTHER_TEXT
        logged = {}
        rewards = reward_for(
            [message(a), message(b), "bad", "bad"],
            ["Lucas"] * 4,
            {a: 0.1, b: 0.9},
            log_metric=lambda key, value: logged.__setitem__(key, value),
        )
        self.assertEqual(len(rewards), 4)
        self.assertEqual(logged["valid_rate"], 0.5)
        self.assertEqual(logged["true_miss_rate"], 0.25)
        self.assertEqual(logged["truncated_rate"], 0.0)

    def test_extra_columns_are_logged_per_completion(self):
        text = GOOD_TEXT
        extra = {}
        reward_for(
            [message(text), "bad"],
            ["Lucas", "Lucas"],
            {text: 0.2},
            log_extra=lambda column, values: extra.__setitem__(column, values),
        )
        self.assertEqual(extra["status"], ["valid", "invalid:json"])
        self.assertEqual(extra["p_target_mean"], [0.2, None])

    def test_rewards_follow_the_input_order_across_prompt_groups(self):
        a, b = GOOD_TEXT, AMARA_TEXT
        rewards = reward_for(
            [message(a), message(a), message(b), message(b)],
            ["Lucas", "Lucas", "Amara", "Amara"],
            {a: 0.3, b: 0.6},
        )
        self.assertEqual([round(r, 1) for r in rewards], [0.7, 0.7, 0.4, 0.4])

    def collapse_batch(self):
        """Three requests; every text opens with "Dear" except Amara's second one."""
        names = ["Lucas", "Lucas", "Amara", "Amara", "Chidi", "Chidi"]
        texts = [
            "Dear Lucas, the invoice for the March delivery is attached for your review.",
            "Dear Lucas, please confirm the delivery window for the replacement parts today.",
            "Dear Amara, your appointment at the clinic is confirmed for Thursday at three.",
            "Row 14 | Amara | keys returned at 09:40 | signed out by the front desk",
            "Dear Chidi, thank you for reporting the broken latch on the west stairwell door.",
            "Dear Chidi, the committee deferred the renewal decision until the broker replies.",
        ]
        return names, texts

    def guarded_reward(self, texts):
        guard = OpeningShare(threshold=0.5, window=64, min_other_texts=4)
        return TargetDetectorReward(FakeScorer(dict.fromkeys(texts, 0.1)), EOS, guard)

    def test_opening_shared_across_requests_loses_the_detector_reward(self):
        names, texts = self.collapse_batch()
        rewards = self.guarded_reward(texts)([message(t) for t in texts], names)
        # Lucas and Chidi: 3 of the 4 other texts open with "dear"; Amara's: all 4 do
        self.assertEqual(rewards[0], REPEATED_OPENING_REWARD)
        self.assertEqual(rewards[2], REPEATED_OPENING_REWARD)
        self.assertAlmostEqual(rewards[3], 0.9)
        self.assertGreater(REPEATED_OPENING_REWARD, INVALID_REWARD)

    def test_guard_groups_by_request_id_when_given(self):
        names, texts = self.collapse_batch()
        one_request = ["r1"] * len(texts)
        rewards = self.guarded_reward(texts)(
            [message(t) for t in texts], names, request_id=one_request
        )
        # a single request has no other requests to share an opening with
        self.assertEqual([round(r, 1) for r in rewards], [0.9] * 6)

    def test_guard_ignores_invalid_completions(self):
        names, texts = self.collapse_batch()
        completions = [message(t) for t in texts] + ["Dear nobody, not json"]
        extra = {}
        rewards = self.guarded_reward(texts)(
            completions, names + ["Lucas"], log_extra=extra.__setitem__
        )
        self.assertEqual(rewards[-1], INVALID_REWARD)
        self.assertIsNone(extra["opening_share"][-1])
        self.assertEqual(extra["opening_share"][3], 0.0)

    def test_guard_metrics_are_logged(self):
        names, texts = self.collapse_batch()
        logged = {}
        self.guarded_reward(texts)(
            [message(t) for t in texts], names, log_metric=logged.__setitem__
        )
        self.assertAlmostEqual(logged["repeated_opening_rate"], 5 / 6)
        self.assertAlmostEqual(logged["opening_share_mean"], (0.75 * 4 + 1.0 + 0.0) / 6)

    def test_without_the_guard_nothing_about_openings_is_logged(self):
        names, texts = self.collapse_batch()
        logged = {}
        rewards = TargetDetectorReward(FakeScorer(dict.fromkeys(texts, 0.1)), EOS)(
            [message(t) for t in texts], names, log_metric=logged.__setitem__
        )
        self.assertEqual([round(r, 1) for r in rewards], [0.9] * 6)
        self.assertNotIn("repeated_opening_rate", logged)

    def test_trl_expects_a_function_name_for_logging(self):
        self.assertEqual(
            TargetDetectorReward(FakeScorer({}), EOS).__name__, "target_detector_reward"
        )


if __name__ == "__main__":
    unittest.main()
