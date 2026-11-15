"""GRPO reward adapter: contract statuses, reward ordering, truncation, logging, groups."""

import json
import unittest

from pii_redteam.training.opening_share import OpeningShare
from pii_redteam.training.person_judge import NOT_PERSON, PERSON, UNPARSED
from pii_redteam.training.rl_reward import (
    INVALID_REWARD,
    JUDGE_UNPARSED_STATUS,
    NOT_PERSON_STATUS,
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


class FakeJudge:
    def __init__(self, verdict_by_text: dict[str, str]) -> None:
        self.verdict_by_text = verdict_by_text
        self.asked: list[tuple[str, str]] = []

    def verdicts(self, names: list[str], texts: list[str]) -> list[str]:
        self.asked += list(zip(names, texts, strict=True))
        return [self.verdict_by_text[text] for text in texts]


class FakeAudit:
    def __init__(self, audit_by_text: dict[str, dict | None]) -> None:
        self.audit_by_text = audit_by_text
        self.contexts: list[str] = []

    def audits(self, names: list[str], texts: list[str], contexts: list[str]) -> list:
        self.contexts += contexts
        return [self.audit_by_text[text] for text in texts]


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
        guard = OpeningShare(
            full_reward_until=0.5, zero_reward_from=1.0, window=64, min_other_texts=4
        )
        return TargetDetectorReward(FakeScorer(dict.fromkeys(texts, 0.1)), EOS, guard)

    def test_opening_shared_across_requests_scales_the_detector_reward_down(self):
        names, texts = self.collapse_batch()
        rewards = self.guarded_reward(texts)([message(t) for t in texts], names)
        # Lucas and Chidi: 3 of the 4 other texts open with "dear" -> share 0.75 -> half reward
        self.assertAlmostEqual(rewards[0], 0.45)
        self.assertAlmostEqual(rewards[4], 0.45)
        # Amara's "dear" text: all 4 other texts do -> share 1.0 -> no reward, still above invalid
        self.assertEqual(rewards[2], 0.0)
        self.assertGreater(rewards[2], INVALID_REWARD)
        self.assertAlmostEqual(rewards[3], 0.9)

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

    def test_name_used_as_a_place_is_invalid_and_never_scored(self):
        place = "I just found a great taco spot near the Lucas for our movie night tonight."
        scorer = FakeScorer({GOOD_TEXT: 0.1})
        judge = FakeJudge({GOOD_TEXT: PERSON, place: NOT_PERSON})
        extra, logged = {}, {}
        rewards = TargetDetectorReward(scorer, EOS, person_judge=judge)(
            [message(GOOD_TEXT), message(place)],
            ["Lucas", "Lucas"],
            log_extra=extra.__setitem__,
            log_metric=logged.__setitem__,
        )
        self.assertAlmostEqual(rewards[0], 0.9)
        self.assertEqual(rewards[1], INVALID_REWARD)
        self.assertEqual(extra["status"], ["valid", NOT_PERSON_STATUS])
        self.assertEqual(scorer.scored_texts, [GOOD_TEXT])
        self.assertEqual(logged["not_person_rate"], 0.5)
        self.assertEqual(logged["valid_rate"], 0.5)

    def test_judge_only_sees_texts_that_pass_the_validity_rules(self):
        judge = FakeJudge({GOOD_TEXT: PERSON})
        TargetDetectorReward(FakeScorer({GOOD_TEXT: 0.1}), EOS, person_judge=judge)(
            [message(GOOD_TEXT), "not json", message("Hi Lucas")], ["Lucas"] * 3
        )
        self.assertEqual(judge.asked, [("Lucas", GOOD_TEXT)])

    def test_unparsed_judge_answer_is_invalid_under_its_own_status(self):
        judge = FakeJudge({GOOD_TEXT: UNPARSED})
        extra = {}
        rewards = TargetDetectorReward(FakeScorer({}), EOS, person_judge=judge)(
            [message(GOOD_TEXT)], ["Lucas"], log_extra=extra.__setitem__
        )
        self.assertEqual(rewards, [INVALID_REWARD])
        self.assertEqual(extra["status"], [JUDGE_UNPARSED_STATUS])

    def test_rejected_texts_do_not_enter_the_opening_window(self):
        names, texts = self.collapse_batch()
        judge = FakeJudge(
            {t: (NOT_PERSON if t.startswith("Dear Chidi") else PERSON) for t in texts}
        )
        guard = OpeningShare(
            full_reward_until=0.5, zero_reward_from=1.0, window=64, min_other_texts=2
        )
        reward = TargetDetectorReward(FakeScorer(dict.fromkeys(texts, 0.1)), EOS, guard, judge)
        reward([message(t) for t in texts], names)
        self.assertEqual(len(guard.recent), 4)

    def audited_reward(self, audit, judge=None):
        scorer = FakeScorer({GOOD_TEXT: 0.1, OTHER_TEXT: 0.1})
        return TargetDetectorReward(scorer, EOS, person_judge=judge, audit=audit)

    def test_audit_is_logged_and_never_changes_the_reward(self):
        bad = {"other_person": True, "natural": False, "fits_format": False}
        good = {"other_person": False, "natural": True, "fits_format": True}
        audit = FakeAudit({GOOD_TEXT: bad, OTHER_TEXT: good})
        extra, logged = {}, {}
        rewards = self.audited_reward(audit)(
            [message(GOOD_TEXT), message(OTHER_TEXT), "not json"],
            ["Lucas"] * 3,
            context=["a chat", "a report", "a chat"],
            log_extra=extra.__setitem__,
            log_metric=logged.__setitem__,
        )
        self.assertEqual([round(r, 1) for r in rewards], [0.9, 0.9, INVALID_REWARD])
        self.assertEqual(extra["other_person"], [True, False, None])
        self.assertEqual(extra["fits_format"], [False, True, None])
        self.assertEqual(audit.contexts, ["a chat", "a report"])
        self.assertEqual(logged["audit_other_person_rate"], 0.5)
        self.assertEqual(logged["audit_unnatural_rate"], 0.5)
        self.assertEqual(logged["audit_off_format_rate"], 0.5)

    def test_failed_audit_is_logged_as_missing(self):
        audit = FakeAudit({GOOD_TEXT: None})
        extra, logged = {}, {}
        rewards = self.audited_reward(audit)(
            [message(GOOD_TEXT)],
            ["Lucas"],
            context=["a chat"],
            log_extra=extra.__setitem__,
            log_metric=logged.__setitem__,
        )
        self.assertAlmostEqual(rewards[0], 0.9)
        self.assertEqual(extra["natural"], [None])
        self.assertNotIn("audit_unnatural_rate", logged)

    def test_audit_rates_leave_out_texts_the_person_judge_rejected(self):
        flagged = {"other_person": True, "natural": True, "fits_format": True}
        clean = {"other_person": False, "natural": True, "fits_format": True}
        audit = FakeAudit({GOOD_TEXT: clean, OTHER_TEXT: flagged})
        judge = FakeJudge({GOOD_TEXT: PERSON, OTHER_TEXT: NOT_PERSON})
        logged = {}
        self.audited_reward(audit, judge)(
            [message(GOOD_TEXT), message(OTHER_TEXT)],
            ["Lucas"] * 2,
            context=["a chat", "a chat"],
            log_metric=logged.__setitem__,
        )
        self.assertEqual(logged["audit_other_person_rate"], 0.0)

    def test_audit_is_skipped_without_the_context_column(self):
        audit = FakeAudit({})
        rewards = self.audited_reward(audit)([message(GOOD_TEXT)], ["Lucas"])
        self.assertAlmostEqual(rewards[0], 0.9)
        self.assertEqual(audit.contexts, [])

    def test_trl_expects_a_function_name_for_logging(self):
        self.assertEqual(
            TargetDetectorReward(FakeScorer({}), EOS).__name__, "target_detector_reward"
        )


if __name__ == "__main__":
    unittest.main()
