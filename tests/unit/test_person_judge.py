"""Person judge: answer parsing, retries, outage handling, order of verdicts."""

import unittest

from pii_redteam.training.person_judge import (
    NOT_PERSON,
    PERSON,
    UNPARSED,
    JudgeUnavailableError,
    PersonJudge,
    TextAudit,
    answer_text,
    parse_audit,
    parse_verdict,
)


def reply(content) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def judge(post, **overrides) -> PersonJudge:
    return PersonJudge(model="fake", post=post, retry_seconds=0.0, **overrides)


class TestParsing(unittest.TestCase):
    def test_yes_and_no_with_punctuation_and_case(self):
        self.assertEqual(parse_verdict("Yes."), PERSON)
        self.assertEqual(parse_verdict(" no\n"), NOT_PERSON)
        self.assertEqual(parse_verdict("**No**"), NOT_PERSON)

    def test_anything_else_is_unparsed(self):
        self.assertEqual(parse_verdict(""), UNPARSED)
        self.assertEqual(parse_verdict("It depends"), UNPARSED)

    def test_reasoning_blocks_are_skipped(self):
        content = [
            {"type": "thinking", "thinking": [{"type": "text", "text": "no, wait, yes"}]},
            {"type": "text", "text": "yes"},
        ]
        self.assertEqual(answer_text(reply(content)), "yes")


class TestPersonJudge(unittest.TestCase):
    def test_verdicts_keep_the_input_order(self):
        def post(body):
            question = body["messages"][0]["content"]
            return reply("no" if "Mulaton store" in question else "yes")

        texts = ["Hi Bob, call me.", "Visit the Mulaton store today.", "Thanks, Amara."]
        verdicts = judge(post).verdicts(["Bob", "Mulaton", "Amara"], texts)
        self.assertEqual(verdicts, [PERSON, NOT_PERSON, PERSON])

    def test_question_carries_the_name_and_the_text(self):
        seen = []

        def post(body):
            seen.append(body)
            return reply("yes")

        judge(post).verdicts(["El Bez"], ["Meet El Bez at noon."])
        self.assertEqual(seen[0]["model"], "fake")
        self.assertEqual(seen[0]["temperature"], 0)
        self.assertIn('"El Bez"', seen[0]["messages"][0]["content"])
        self.assertIn("Meet El Bez at noon.", seen[0]["messages"][0]["content"])

    def test_transient_failure_is_retried(self):
        calls = []

        def post(body):
            calls.append(1)
            if len(calls) < 3:
                raise TimeoutError("slow")
            return reply("yes")

        self.assertEqual(judge(post).verdicts(["Bob"], ["Hi Bob."]), [PERSON])
        self.assertEqual(len(calls), 3)

    def test_outage_stops_the_run_instead_of_passing_texts(self):
        def post(body):
            raise OSError("connection refused")

        with self.assertRaises(JudgeUnavailableError):
            judge(post, attempts=2).verdicts(["Bob"], ["Hi Bob."])

    def test_unparsed_answer_is_asked_again_then_reported(self):
        calls = []

        def post(body):
            calls.append(1)
            return reply("maybe")

        self.assertEqual(judge(post, attempts=3).verdicts(["Bob"], ["Hi Bob."]), [UNPARSED])
        self.assertEqual(len(calls), 3)

    def test_no_texts_means_no_calls(self):
        def post(body):
            raise AssertionError("must not be called")

        self.assertEqual(judge(post).verdicts([], []), [])


class TestTextAudit(unittest.TestCase):
    GOOD = '{"other_person": false, "natural": true, "fits_format": true}'

    def test_object_is_parsed_even_inside_a_code_fence(self):
        expected = {"other_person": False, "natural": True, "fits_format": True}
        self.assertEqual(parse_audit(self.GOOD), expected)
        self.assertEqual(parse_audit(f"```json\n{self.GOOD}\n```"), expected)

    def test_missing_or_non_boolean_fields_are_rejected(self):
        self.assertIsNone(parse_audit('{"other_person": false, "natural": true}'))
        self.assertIsNone(
            parse_audit('{"other_person": "no", "natural": true, "fits_format": true}')
        )
        self.assertIsNone(parse_audit("yes"))
        self.assertIsNone(parse_audit("{not json}"))

    def test_question_carries_context_name_and_text(self):
        seen = []

        def post(body):
            seen.append(body["messages"][0]["content"])
            return reply(self.GOOD)

        audits = TextAudit("fake", post).audits(["Bob"], ["Hi Bob."], ["a casual chat message"])
        self.assertEqual(audits, [{"other_person": False, "natural": True, "fits_format": True}])
        for part in ("a casual chat message", '"Bob"', "Hi Bob."):
            self.assertIn(part, seen[0])

    def test_failed_call_gives_none_and_never_raises(self):
        def post(body):
            if "Amara" in body["messages"][0]["content"]:
                raise OSError("rate limited")
            return reply(self.GOOD)

        audits = TextAudit("fake", post).audits(
            ["Bob", "Amara"], ["Hi Bob.", "Hi Amara."], ["a chat", "a chat"]
        )
        self.assertIsNotNone(audits[0])
        self.assertIsNone(audits[1])


if __name__ == "__main__":
    unittest.main()
