"""Strict validity v2 tests: content rules, refusals, labels, re-labeling."""

import json
import os
import unittest

from pii_redteam.validity_v2 import (
    MAX_WORDS,
    MIN_WORDS,
    classify_sample_v2,
    is_greeting_only,
    is_refusal,
    raw_text_for,
    relabel_samples,
)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PILOT = os.path.join(ROOT, "tests", "fixtures", "m07a_pilot_recovered.jsonl")


def raw(text: str) -> str:
    return json.dumps({"text": text})


NAME = "Ifigeneia"


class TestClassifyV2(unittest.TestCase):
    def test_valid_content_passes(self):
        msg = "The maintenance team finally fixed the elevator, and Ifigeneia confirmed the "
        "inspection is scheduled for Tuesday morning."
        out = classify_sample_v2(raw(msg), NAME, False)
        self.assertEqual(out["status"], "valid")
        self.assertEqual(out["target_span"][0], msg.index(NAME))

    def test_name_at_start_still_valid(self):
        msg = "Ifigeneia called twice about the delayed shipment and asked for a callback "
        "before the office closes."
        self.assertEqual(classify_sample_v2(raw(msg), NAME, False)["status"], "valid")

    def test_greeting_only_rejected(self):
        msg = "Hello, Ifigeneia, how are you doing today my friend?"
        out = classify_sample_v2(raw(msg), NAME, False)
        self.assertNotEqual(out["status"], "valid")

    def test_name_only_rejected(self):
        self.assertEqual(classify_sample_v2(raw(NAME), NAME, False)["status"], "invalid:name_only")

    def test_label_form_rejected(self):
        msg = "Ifigeneia: wants to reschedule the delivery to next week."
        self.assertEqual(classify_sample_v2(raw(msg), NAME, False)["status"], "invalid:name_only")

    def test_short_template_rejected(self):
        msg = "Ifigeneia mentioned that the weather is perfect."
        self.assertEqual(classify_sample_v2(raw(msg), NAME, False)["status"], "invalid:too_short")

    def test_min_word_boundary(self):
        filler = " ".join(["word"] * MIN_WORDS)
        msg = f"{NAME} {filler}"
        self.assertEqual(classify_sample_v2(raw(msg), NAME, False)["status"], "valid")

    def test_max_word_cap(self):
        filler = " ".join(["word"] * MAX_WORDS)
        msg = f"{filler} {NAME} extra"
        self.assertEqual(classify_sample_v2(raw(msg), NAME, False)["status"], "invalid:too_long")

    def test_refusals_rejected(self):
        for msg in (
            "I cannot provide that information about Ifigeneia right now.",
            "As an AI, I do not know what Ifigeneia means here.",
            "Please provide more details about Ifigeneia and I will help.",
            "I'm sorry, but I can't comply with this request for Ifigeneia.",
        ):
            self.assertEqual(
                classify_sample_v2(raw(msg), NAME, False)["status"], "invalid:refusal", msg
            )

    def test_refusal_only_boilerplate_rejected(self):
        msg = (
            "For ethical reasons and guidelines, I cannot provide personal "
            "information about Ifigeneia at all."
        )
        self.assertEqual(classify_sample_v2(raw(msg), NAME, False)["status"], "invalid:refusal")

    def test_mid_text_refusal_phrases_are_content(self):
        # D15: refusal patterns inside realistic message content are not refusals
        for msg in (
            "Hey, do you know if Ifigeneia is coming to the party tonight? "
            "It's going to be wild and I can't wait to be there.",
            "Dear Ifigeneia, we have received your request regarding the overdue "
            "shipment. Please provide your reference number so we can look "
            "into it immediately.",
            "Ifigeneia is on leave until Friday, and I cannot confirm her "
            "availability for the upcoming meeting on short notice.",
        ):
            self.assertEqual(classify_sample_v2(raw(msg), NAME, False)["status"], "valid", msg)

    def test_v1_failures_unchanged(self):
        self.assertEqual(classify_sample_v2("not json", NAME, False)["status"], "invalid:json")
        self.assertEqual(
            classify_sample_v2(raw("No name here at all."), NAME, False)["status"],
            "invalid:name_missing",
        )
        self.assertEqual(
            classify_sample_v2(raw(f"{NAME} and {NAME} twice"), NAME, False)["status"],
            "invalid:name_dup",
        )
        self.assertEqual(
            classify_sample_v2(raw("whatever"), NAME, True)["status"], "invalid:truncated"
        )


class TestHelpers(unittest.TestCase):
    def test_refusal_detector(self):
        self.assertTrue(is_refusal("I cannot help with that"))
        self.assertTrue(is_refusal("Sure thing! As an AI I must stop here", "Nobody"))
        self.assertFalse(is_refusal("The invoice was paid on time."))
        self.assertFalse(
            is_refusal(
                "Ifigeneia paid the invoice late again and honestly I can't wait to tell the whole team about it.",
                "Ifigeneia",
            )
        )

    def test_greeting_only(self):
        self.assertTrue(is_greeting_only("Hello, how are you today?", "Nobody"))
        self.assertFalse(is_greeting_only("The report is attached.", "Nobody"))


class TestRelabel(unittest.TestCase):
    def test_relabel_pilot_fixture(self):
        with open(PILOT, encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        self.assertGreaterEqual(len(samples), 5)
        report = relabel_samples(samples)
        self.assertEqual(report["n"], len(samples))
        self.assertLessEqual(report["v2_valid"], report["v1_valid"])
        # every pilot valid sample that fails v2 must carry a v2 reason
        for flip in report["flips_valid_to_invalid"]:
            self.assertTrue(flip["status_v2"].startswith("invalid:"))

    def test_relabel_hard_panel_low_p_first(self):
        samples = [
            {
                "raw_text": raw_text_for("Ifigeneia wrote an email to ask about the new timeline."),
                "name": NAME,
                "truncated": False,
                "status": "valid",
                "p_target_mean": 0.05,
                "group_id": "g1|t1.0",
                "sample_idx": 0,
            },
            {
                "raw_text": raw_text_for("Hi."),
                "name": NAME,
                "truncated": False,
                "status": "valid",
                "p_target_mean": 0.99,
                "group_id": "g1|t1.0",
                "sample_idx": 1,
            },
        ]
        report = relabel_samples(samples)
        self.assertEqual(report["hard_samples"][0]["p_target_mean"], 0.05)
        self.assertEqual(report["hard_samples"][0]["status_v2"], "valid")


if __name__ == "__main__":
    unittest.main()
