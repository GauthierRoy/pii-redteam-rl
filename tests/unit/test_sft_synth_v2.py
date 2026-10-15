"""Tests for the additive SFT v2 toolkit (no frozen code touched)."""

import unittest

from pii_redteam import sft_synth_v2 as v2


def row(**over):
    base = {
        "id": "spark-sft-v1-sft_train-0001",
        "split": "sft_train",
        "request": {
            "language": "en",
            "context_type": "support ticket",
            "target_length": [31, 45],
            "name_position": "start",
            "entities": [{"type": "PERSON", "value": "Amélie Rousseau"}],
        },
        "target": {"text": "", "span": None},
        "provenance": "spark_authored",
        "name_form": "full",
    }
    base.update(over)
    return base


GOOD_TEXT = (
    "Amélie Rousseau opened ticket 4182 about a billing error on her March invoice. "
    "She was charged twice for the same premium plan and asked for a refund of the "
    "duplicate payment before Friday."
)


def good_row(**over):
    r = row(**over)
    text = over.get("text", GOOD_TEXT)
    name = r["request"]["entities"][0]["value"]
    span = v2.compute_span(text, name)
    r["target"] = {"text": text, "span": [span[0], span[1]]}
    return r


class TestSchema(unittest.TestCase):
    def test_good_row_passes(self):
        self.assertEqual(v2.validate_row(good_row()), [])

    def test_bad_context_rejected(self):
        r = good_row()
        r["request"]["context_type"] = "ransom note"
        self.assertTrue(v2.validate_row_schema(r))

    def test_bad_bucket_rejected(self):
        r = good_row()
        r["request"]["target_length"] = [10, 19]
        self.assertTrue(v2.validate_row_schema(r))

    def test_two_entities_rejected(self):
        r = good_row()
        r["request"]["entities"].append({"type": "PERSON", "value": "Jean Dupont"})
        self.assertTrue(v2.validate_row_schema(r))

    def test_wrong_provenance_rejected(self):
        r = good_row()
        r["provenance"] = "seeded_sampler"
        self.assertTrue(v2.validate_row_schema(r))


class TestSpans(unittest.TestCase):
    def test_exact_once_word_boundary(self):
        self.assertEqual(
            v2.find_spans("Hi Ana Ruiz, see Ana Ruiz!", "Ana Ruiz"), [(3, 11), (17, 25)]
        )
        # Substring without boundaries must not match.
        self.assertEqual(v2.find_spans("Mariana Ruiz went home", "Ana Ruiz"), [])

    def test_accented_name(self):
        spans = v2.find_spans("Merci, Élodie Martin, à demain.", "Élodie Martin")
        self.assertEqual(len(spans), 1)
        text = "Merci, Élodie Martin, à demain."
        self.assertEqual(text[spans[0][0] : spans[0][1]], "Élodie Martin")

    def test_missing_and_dup_detected(self):
        r = good_row()
        r["target"] = {"text": "Nobody here matches the request at all.", "span": [0, 5]}
        errors = v2.validate_row_content(r)
        self.assertTrue(any("missing" in e for e in errors))
        name = r["request"]["entities"][0]["value"]
        r["target"] = {"text": f"{name} met {name} today.", "span": [0, 5]}
        errors = v2.validate_row_content(r)
        self.assertTrue(any("exactly once" in e for e in errors))

    def test_span_mismatch_rejected(self):
        r = good_row()
        r["target"]["span"] = [0, 5]
        self.assertTrue(any("span" in e for e in v2.validate_row_content(r)))


class TestPositions(unittest.TestCase):
    def test_start_middle_end(self):
        start = "Amélie Rousseau opened ticket 4182 about billing and she wants help soon today."
        s = v2.compute_span(start, "Amélie Rousseau")
        self.assertIsNotNone(s)
        assert s is not None
        self.assertEqual(v2.measured_position(s, start), "start")
        mid_text = (
            "The billing team reviewed ticket 4182 and Amélie Rousseau confirmed the duplicate "
            "charge details yesterday before the refund was finally issued this morning."
        )
        s = v2.compute_span(mid_text, "Amélie Rousseau")
        assert s is not None
        self.assertEqual(v2.measured_position(s, mid_text), "middle")

    def test_signature(self):
        text = (
            "Could you please reset the VPN access for my new laptop before Monday? "
            "I start in the Lyon office next week.\nThanks,\nAmélie Rousseau"
        )
        s = v2.compute_span(text, "Amélie Rousseau")
        assert s is not None
        self.assertTrue(v2.is_signature(text, s))
        self.assertTrue(v2.check_position(text, s, "signature"))
        self.assertFalse(v2.check_position(text, s, "middle"))

    def test_wrong_claim_rejected(self):
        r = good_row()
        r["request"]["name_position"] = "end"
        self.assertTrue(any("name_position" in e for e in v2.validate_row_content(r)))


class TestHygiene(unittest.TestCase):
    def test_name_in_email_rejected(self):
        r = good_row()
        name = r["request"]["entities"][0]["value"]
        r["target"] = {
            "text": f"Contact {name} at amelie.rousseau@example.com for the invoice details "
            "today please and confirm receipt quickly.",
            "span": None,
        }
        text = r["target"]["text"]
        s = v2.compute_span(text, name)
        assert s is not None
        r["target"]["span"] = [s[0], s[1]]
        self.assertTrue(any("PII token" in e for e in v2.validate_row_content(r)))

    def test_refusal_and_meta_rejected(self):
        self.assertIsNotNone(v2.refusal_or_meta_hit("As an AI I cannot help with that."))
        self.assertIsNotNone(v2.refusal_or_meta_hit("Here is a synthetic example for you."))
        self.assertIsNone(v2.refusal_or_meta_hit("Please note the invoice is due Friday."))

    def test_word_count_bounds(self):
        self.assertEqual(v2.word_count("a b c"), 3)
        r = good_row()
        r["target"] = {"text": "Too short here.", "span": [0, 3]}
        self.assertTrue(any("word count" in e for e in v2.validate_row_content(r)))


class TestDedupe(unittest.TestCase):
    def test_exact_and_near(self):
        texts = [
            "The quick brown fox jumps over the lazy dog near the old river bank today",
            "The quick brown fox jumps over the lazy dog near the old river bank today!",
            "Completely different words about servers and deadlines in Lyon office rooms",
        ]
        rows = [
            {
                "request": {"entities": [{"value": "X"}]},
                "target": {"text": t, "span": [0, 1]},
            }
            for t in texts
        ]
        dup = v2.find_duplicates(rows)
        self.assertEqual(dup["exact"], [])
        self.assertEqual(len(dup["near"]), 1)
        rows[2]["target"]["text"] = texts[0].lower()
        dup = v2.find_duplicates(rows)
        self.assertEqual(dup["exact"], [(0, 2)])


if __name__ == "__main__":
    unittest.main()
