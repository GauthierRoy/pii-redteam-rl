"""Person-check + position-compliance tests (M07b.2 gates, pure functions)."""

import unittest

from pii_redteam.person_check import is_same_name, parse_yes_no, spacy_extra_persons
from pii_redteam.prompt_v2 import hint_compliance, position_matches_hint


class _Ent:
    def __init__(self, text: str, label: str):
        self.text = text
        self.label_ = label


class _Doc:
    def __init__(self, ents: list[_Ent]):
        self.ents = ents


class TestSpacyGate(unittest.TestCase):
    def test_same_name(self):
        self.assertTrue(is_same_name("Mermolja", "Mermolja"))
        self.assertTrue(is_same_name("  mermolja ", "Mermolja"))
        self.assertTrue(is_same_name("Mr. Gulzar", "Gulzar"))
        self.assertTrue(is_same_name("Dr. Ana Ruiz", "Ana Ruiz"))
        self.assertTrue(is_same_name("F. Bettschen", "Fithi Bettschen"))
        self.assertFalse(is_same_name("Sarah", "Mermolja"))
        self.assertFalse(is_same_name("Mermolja", ""))

    def test_extra_persons(self):
        doc = _Doc(
            [
                _Ent("Mermolja", "PERSON"),
                _Ent("Sarah", "PERSON"),
                _Ent("Berlin", "GPE"),
                _Ent("Gulzar", "PERSON"),
            ]
        )
        self.assertEqual(spacy_extra_persons(doc, "Mermolja"), ["Gulzar", "Sarah"])

    def test_no_extras(self):
        doc = _Doc([_Ent("Ifigeneia", "PERSON"), _Ent("Tuesday", "DATE")])
        self.assertEqual(spacy_extra_persons(doc, "Ifigeneia"), [])


class TestJudgeParse(unittest.TestCase):
    def test_parse_yes_no(self):
        self.assertIs(parse_yes_no("Yes"), True)
        self.assertIs(parse_yes_no("no"), False)
        self.assertIs(parse_yes_no("  No."), False)
        self.assertIs(parse_yes_no("maybe"), None)
        self.assertIs(parse_yes_no(""), None)


class TestPositionCompliance(unittest.TestCase):
    def test_bins(self):
        text = "abcdefghij"
        self.assertTrue(position_matches_hint("start", (0, 2), text))
        self.assertTrue(position_matches_hint("middle", (4, 6), text))
        self.assertTrue(position_matches_hint("end", (8, 10), text))
        self.assertTrue(position_matches_hint("signature", (9, 10), text))
        self.assertFalse(position_matches_hint("signature", (0, 2), text))
        self.assertFalse(position_matches_hint("unknown-hint", (0, 2), text))

    def test_hint_compliance_aggregation(self):
        samples = [
            {
                "status": "valid",
                "request_id": "r1",
                "variant": "P1",
                "temperature": 0.7,
                "text": "abcdefghij",
                "target_span": (0, 2),
            },
            {
                "status": "valid",
                "request_id": "r1",
                "variant": "P1",
                "temperature": 0.7,
                "text": "abcdefghij",
                "target_span": (8, 10),
            },
            {
                "status": "invalid:too_short",
                "request_id": "r1",
                "variant": "P1",
                "temperature": 0.7,
                "text": "ab",
                "target_span": (0, 2),
            },
        ]
        out = hint_compliance(samples, {"r1": "start"})
        self.assertEqual(out["start"]["P1|t0.7"], {"compliant": 1, "n": 2, "rate": 0.5})

    def test_hint_compliance_skips_unhinted(self):
        samples = [
            {
                "status": "valid",
                "request_id": "r2",
                "variant": "P0",
                "temperature": 1.0,
                "text": "abcdefghij",
                "target_span": (0, 2),
            }
        ]
        self.assertEqual(hint_compliance(samples, {"r2": None}), {})


if __name__ == "__main__":
    unittest.main()
