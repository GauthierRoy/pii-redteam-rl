"""RL augmentation texts: which samples are kept, the rows written, and fresh names."""

import json
import unittest

from pii_redteam.training.rl_bank import build_rl_requests
from pii_redteam.training.rl_generate import (
    ACCEPTED,
    accepted_rows,
    generation_counts,
    generation_requests,
    with_acceptance,
)

LETTER = "Dear Lucas, we have reviewed your support request and will call you back tomorrow."
NOTE = "Yesterday Amara wrote a long note about the delayed shipment and the missing invoice."


def sample(text: str, name: str, **fields) -> dict:
    return {
        "request_id": f"aug-{name}",
        "context_type": "support_ticket",
        "name": name,
        "raw_text": json.dumps({"text": text}),
        "status": "valid",
        "p_target_mean": 0.1,
        "other_person": False,
        "natural": True,
        "fits_format": True,
        **fields,
    }


def names_of(requests: list[dict]) -> set[str]:
    return {e["value"] for r in requests for e in r["task"]["entities"]}


class TestAcceptance(unittest.TestCase):
    def test_valid_text_without_another_person_is_accepted(self):
        (decided,) = with_acceptance([sample(LETTER, "Lucas")])
        self.assertEqual((decided["decision"], decided["text"]), (ACCEPTED, LETTER))

    def test_each_rejection_reason_is_recorded(self):
        decided = with_acceptance(
            [
                sample(LETTER, "Lucas", status="invalid:name_not_person"),
                sample(LETTER, "Lucas", other_person=True),
                sample(LETTER, "Lucas", other_person=None),
                sample(LETTER, "Lucas"),
                sample(LETTER, "Lucas"),
            ]
        )
        self.assertEqual(
            [d["decision"] for d in decided],
            [
                "rejected:invalid:name_not_person",
                "rejected:other_person",
                "rejected:audit_missing",
                ACCEPTED,
                "rejected:duplicate",
            ],
        )

    def test_accepted_rows_carry_the_name_span(self):
        decided = with_acceptance([sample(NOTE, "Amara"), sample(LETTER, "Lucas", status="x")])
        (row,) = accepted_rows(decided)
        start, end = row["name_span"]
        self.assertEqual(row["text"][start:end], "Amara")
        self.assertEqual((row["name"], row["p_target_mean"]), ("Amara", 0.1))

    def test_counts_cover_decisions_misses_and_openings(self):
        decided = with_acceptance(
            [
                sample(LETTER, "Lucas"),
                sample(NOTE, "Amara", p_target_mean=0.9, context_type="chat_message"),
                sample(NOTE, "Amara", other_person=True),
            ]
        )
        counts = generation_counts(decided)
        self.assertEqual(counts["generated"], 3)
        self.assertEqual(counts["decisions"], {ACCEPTED: 2, "rejected:other_person": 1})
        self.assertEqual((counts["accepted"], counts["accepted_names"]), (2, 2))
        self.assertEqual(counts["accepted_detector_misses"], 1)
        self.assertEqual(counts["accepted_misses_by_context"], {"support_ticket": 1})
        self.assertEqual(counts["accepted_top_openings"], {"dear": 1, "yesterday": 1})

    def test_counts_with_nothing_accepted(self):
        counts = generation_counts(with_acceptance([sample(LETTER, "Lucas", status="x")]))
        self.assertEqual((counts["accepted"], counts["accepted_mean_words"]), (0, None))


class TestGenerationRequests(unittest.TestCase):
    def test_names_avoid_the_rl_bank_and_the_excluded_names(self):
        pool = frozenset(f"Name{i:03d}" for i in range(60))
        excluded = frozenset({"Name000", "Name001"})
        bank = {"seed": 2, "n": 20, "prefix": "aug", "exclude_rl_bank": {"seed": 1, "n": 30}}
        requests = generation_requests(pool, excluded, bank)
        rl_names = names_of(build_rl_requests(pool, n=30, seed=1, exclude=excluded))
        self.assertEqual(len(names_of(requests)), 20)
        self.assertFalse(names_of(requests) & (rl_names | excluded))
        self.assertTrue(all(r["request_id"].startswith("aug") for r in requests))


if __name__ == "__main__":
    unittest.main()
