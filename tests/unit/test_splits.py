"""Split-role and name-pool helper tests (M02.3–M02.4)."""

import unittest
from collections import Counter

from pii_redteam.data.splits import (
    carve_final_eval,
    carve_heldout_names,
    dedup_by_text,
    person_name_counts,
    template_skeleton,
    text_sha256,
)


def canonical(example_id: str, text: str, spans: list[tuple[int, int]]) -> dict:
    return {
        "example_id": example_id,
        "text": text,
        "spans": [{"label": "PERSON", "start": s, "end": e} for s, e in spans],
    }


class TestSplits(unittest.TestCase):
    def test_template_skeleton_masks_all_labels(self):
        row = {
            "source_text": "Contact Ana Ruiz at ana@example.com.",
            "privacy_mask": [
                {"value": "Ana", "start": 8, "end": 11, "label": "GIVENNAME1"},
                {"value": "Ruiz", "start": 12, "end": 16, "label": "LASTNAME1"},
                {"value": "ana@example.com", "start": 20, "end": 35, "label": "EMAIL"},
            ],
        }
        self.assertEqual(
            template_skeleton(row["source_text"], row["privacy_mask"]),
            "Contact <GIVENNAME1> <LASTNAME1> at <EMAIL>.",
        )

    def test_text_sha256_stable_and_distinct(self):
        self.assertEqual(text_sha256("Ana"), text_sha256("Ana"))
        self.assertNotEqual(text_sha256("Ana"), text_sha256("ana"))

    def test_dedup_keeps_first_occurrence(self):
        rows = [
            canonical("a", "Same text", []),
            canonical("b", "Same text", []),
            canonical("c", "Other text", []),
        ]
        kept, dupes = dedup_by_text(rows)
        self.assertEqual([r["example_id"] for r in kept], ["a", "c"])
        self.assertEqual(dupes, ["b"])

    def test_carve_final_eval_deterministic_and_bounded(self):
        ids = [f"r{i}" for i in range(50)]
        first = carve_final_eval(ids, seed=7, n=5)
        self.assertEqual(first, carve_final_eval(sorted(ids, reverse=True), seed=7, n=5))
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)
        self.assertTrue(set(first) <= set(ids))

    def test_person_name_counts(self):
        rows = [
            canonical("a", "Ana Ruiz met Jean.", [(0, 8), (13, 17)]),
            canonical("b", "Ask Ana Ruiz.", [(4, 12)]),
        ]
        self.assertEqual(person_name_counts(rows), Counter({"Ana Ruiz": 2, "Jean": 1}))

    def test_heldout_names_disjoint_from_train_side(self):
        final_eval = Counter({"Ana Ruiz": 2, "Élodie Martin": 1, "Jean Dupont": 1})
        train_side = Counter({"Ana Ruiz": 9, "Zed Quill": 3})
        heldout = carve_heldout_names(final_eval, train_side, seed=7, heldout_n=2)
        self.assertEqual(heldout, ["Jean Dupont", "Élodie Martin"])
        self.assertFalse(set(heldout) & set(train_side))
        self.assertEqual(carve_heldout_names(final_eval, train_side, seed=7, heldout_n=10), heldout)


if __name__ == "__main__":
    unittest.main()
