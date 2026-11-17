"""M06 baselines: template filling, plain and generate-and-filter selection, row format."""

import random
import unittest

from pii_redteam.baselines import (
    TEMPLATES,
    aug_row,
    fill_template,
    hard_selection,
    plain_selection,
    template_rows,
)


def accepted(
    request_id: str, name: str, p: float, sample_idx: int = 0, context: str = "chat_message"
):
    text = f"hey {name}, lunch?"
    return {
        "request_id": request_id,
        "sample_idx": sample_idx,
        "context_type": context,
        "name": name,
        "text": text,
        "name_span": [4, 4 + len(name)],
        "p_target_mean": p,
    }


class TemplateTest(unittest.TestCase):
    def test_every_template_gives_one_name_with_its_span(self) -> None:
        rng = random.Random(0)
        for templates in TEMPLATES.values():
            for template in templates:
                text, (start, end) = fill_template(template, "Amara", rng)
                self.assertEqual(text[start:end], "Amara")
                self.assertEqual(text.count("Amara"), 1)
                self.assertNotIn("{", text)

    def test_rows_keep_name_and_context_of_the_rl_rows(self) -> None:
        rl = [
            {
                "request": {"context_type": "form_record_line", "entities": [{"value": "Lucas"}]},
                "source_request_id": "aug-1",
            }
        ]
        (row,) = template_rows(rl, seed=1)
        self.assertEqual(row["request"]["context_type"], "form_record_line")
        start, end = row["target"]["span"]
        self.assertEqual(row["target"]["text"][start:end], "Lucas")
        self.assertEqual(row, template_rows(rl, seed=1)[0])


class SelectionTest(unittest.TestCase):
    def test_plain_keeps_first_samples_in_request_order(self) -> None:
        rows = [accepted("r2", "B", 0.9), accepted("r1", "A", 0.2, 1), accepted("r1", "A", 0.8)]
        self.assertEqual([r["request_id"] for r in plain_selection(rows, 5)], ["r1", "r2"])
        self.assertEqual(plain_selection(rows, 5)[0]["sample_idx"], 0)

    def test_hard_takes_lowest_probability_once_per_name(self) -> None:
        rows = [
            accepted("r1", "A", 0.6),
            accepted("r1", "A", 0.1, 1),
            accepted("r2", "B", 0.3),
            accepted("r3", "C", 0.9),
        ]
        chosen = hard_selection(rows, 2)
        self.assertEqual(
            [(r["name"], r["p_target_mean"]) for r in chosen], [("A", 0.1), ("B", 0.3)]
        )

    def test_hard_caps_each_context(self) -> None:
        rows = [accepted(f"r{i}", f"N{i}", i / 10, context="chat_message") for i in range(5)]
        rows.append(accepted("x", "X", 0.95, context="meeting_notes"))
        chosen = hard_selection(rows, 10, context_cap=2)
        self.assertEqual([r["name"] for r in chosen], ["N0", "N1", "X"])

    def test_aug_row_checks_the_span(self) -> None:
        row = aug_row("plain-aug-v1", 0, accepted("r1", "Lucas", 0.4), "base_attacker:run")
        self.assertEqual(row["target"]["span"], [4, 9])
        with self.assertRaises(ValueError):
            aug_row("x", 0, {**accepted("r1", "Lucas", 0.4), "name_span": [0, 3]}, "p")


if __name__ == "__main__":
    unittest.main()
