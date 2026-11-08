import json
import os
import tempfile
import unittest

from pii_redteam.d1_metrics import (
    is_better_candidate,
    load_synthetic_rows,
    no_regression_gate,
    paired_changes,
    short_eval_gate,
    span_metrics,
    split_by_name,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _row(row_id, spans, name="Ann"):
    return {
        "example_id": row_id,
        "text": "x" * 50,
        "spans": [{"label": "PERSON", "start": s, "end": e} for s, e in spans],
        "name": name,
    }


class LoadSyntheticRowsTest(unittest.TestCase):
    def _write(self, record):
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        handle.write(json.dumps(record) + "\n")
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_converts_a_valid_row_with_accents(self):
        path = self._write(
            {
                "id": "a",
                "request": {"entities": [{"type": "PERSON", "value": "Élodie"}]},
                "target": {"text": "Ask Élodie now", "span": [4, 10]},
            }
        )
        row = load_synthetic_rows(path)[0]
        self.assertEqual(row["spans"], [{"label": "PERSON", "start": 4, "end": 10}])
        self.assertEqual(row["text"][4:10], "Élodie")

    def test_rejects_a_span_that_is_not_the_entity(self):
        path = self._write(
            {
                "id": "a",
                "request": {"entities": [{"type": "PERSON", "value": "Ann"}]},
                "target": {"text": "Ask Ann now", "span": [0, 3]},
            }
        )
        with self.assertRaises(ValueError):
            load_synthetic_rows(path)

    def test_every_committed_d1_data_file_loads(self):
        synthetic = os.path.join(REPO, "data", "synthetic")
        paths = {
            "d1_aug_v1.jsonl": 2000,
            "short_eval_v2.jsonl": 150,
        }
        for name, expected in paths.items():
            self.assertEqual(len(load_synthetic_rows(os.path.join(synthetic, name))), expected)


class SplitByNameTest(unittest.TestCase):
    def setUp(self):
        self.rows = [_row(f"r{i}", [(0, 3)], name=f"Name{i % 40}") for i in range(200)]

    def test_no_name_in_both_parts_including_case_variants(self):
        rows = [*self.rows, _row("upper", [(0, 3)], name="NAME1")]
        train, dev = split_by_name(rows, 0.1, seed=1)
        self.assertFalse(
            {r["name"].casefold() for r in train} & {r["name"].casefold() for r in dev}
        )
        self.assertEqual(len(train) + len(dev), len(rows))

    def test_is_seeded_and_near_the_requested_size(self):
        first = split_by_name(self.rows, 0.1, seed=1)
        self.assertEqual(first, split_by_name(self.rows, 0.1, seed=1))
        self.assertEqual(len(first[1]), 20)


class SpanMetricsTest(unittest.TestCase):
    def test_known_answers(self):
        rows = [
            _row("hit", [(0, 3)]),
            _row("miss", [(0, 3)]),
            _row("partial", [(0, 10)]),
            _row("negative", []),
        ]
        preds = [[(0, 3)], [(20, 25)], [(0, 4)], [(1, 2)]]
        m = span_metrics(rows, preds)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 3, 2))
        self.assertEqual(m["span_precision"], 0.25)
        self.assertEqual(m["span_recall"], 0.3333)
        self.assertEqual(m["outcomes"]["exact_hit"], 1)
        self.assertEqual(m["outcomes"]["complete_miss"], 1)
        self.assertEqual(m["outcomes"]["partial_overlap"], 1)
        self.assertEqual(m["complete_miss_rate"], 0.3333)
        self.assertEqual((m["neg_rows"], m["neg_false_positives"]), (1, 1))
        self.assertEqual(m["rows_with_extra_pred"], 2)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            span_metrics([_row("a", [(0, 3)])], [])


class SelectionAndGatesTest(unittest.TestCase):
    def test_epoch_below_the_dev_floor_is_never_selected(self):
        self.assertFalse(is_better_candidate(0.94, 0.99, None, 0.9633, 0.01))

    def test_first_eligible_epoch_is_selected_and_ties_keep_it(self):
        self.assertTrue(is_better_candidate(0.96, 0.80, None, 0.9633, 0.01))
        self.assertFalse(is_better_candidate(0.96, 0.80, 0.80, 0.9633, 0.01))
        self.assertTrue(is_better_candidate(0.96, 0.81, 0.80, 0.9633, 0.01))

    def test_paired_changes_requires_the_same_rows(self):
        d0 = {"per_row_outcome": {"a": "complete_miss", "b": "exact_hit", "c": "exact_hit"}}
        d1 = {"per_row_outcome": {"a": "exact_hit", "b": "complete_miss", "c": "exact_hit"}}
        self.assertEqual(
            paired_changes(d0, d1), {"gained_exact": 1, "lost_exact": 1, "lost_exact_ids": ["b"]}
        )
        with self.assertRaises(ValueError):
            paired_changes(d0, {"per_row_outcome": {"a": "exact_hit"}})

    def test_short_eval_gate_uses_recall_gain(self):
        self.assertTrue(short_eval_gate({"span_recall": 0.5}, {"span_recall": 0.6}, 0.1)["ok"])
        self.assertFalse(short_eval_gate({"span_recall": 0.5}, {"span_recall": 0.59}, 0.1)["ok"])

    def test_no_regression_gate_fails_on_either_metric(self):
        d0 = {"span_precision": 0.9, "span_recall": 0.9}
        keep = {"span_precision": 0.895, "span_recall": 0.95}
        low_precision = {"span_precision": 0.88, "span_recall": 0.95}
        low_recall = {"span_precision": 0.95, "span_recall": 0.88}
        self.assertTrue(no_regression_gate(d0, keep, 0.01)["ok"])
        self.assertFalse(no_regression_gate(d0, low_precision, 0.01)["ok"])
        self.assertFalse(no_regression_gate(d0, low_recall, 0.01)["ok"])


if __name__ == "__main__":
    unittest.main()
