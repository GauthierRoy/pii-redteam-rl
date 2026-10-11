"""M07a probe-metric tests (pure functions, no GPU)."""

import unittest

from pii_redteam.probe_metrics import (
    classify_sample,
    diversity_metrics,
    logit,
    name_spans,
    outcome_for_sample,
    parse_json_strict,
    parse_json_tolerant,
    reward_metrics,
    reward_variants,
    threshold_report,
)


def sample(group_id, idx, temp, text, target, status, p=None, chars=None, pos=0.5, extra=0):
    return {
        "group_id": group_id,
        "sample_idx": idx,
        "temperature": temp,
        "text": text,
        "target_span": target,
        "status": status,
        "p_target_mean": p,
        "p_target_min": p,
        "outcome": "complete_miss" if status == "valid" else "invalid",
        "n_chars": chars if chars is not None else len(text or ""),
        "name_position": pos,
        "extra_person_flags": extra,
        "raw_text": text,
        "request_id": group_id.split("|")[0],
        "n_generated_tokens": 0,
        "truncated": False,
        "json_tolerant_ok": True,
        "r_prob": reward_variants(p)["r_prob"] if p is not None else -1.0,
        "r_logit": reward_variants(p)["r_logit"] if p is not None else -1.0,
    }


class TestValidation(unittest.TestCase):
    def test_parse_strict_and_tolerant(self):
        good = '{"text": "Ana is here."}'
        self.assertEqual(parse_json_strict(good), "Ana is here.")
        self.assertIsNone(parse_json_strict("```json\n" + good + "\n```"))
        self.assertEqual(parse_json_tolerant("```json\n" + good + "\n```"), "Ana is here.")
        self.assertIsNone(parse_json_strict("not json"))

    def test_name_spans_word_boundary(self):
        text = "Ana met Anastasia and Ana again"
        spans = name_spans(text, "Ana")
        self.assertEqual(len(spans), 2)  # Anastasia does NOT match
        self.assertEqual(text[spans[0][0] : spans[0][1]], "Ana")

    def test_classify_all_statuses(self):
        self.assertEqual(classify_sample('{"text": "hi Ana"}', "Ana", False)["status"], "valid")
        self.assertEqual(classify_sample("nope", "Ana", False)["status"], "invalid:json")
        self.assertEqual(
            classify_sample('{"text": "no one"}', "Ana", False)["status"], "invalid:name_missing"
        )
        self.assertEqual(
            classify_sample('{"text": "Ana and Ana"}', "Ana", False)["status"], "invalid:name_dup"
        )
        self.assertEqual(
            classify_sample('{"text": "hi Ana partial', "Ana", True)["status"], "invalid:truncated"
        )


class TestOutcomes(unittest.TestCase):
    def test_outcome_classification(self):
        self.assertEqual(outcome_for_sample((5, 8), [(5, 8)]), "exact_hit")
        self.assertEqual(outcome_for_sample((5, 8), [(4, 9)]), "partial_overlap")  # contains
        self.assertEqual(outcome_for_sample((5, 8), [(5, 9)]), "partial_overlap")  # contained
        self.assertEqual(outcome_for_sample((5, 8), [(7, 10)]), "boundary_error")  # crosses end
        self.assertEqual(outcome_for_sample((5, 8), [(20, 25)]), "complete_miss")

    def test_reward_variants_and_logit(self):
        self.assertAlmostEqual(reward_variants(0.9)["r_prob"], 0.1)
        self.assertGreater(logit(0.99), logit(0.9999))  # lower p -> higher reward; 0.99 != 0.9999
        self.assertGreater(abs(logit(0.99) - logit(0.9999)), 4.0)
        self.assertEqual(logit(0.5, clip=10), 0.0)


class TestDiversity(unittest.TestCase):
    def test_diversity_metrics(self):
        samples = [
            sample("g1|t1.0", 0, 1.0, "Hello Ana today", (6, 9), "valid", p=0.5),
            sample("g1|t1.0", 1, 1.0, "Hello Ana tomorrow", (6, 9), "valid", p=0.3),
            sample("g2|t1.0", 0, 1.0, "Bye Ana now", (4, 7), "valid", p=0.9),
            sample("g2|t1.0", 1, 1.0, "Bye Ana now", (4, 7), "valid", p=0.9),  # duplicate
            sample("g3|t1.0", 0, 1.0, None, None, "invalid:json"),
        ]
        m = diversity_metrics(samples)
        self.assertEqual(m["total"], 5)
        self.assertEqual(m["valid"], 4)
        self.assertAlmostEqual(m["contract_rate"], 0.8)
        self.assertEqual(m["failure_breakdown"], {"invalid:json": 1})
        self.assertGreater(m["exact_dup_rate"], 0)  # the g2 duplicate
        self.assertEqual(m["unique_per_group_median"], 2)  # g1: 2 unique, g2: 1 -> sorted [1,2]
        self.assertEqual(m["top10_opening_share"], 1.0)  # only 4 samples: top-10 covers all


class TestRewardsAndThresholds(unittest.TestCase):
    def test_reward_metrics_signal(self):
        samples = []
        for g in range(10):
            ps = [0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 0.5, 0.3][:4]
            for i, p in enumerate(ps):
                samples.append(sample(f"g{g}|t1.0", i, 1.0, f"text {g} {i}", (0, 1), "valid", p=p))
        m = reward_metrics(samples)
        self.assertEqual(m["valid_scored"], 40)
        self.assertGreater(m["live_group_share_r_prob"], 0)  # groups vary -> live
        self.assertLessEqual(m["dead_group_share"], 0.1)

    def test_threshold_report(self):
        metrics = {
            "contract_rate": 0.95,
            "exact_dup_rate": 0.02,
            "unique_per_group_median": 7,
            "live_group_share_r_prob": 0.4,
            "live_group_share_r_logit": 0.5,
            "outcome_rates": {"complete_miss": 0.1},
            "target_spans_lost_to_d0_truncation": 0,
        }
        thresholds = {
            "format_valid_min": 0.9,
            "exact_dup_max": 0.05,
            "unique_per_group_median_min": 6,
            "live_group_share_min": 0.3,
            "miss_rate_range": [0.01, 0.3],
            "target_spans_lost_max": 0,
        }
        report = threshold_report(metrics, thresholds)
        self.assertTrue(all(c["ok"] for c in report.values()), report)
        report2 = threshold_report(metrics | {"contract_rate": 0.5}, thresholds)
        self.assertFalse(report2["format_valid"]["ok"])


if __name__ == "__main__":
    unittest.main()
