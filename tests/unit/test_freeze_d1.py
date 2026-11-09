import json
import os
import shutil
import tempfile
import unittest

import yaml

from pii_redteam.d1_metrics import load_negative_rows, negative_fp_metrics, span_metrics
from pii_redteam.freeze_gates import (
    MANIFEST_FILE,
    build_freeze_manifest,
    check_counts,
    dir_file_hashes,
    file_sha256,
    final_eval_f1_gate,
    freeze_candidate,
    freeze_decision,
    frozen_dir_for,
    make_check,
    precision_guard,
    probe_comparison,
    probe_gate,
    repo_commit_recorded,
    resolve_optional_input,
    short_negatives_check,
    short_negatives_gate,
    short_negatives_report,
)
from pii_redteam.probe_metrics import per_temperature_metrics

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _metrics(p, r, f1, neg_rows=100, neg_with_pred=10, neg_fp=12):
    return {
        "span_precision": p,
        "span_recall": r,
        "span_f1": f1,
        "neg_rows": neg_rows,
        "neg_rows_with_pred": neg_with_pred,
        "neg_false_positives": neg_fp,
    }


def _neg_record(
    row_id,
    text="Lunch moved to noon, bring the slides.",
    entities=None,
    span=None,
    distractor="calendar",
):
    return {
        "id": row_id,
        "split": "short_eval_neg",
        "distractor": distractor,
        "request": {"language": "en", "entities": [] if entities is None else entities},
        "target": {"text": text, "span": span},
    }


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def write_jsonl(self, name, records):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return path


class ChecksAndDecisionTest(unittest.TestCase):
    def test_check_status(self):
        self.assertEqual(make_check(True)["status"], "pass")
        self.assertEqual(make_check(False)["status"], "fail")
        skipped = make_check(None, "skipped: not provided", required=False)
        self.assertEqual((skipped["status"], skipped["ok"]), ("skipped", None))

    def test_all_required_pass_allows_freeze(self):
        checks = {"a": make_check(True), "b": make_check(True)}
        self.assertTrue(freeze_decision(checks)["freeze_allowed"])

    def test_a_failing_required_check_blocks(self):
        checks = {"a": make_check(True), "b": make_check(False)}
        decision = freeze_decision(checks)
        self.assertFalse(decision["freeze_allowed"])
        self.assertEqual(decision["failing_required"], ["b"])

    def test_a_skipped_required_check_blocks(self):
        decision = freeze_decision({"probe": make_check(None, "skipped")})
        self.assertFalse(decision["freeze_allowed"])
        self.assertEqual(decision["failing_required"], ["probe"])

    def test_a_skipped_optional_check_is_listed_not_passed(self):
        checks = {"a": make_check(True), "neg": make_check(None, "skipped", required=False)}
        decision = freeze_decision(checks)
        self.assertTrue(decision["freeze_allowed"])
        self.assertEqual(decision["skipped_optional"], ["neg"])
        self.assertEqual(check_counts(checks), {"pass": 1, "fail": 0, "skipped": 1})

    def test_repo_commit_recorded(self):
        self.assertTrue(repo_commit_recorded("b8550eb"))
        for missing in (None, "", "unrecorded", "recorded-by-notebook"):
            self.assertFalse(repo_commit_recorded(missing))


class FinalEvalGatesTest(unittest.TestCase):
    def test_f1_gate_boundary(self):
        d0 = _metrics(0.96, 0.96, 0.9633)
        self.assertTrue(final_eval_f1_gate(d0, _metrics(0.95, 0.96, 0.9533), 0.01)["ok"])
        self.assertFalse(final_eval_f1_gate(d0, _metrics(0.95, 0.95, 0.9532), 0.01)["ok"])
        self.assertTrue(final_eval_f1_gate(d0, _metrics(0.97, 0.97, 0.97), 0.01)["ok"])
        self.assertEqual(final_eval_f1_gate(d0, d0, 0.01)["floor"], 0.9533)

    def test_precision_guard_relative_threshold(self):
        d0 = _metrics(0.96, 0.96, 0.96, neg_rows=200, neg_with_pred=20)
        at_limit = _metrics(0.95, 0.96, 0.955, neg_rows=200, neg_with_pred=25)
        over = _metrics(0.95, 0.96, 0.955, neg_rows=200, neg_with_pred=26)
        self.assertTrue(precision_guard(d0, at_limit, 0.25)["ok"])
        result = precision_guard(d0, over, 0.25)
        self.assertFalse(result["ok"])
        self.assertEqual(result["relative_increase"], 0.3)
        self.assertEqual(result["d0_neg_fp_row_rate"], 0.1)
        self.assertEqual(result["allowed_d1_rate"], 0.125)

    def test_precision_guard_fewer_false_positives_passes(self):
        d0 = _metrics(0.96, 0.96, 0.96, neg_with_pred=20)
        self.assertTrue(
            precision_guard(d0, _metrics(0.97, 0.96, 0.96, neg_with_pred=5), 0.25)["ok"]
        )

    def test_precision_guard_zero_baseline(self):
        d0 = _metrics(0.96, 0.96, 0.96, neg_with_pred=0)
        self.assertTrue(
            precision_guard(d0, _metrics(0.96, 0.96, 0.96, neg_with_pred=0), 0.25)["ok"]
        )
        self.assertFalse(
            precision_guard(d0, _metrics(0.96, 0.96, 0.96, neg_with_pred=1), 0.25)["ok"]
        )

    def test_precision_guard_unmeasurable_fails(self):
        old_report = {"span_precision": 0.9, "span_recall": 0.9, "span_f1": 0.9, "neg_rows": 10}
        self.assertFalse(precision_guard(old_report, old_report, 0.25)["ok"])
        no_neg = _metrics(0.9, 0.9, 0.9, neg_rows=0, neg_with_pred=0)
        self.assertFalse(precision_guard(no_neg, no_neg, 0.25)["ok"])

    def test_span_metrics_counts_flagged_name_free_rows(self):
        rows = [
            {"example_id": "a", "text": "x" * 20, "spans": []},
            {"example_id": "b", "text": "x" * 20, "spans": []},
            {"example_id": "c", "text": "x" * 20, "spans": [{"start": 0, "end": 3}]},
        ]
        m = span_metrics(rows, [[(0, 2), (4, 6)], [], [(0, 3)]])
        self.assertEqual((m["neg_rows"], m["neg_rows_with_pred"]), (2, 1))
        self.assertEqual(m["neg_false_positives"], 2)


class NegativeRowsTest(TempDirTest):
    def test_loads_name_free_rows(self):
        path = self.write_jsonl("neg.jsonl", [_neg_record("n1"), _neg_record("n2")])
        rows = load_negative_rows(path)
        self.assertEqual([r["example_id"] for r in rows], ["n1", "n2"])
        self.assertTrue(all(r["spans"] == [] for r in rows))
        self.assertEqual({r["distractor"] for r in rows}, {"calendar"})

    def test_rejects_a_row_without_a_distractor_type(self):
        rec = _neg_record("n1")
        del rec["distractor"]
        with self.assertRaises(ValueError):
            load_negative_rows(self.write_jsonl("nodist.jsonl", [rec]))

    def test_rejects_rows_with_an_entity_or_span(self):
        with_entity = _neg_record("n1", entities=[{"type": "PERSON", "value": "Ann"}])
        with_span = _neg_record("n2", span=[0, 3])
        for rec in (with_entity, with_span):
            path = self.write_jsonl("bad.jsonl", [rec])
            with self.assertRaises(ValueError):
                load_negative_rows(path)

    def test_rejects_an_empty_file(self):
        with self.assertRaises(ValueError):
            load_negative_rows(self.write_jsonl("empty.jsonl", []))

    def test_negative_fp_metrics(self):
        rows = [{"example_id": f"n{i}", "text": "the rose bush", "spans": []} for i in range(4)]
        m = negative_fp_metrics(rows, [[], [(4, 8)], [(4, 8), (9, 13)], []])
        self.assertEqual((m["rows_with_pred"], m["pred_spans"], m["fp_row_rate"]), (2, 3, 0.5))
        self.assertEqual([f["id"] for f in m["flagged"]], ["n1", "n2"])
        self.assertEqual(m["flagged"][0]["spans"], [[4, 8, "rose"]])

    def test_negative_fp_metrics_refuses_labeled_rows(self):
        rows = [{"example_id": "a", "text": "t", "spans": [{"start": 0, "end": 1}]}]
        with self.assertRaises(ValueError):
            negative_fp_metrics(rows, [[]])


class ShortNegativesSlotTest(TempDirTest):
    def test_absent_file_is_skipped_never_passed(self):
        resolved = resolve_optional_input(self.tmp, "missing.jsonl", None)
        self.assertEqual(resolved["state"], "absent")
        chk = short_negatives_check(resolved)
        self.assertEqual(chk["status"], "skipped")
        self.assertIsNone(chk["ok"])
        self.assertFalse(chk["required"])
        self.assertIn("skipped: not provided", chk["detail"])

    def test_present_but_unpinned_or_mismatched_blocks(self):
        path = self.write_jsonl("neg.jsonl", [_neg_record("n1")])
        unpinned = resolve_optional_input(self.tmp, "neg.jsonl", None)
        mismatch = resolve_optional_input(self.tmp, "neg.jsonl", "0" * 64)
        self.assertEqual((unpinned["state"], mismatch["state"]), ("unpinned", "mismatch"))
        for resolved in (unpinned, mismatch):
            chk = short_negatives_check(resolved)
            self.assertEqual((chk["status"], chk["required"]), ("fail", True))
        ok = resolve_optional_input(self.tmp, "neg.jsonl", file_sha256(path))
        self.assertEqual(ok["state"], "ok")
        self.assertEqual(short_negatives_check(ok)["status"], "fail")  # verified, not scored

    def test_gate_threshold(self):
        d0 = {"rows": 100, "rows_with_pred": 9, "pred_spans": 10, "fp_row_rate": 0.09}
        d1_ok = {"rows": 100, "rows_with_pred": 5, "pred_spans": 5, "fp_row_rate": 0.05}
        d1_bad = {"rows": 100, "rows_with_pred": 6, "pred_spans": 6, "fp_row_rate": 0.06}
        resolved = {"state": "ok", "path": "neg.jsonl", "sha256": "x"}
        passed = short_negatives_check(resolved, short_negatives_gate(d0, d1_ok, 0.05))
        failed = short_negatives_check(resolved, short_negatives_gate(d0, d1_bad, 0.05))
        self.assertEqual((passed["status"], passed["required"]), ("pass", True))
        self.assertEqual(failed["status"], "fail")


GATED = ["place", "org_product", "calendar", "role_only"]
REPORT_ONLY = ["homograph"]


def _typed_rows(per_type=3):
    rows = []
    for dtype in [*GATED, *REPORT_ONLY]:
        for i in range(per_type):
            rows.append(
                {
                    "example_id": f"{dtype}-{i}",
                    "text": "Frank feedback on the May deadline",
                    "spans": [],
                    "distractor": dtype,
                }
            )
    return rows


def _preds(rows, flag):
    """Flag (predict one span on) every row for which flag(row) is true."""
    return [[(0, 5)] if flag(r) else [] for r in rows]


class ShortNegativesByDistractorTest(unittest.TestCase):
    def test_per_type_split(self):
        rows = _typed_rows()
        d0 = _preds(rows, lambda r: r["example_id"] == "place-0")
        d1 = _preds(rows, lambda r: r["example_id"] in ("place-0", "calendar-1", "calendar-2"))
        rep = short_negatives_report(rows, d0, d1, GATED, REPORT_ONLY, 0.05)
        self.assertEqual(set(rep["by_distractor"]), set(GATED + REPORT_ONLY))
        self.assertEqual(rep["by_distractor"]["calendar"]["d0"]["rows_with_pred"], 0)
        self.assertEqual(rep["by_distractor"]["calendar"]["d1"]["rows_with_pred"], 2)
        self.assertEqual(rep["by_distractor"]["place"]["d1"]["rows_with_pred"], 1)
        self.assertEqual(rep["gated"]["d1"]["rows"], 12)
        self.assertEqual(rep["gated"]["d1"]["rows_with_pred"], 3)
        self.assertEqual(rep["report_only"]["d1"]["rows"], 3)
        self.assertEqual(rep["gate"]["d1_fp_row_rate"], 0.25)
        self.assertFalse(rep["gate"]["ok"])

    def test_homograph_rows_never_affect_the_gate(self):
        rows = _typed_rows()
        clean = _preds(rows, lambda r: False)
        homograph_all = _preds(rows, lambda r: r["distractor"] == "homograph")
        base = short_negatives_report(rows, clean, clean, GATED, REPORT_ONLY, 0.05)
        noisy = short_negatives_report(rows, clean, homograph_all, GATED, REPORT_ONLY, 0.05)
        self.assertTrue(base["gate"]["ok"])
        self.assertTrue(noisy["gate"]["ok"])
        self.assertEqual(base["gate"]["d1_fp_row_rate"], noisy["gate"]["d1_fp_row_rate"])
        # ...but they are reported, D0 vs D1, with the flagged texts and spans
        homograph = noisy["by_distractor"]["homograph"]
        self.assertEqual(
            (homograph["d0"]["rows_with_pred"], homograph["d1"]["rows_with_pred"]), (0, 3)
        )
        self.assertEqual(homograph["d1"]["flagged"][0]["spans"], [[0, 5, "Frank"]])

    def test_gate_passes_at_the_threshold(self):
        rows = _typed_rows(per_type=25)  # 100 gated rows
        d1 = _preds(rows, lambda r: r["distractor"] == "place" and r["example_id"] < "place-13")
        rep = short_negatives_report(rows, d1, d1, GATED, REPORT_ONLY, 0.05)
        self.assertEqual(rep["gate"]["d1_rows_with_pred"], 5)
        self.assertTrue(rep["gate"]["ok"])

    def test_unknown_or_overlapping_types_raise(self):
        rows = _typed_rows()
        rows[0] = dict(rows[0], distractor="surname_only")
        empty = [[] for _ in rows]
        with self.assertRaises(ValueError):
            short_negatives_report(rows, empty, empty, GATED, REPORT_ONLY, 0.05)
        with self.assertRaises(ValueError):
            short_negatives_report(
                _typed_rows(), empty, empty, GATED, [*REPORT_ONLY, "place"], 0.05
            )

    def test_no_gated_rows_fails(self):
        rows = [r for r in _typed_rows() if r["distractor"] == "homograph"]
        empty = [[] for _ in rows]
        rep = short_negatives_report(rows, empty, empty, GATED, REPORT_ONLY, 0.05)
        self.assertFalse(rep["gate"]["ok"])


def _probe_temp(miss, all_ok=True):
    thresholds = {
        "format_valid": {"ok": True, "detail": ""},
        "reward_signal": {"ok": all_ok, "detail": ""},
    }
    return {"outcome_rates": {"complete_miss": miss}, "thresholds": thresholds}


class ProbeGateTest(unittest.TestCase):
    band = [0.01, 0.30]

    def test_passes_with_go_and_in_band(self):
        gate = probe_gate({"1.0": _probe_temp(0.5), "0.7": _probe_temp(0.12)}, self.band, True)
        self.assertTrue(gate["ok"])
        self.assertEqual(gate["passing_temperatures"], ["0.7"])

    def test_go_outside_band_fails_only_when_band_required(self):
        per_temp = {"1.0": _probe_temp(0.0)}
        self.assertFalse(probe_gate(per_temp, self.band, True)["ok"])
        self.assertTrue(probe_gate(per_temp, self.band, False)["ok"])

    def test_no_go_fails(self):
        self.assertFalse(probe_gate({"1.0": _probe_temp(0.1, all_ok=False)}, self.band, True)["ok"])

    def test_comparison_table(self):
        table = probe_comparison({"1.0": _probe_temp(0.53)}, {"1.0": _probe_temp(0.08)}, self.band)
        self.assertEqual(table["1.0"]["d0_miss_rate"], 0.53)
        self.assertFalse(table["1.0"]["d0_in_band"])
        self.assertTrue(table["1.0"]["d1_in_band"])
        with self.assertRaises(ValueError):
            probe_comparison({"1.0": _probe_temp(0.5)}, {"0.7": _probe_temp(0.1)}, self.band)

    def test_per_temperature_metrics_uses_the_m07a_table(self):
        with open(os.path.join(REPO, "configs", "m07a_probe.yaml"), encoding="utf-8") as f:
            thresholds = yaml.safe_load(f)["thresholds"]
        samples = []
        for g in range(4):
            for k in range(8):
                p = 0.9 if k % 2 else 0.1
                samples.append(
                    {
                        "group_id": f"g{g}|t1.0",
                        "temperature": 1.0,
                        "status": "valid",
                        "text": f"Message {g} {k} for Ann about item {g * 8 + k}",
                        "raw_text": "",
                        "json_tolerant_ok": True,
                        "target_span": (16, 19),
                        "n_chars": 40,
                        "p_target_mean": p,
                        "p_target_min": p,
                        "outcome": "complete_miss" if k % 4 == 0 else "exact_hit",
                        "name_position": 0.4,
                        "extra_person_flags": 0,
                        "r_prob": 1 - p,
                    }
                )
        per_temp = per_temperature_metrics(samples, [1.0], 10.0, thresholds, 0)
        self.assertEqual(per_temp["1.0"]["outcome_rates"]["complete_miss"], 0.25)
        self.assertEqual(
            set(per_temp["1.0"]["thresholds"]),
            {"format_valid", "duplicates", "within_group_diversity", "reward_signal", "truncation"},
        )


class ManifestAndCopyTest(TempDirTest):
    def make_candidate(self):
        cand = os.path.join(self.tmp, "d1-candidate")
        os.makedirs(cand)
        for name, content in (
            ("model.safetensors", b"weights"),
            ("config.json", b"{}"),
            ("tokenizer.json", b"{}"),
            ("tokenizer_config.json", b"{}"),
        ):
            with open(os.path.join(cand, name), "wb") as f:
                f.write(content)
        return cand

    def fields(self):
        return {
            "source_run_id": "d1-20261108-145253",
            "source_dir": "/drive/d1-20261108-145253",
            "freeze_run_id": "d1-freeze-x",
            "repo_commit": "abc1234",
            "source_repo_commit": "b8550eb",
            "locks": {"final_eval_sha256": "f" * 64},
            "checks": {"a": make_check(True)},
            "gates": {"final_eval_f1": {"ok": True}},
        }

    def test_frozen_dir_name(self):
        self.assertEqual(
            frozen_dir_for("/drive/root", "d1-20261108-145253"),
            "/drive/root/d1-frozen-d1-20261108-145253",
        )

    def test_freeze_copies_verifies_and_writes_manifest(self):
        cand = self.make_candidate()
        sha = file_sha256(os.path.join(cand, "model.safetensors"))
        frozen = os.path.join(self.tmp, "d1-frozen-run")
        manifest = freeze_candidate(cand, frozen, sha, self.fields())
        self.assertEqual(manifest["status"], "frozen")
        self.assertEqual(manifest["d1_model_sha256"], sha)
        self.assertEqual(set(manifest["files"]), set(os.listdir(cand)))
        self.assertEqual(dir_file_hashes(frozen, exclude=(MANIFEST_FILE,)), manifest["files"])
        with open(os.path.join(frozen, MANIFEST_FILE), encoding="utf-8") as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk["source_run_id"], "d1-20261108-145253")
        self.assertEqual(on_disk["repo_commit"], "abc1234")
        self.assertFalse(os.path.exists(frozen + ".partial"))

    def test_never_overwrites_an_existing_frozen_dir(self):
        cand = self.make_candidate()
        sha = file_sha256(os.path.join(cand, "model.safetensors"))
        frozen = os.path.join(self.tmp, "d1-frozen-run")
        os.makedirs(frozen)
        with open(os.path.join(frozen, "keep.txt"), "w") as f:
            f.write("original")
        with self.assertRaises(FileExistsError):
            freeze_candidate(cand, frozen, sha, self.fields())
        self.assertEqual(os.listdir(frozen), ["keep.txt"])

    def test_refuses_a_candidate_with_the_wrong_sha(self):
        cand = self.make_candidate()
        frozen = os.path.join(self.tmp, "d1-frozen-run")
        with self.assertRaises(ValueError):
            freeze_candidate(cand, frozen, "0" * 64, self.fields())
        self.assertFalse(os.path.exists(frozen))

    def test_manifest_needs_model_weights(self):
        with self.assertRaises(ValueError):
            build_freeze_manifest(files={"config.json": "x"}, **self.fields())


class FreezeConfigTest(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(REPO, "configs", "m07b_freeze_d1.yaml"), encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def test_pinned_configs_match_the_repo(self):
        source, probe = self.cfg["source"], self.cfg["probe"]
        self.assertEqual(
            file_sha256(os.path.join(REPO, source["train_config"])),
            source["train_config_sha256"],
        )
        self.assertEqual(file_sha256(os.path.join(REPO, probe["config"])), probe["config_sha256"])

    def test_locks_agree_with_the_training_and_probe_configs(self):
        with open(os.path.join(REPO, self.cfg["source"]["train_config"]), encoding="utf-8") as f:
            train = yaml.safe_load(f)
        with open(os.path.join(REPO, self.cfg["probe"]["config"]), encoding="utf-8") as f:
            probe = yaml.safe_load(f)
        locks = self.cfg["locks"]
        self.assertEqual(locks["final_eval_sha256"], train["locks"]["final_eval_sha256"])
        self.assertEqual(locks["d0_sha256"], train["detector"]["sha256"])
        self.assertEqual(locks["d0_sha256"], probe["detector"]["sha256"])

    def test_short_negatives_pin_matches_when_the_file_exists(self):
        neg = self.cfg["short_negatives"]
        resolved = resolve_optional_input(REPO, neg["path"], neg["sha256"])
        self.assertIn(resolved["state"], ("absent", "ok"))


if __name__ == "__main__":
    unittest.main()
