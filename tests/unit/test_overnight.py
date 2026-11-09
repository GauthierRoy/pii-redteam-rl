import json
import os
import shutil
import tempfile
import unittest

from pii_redteam.freeze_gates import file_sha256
from pii_redteam.overnight import (
    ProgressLog,
    completed_train_run,
    find_reusable_probe_run,
    find_reusable_train_run,
)

M07A = {"thresholds": {"miss_rate_range": [0.01, 0.3]}, "generation": {"n": 8}}


class OvernightTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def make_train_run(self, name, epochs=(2, 3), status="trained_pending_final_eval"):
        run = os.path.join(self.tmp, name)
        checkpoints = {}
        for e in epochs:
            d = os.path.join(run, f"d1-epoch{e}")
            os.makedirs(d)
            path = os.path.join(d, "model.safetensors")
            with open(path, "wb") as f:
                f.write(f"weights {e}".encode())
            checkpoints[str(e)] = {"epoch": e, "dir": f"d1-epoch{e}", "sha256": file_sha256(path)}
        report = {
            "run_id": name,
            "status": status,
            "finished_utc": "2026-11-10T03:00:00",
            "experiments_entry": {"run_id": name},
            "epoch_checkpoints": checkpoints,
            "freeze_selection": {"chosen_epoch": epochs[0]},
        }
        with open(os.path.join(run, "d1_report.json"), "w") as f:
            json.dump(report, f)
        return run

    def test_complete_run_is_reusable(self):
        run = self.make_train_run("d1v2-20261110-010101")
        self.assertEqual(completed_train_run(run, [2, 3]), (True, "complete"))

    def test_incomplete_runs_are_not_reusable(self):
        run = self.make_train_run("d1v2-20261110-010101", epochs=(2,))
        self.assertFalse(completed_train_run(run, [2, 3])[0])
        run = self.make_train_run("d1v2-20261110-020202", status="running")
        self.assertFalse(completed_train_run(run, [2, 3])[0])
        self.assertFalse(completed_train_run(os.path.join(self.tmp, "nothing"), [2, 3])[0])

    def test_changed_weights_are_not_reusable(self):
        run = self.make_train_run("d1v2-20261110-010101")
        with open(os.path.join(run, "d1-epoch3", "model.safetensors"), "ab") as f:
            f.write(b"x")
        ok, reason = completed_train_run(run, [2, 3])
        self.assertFalse(ok)
        self.assertIn("sha256", reason)

    def test_newest_reusable_run_wins(self):
        good = self.make_train_run("d1v2-20261110-010101")
        self.make_train_run("d1v2-20261110-050505", status="running")  # newer, broken
        found, seen = find_reusable_train_run(self.tmp, "d1v2", [2, 3])
        self.assertEqual(found, good)
        self.assertEqual([s["reusable"] for s in seen], [False, True])
        self.assertIsNone(find_reusable_train_run(self.tmp, "nope", [2, 3])[0])

    def make_probe_run(self, name, sha, finished=True, generation=None):
        run = os.path.join(self.tmp, name)
        os.makedirs(run)
        if finished:
            cfg = {
                "detector": {"sha256": sha},
                "thresholds": M07A["thresholds"],
                "generation": generation or M07A["generation"],
            }
            with open(os.path.join(run, "probe_report.json"), "w") as f:
                json.dump({"config": cfg}, f)
            open(os.path.join(run, "samples.jsonl"), "w").close()
        else:
            open(os.path.join(run, "samples.partial.jsonl"), "w").close()
        return run

    def test_probe_reuse_needs_a_finished_run_for_the_same_sha_and_protocol(self):
        good = self.make_probe_run("m07a-d1v2-20261110-010101", "a" * 64)
        self.make_probe_run("m07a-d1v2-20261110-020202", "a" * 64, finished=False)
        self.make_probe_run("m07a-d1v2-20261110-030303", "b" * 64)
        self.make_probe_run("m07a-d1v2-20261110-040404", "a" * 64, generation={"n": 2})
        self.assertEqual(find_reusable_probe_run(self.tmp, "m07a-d1v2", "a" * 64, M07A), good)
        self.assertIsNone(find_reusable_probe_run(self.tmp, "m07a-d1v2", "c" * 64, M07A))

    def test_progress_log_records_stages_and_sessions(self):
        path = os.path.join(self.tmp, "progress.json")
        log = ProgressLog(path, session="s1")
        log.start("train")
        log.end("train", "reused", run_dir="x")
        with open(path) as f:
            state = json.load(f)
        stage = state["sessions"][0]["stages"]["train"]
        self.assertEqual((stage["status"], stage["run_dir"]), ("reused", "x"))
        self.assertIn("seconds", stage)
        ProgressLog(path, session="s2")
        with open(path) as f:
            self.assertEqual([s["session"] for s in json.load(f)["sessions"]], ["s1", "s2"])


if __name__ == "__main__":
    unittest.main()
