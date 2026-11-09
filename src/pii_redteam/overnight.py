"""Torch-free helpers for the unattended M07b overnight notebook (train -> choose -> freeze).

Resume rules: a training run is reused only when its report is complete and every saved
epoch checkpoint on disk still has the sha256 the report recorded; a probe run is reused
only when it finished (report + samples) for the same detector sha and M07a protocol.
Progress is written atomically after every stage, so Drive always holds the latest state.
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import time

from pii_redteam.freeze_gates import file_sha256

REUSABLE_TRAIN_STATUSES = ("trained_pending_final_eval",)


def completed_train_run(run_dir: str, save_epochs: list[int]) -> tuple[bool, str]:
    """(reusable, reason) for one train-d1 run directory."""
    path = os.path.join(run_dir, "d1_report.json")
    if not os.path.exists(path):
        return False, "no d1_report.json"
    try:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable d1_report.json: {exc}"
    if report.get("status") not in REUSABLE_TRAIN_STATUSES:
        return False, f"status {report.get('status')!r}"
    if not report.get("finished_utc") or not report.get("experiments_entry"):
        return False, "report not finalized"
    checkpoints = report.get("epoch_checkpoints") or {}
    if sorted(int(k) for k in checkpoints) != sorted(save_epochs):
        return False, f"saved epochs {sorted(checkpoints)} != {sorted(save_epochs)}"
    if (report.get("freeze_selection") or {}).get("chosen_epoch") is None:
        return False, "no freeze_selection.chosen_epoch"
    for ckpt in checkpoints.values():
        weights = os.path.join(run_dir, ckpt["dir"], "model.safetensors")
        if not os.path.exists(weights):
            return False, f"missing {ckpt['dir']}/model.safetensors"
        if file_sha256(weights) != ckpt["sha256"]:
            return False, f"{ckpt['dir']} sha256 differs from the report"
    return True, "complete"


def find_reusable_train_run(
    work_dir: str, prefix: str, save_epochs: list[int]
) -> tuple[str | None, list[dict]]:
    """Newest reusable `<prefix>-*` run dir, plus why each other candidate was skipped."""
    seen = []
    for run_dir in sorted(glob.glob(os.path.join(work_dir, f"{prefix}-2*")), reverse=True):
        if not os.path.isdir(run_dir):
            continue
        ok, reason = completed_train_run(run_dir, save_epochs)
        seen.append({"run_dir": run_dir, "reusable": ok, "reason": reason})
        if ok:
            return run_dir, seen
    return None, seen


def find_reusable_probe_run(
    work_dir: str, prefix: str, detector_sha: str, m07a_cfg: dict
) -> str | None:
    """Newest FINISHED probe run that scored `detector_sha` under the M07a protocol."""
    for run_dir in sorted(glob.glob(os.path.join(work_dir, f"{prefix}-2*")), reverse=True):
        report_path = os.path.join(run_dir, "probe_report.json")
        if not (
            os.path.exists(report_path) and os.path.exists(os.path.join(run_dir, "samples.jsonl"))
        ):
            continue
        try:
            with open(report_path, encoding="utf-8") as f:
                cfg = json.load(f)["config"]
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        if (
            cfg["detector"]["sha256"] == detector_sha
            and cfg["thresholds"] == m07a_cfg["thresholds"]
            and cfg["generation"] == m07a_cfg["generation"]
        ):
            return run_dir
    return None


class ProgressLog:
    """Top-level progress/status JSON, rewritten atomically after every stage change."""

    def __init__(self, path: str, session: str | None = None):
        self.path = path
        self.state: dict = {"sessions": []}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.state = json.load(f)
            except (OSError, json.JSONDecodeError):
                self.state = {"sessions": [], "note": "previous progress file was unreadable"}
        self.session: dict = {
            "session": session or datetime.datetime.now(datetime.UTC).isoformat(),
            "stages": {},
        }
        self.state["sessions"].append(self.session)
        self._t0: dict[str, float] = {}
        self.write()

    def write(self) -> None:
        self.state["updated_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    def start(self, stage: str, **info) -> None:
        self._t0[stage] = time.time()
        self.session["stages"][stage] = {
            "status": "running",
            "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            **info,
        }
        self.state["current_stage"] = stage
        self.write()

    def end(self, stage: str, status: str, **info) -> None:
        entry = self.session["stages"].setdefault(stage, {})
        entry.update(status=status, **info)
        entry["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        if stage in self._t0:
            entry["seconds"] = round(time.time() - self._t0[stage], 1)
        self.state["current_stage"] = None
        self.state["last_stage"] = {"stage": stage, "status": status}
        self.write()
