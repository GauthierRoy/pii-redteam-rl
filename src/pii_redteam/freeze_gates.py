"""Pure (torch-free) logic for freezing D1 (M07b.3/.4): checks, gates, manifest, copy.

`freeze_d1.py` does the model work and calls into this module. Conventions:
  - every check is {"ok": True|False|None, "status": pass|fail|skipped, "required", "detail"};
  - a required check that did not pass (fail OR skipped) blocks the freeze;
  - an optional check may be skipped, but the skip is always recorded, never a pass.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil

from pii_redteam.d1_metrics import negative_fp_metrics

MANIFEST_FILE = "freeze_manifest.json"
UNRECORDED_COMMITS = {None, "", "unrecorded", "recorded-by-notebook"}


# ---------------------------------------------------------------- checks


def make_check(ok: bool | None, detail: object = "", *, required: bool = True) -> dict:
    """One check record. `ok=None` means skipped (did not run)."""
    status = "skipped" if ok is None else ("pass" if ok else "fail")
    return {
        "ok": None if ok is None else bool(ok),
        "status": status,
        "required": required,
        "detail": str(detail),
    }


def freeze_decision(checks: dict[str, dict]) -> dict:
    """Freeze only if every required check passed. Skipped optional checks are listed."""
    failing = sorted(n for n, c in checks.items() if c["required"] and c["ok"] is not True)
    skipped_optional = sorted(
        n for n, c in checks.items() if not c["required"] and c["status"] == "skipped"
    )
    failing_optional = sorted(
        n for n, c in checks.items() if not c["required"] and c["status"] == "fail"
    )
    return {
        "freeze_allowed": not failing,
        "failing_required": failing,
        "failing_optional": failing_optional,
        "skipped_optional": skipped_optional,
    }


def check_counts(checks: dict[str, dict]) -> dict:
    return {
        status: sum(1 for c in checks.values() if c["status"] == status)
        for status in ("pass", "fail", "skipped")
    }


def repo_commit_recorded(commit: str | None) -> bool:
    return commit not in UNRECORDED_COMMITS


# ---------------------------------------------------------------- final-eval gates


def final_eval_f1_gate(d0: dict, d1: dict, max_drop: float) -> dict:
    """No-regression gate: D1 span-F1 >= D0 span-F1 - max_drop."""
    drop = round(d0["span_f1"] - d1["span_f1"], 4)
    return {
        "d0_f1": d0["span_f1"],
        "d1_f1": d1["span_f1"],
        "f1_drop": drop,
        "max_drop": max_drop,
        "floor": round(d0["span_f1"] - max_drop, 4),
        "ok": drop <= max_drop,
    }


def neg_row_fp_rate(metrics: dict) -> float | None:
    """Share of name-free rows with at least one predicted span (None if not measurable)."""
    neg_rows = metrics.get("neg_rows")
    with_pred = metrics.get("neg_rows_with_pred")
    if not neg_rows or with_pred is None:
        return None
    return with_pred / neg_rows


def precision_guard(d0: dict, d1: dict, max_relative_increase: float) -> dict:
    """Pre-registered precision guard on the name-free rows of the same partition.

    Passes when D1's name-free-row false-positive rate is at most
    D0's rate * (1 + max_relative_increase). With D0 at 0, D1 must also be 0.
    Not measurable (no name-free rows, or an old report without the row count) fails.
    """
    d0_rate, d1_rate = neg_row_fp_rate(d0), neg_row_fp_rate(d1)
    out = {
        "d0_precision": d0["span_precision"],
        "d1_precision": d1["span_precision"],
        "precision_drop": round(d0["span_precision"] - d1["span_precision"], 4),
        "d0_neg_rows": d0.get("neg_rows"),
        "d1_neg_rows": d1.get("neg_rows"),
        "d0_neg_rows_with_pred": d0.get("neg_rows_with_pred"),
        "d1_neg_rows_with_pred": d1.get("neg_rows_with_pred"),
        "d0_neg_false_positive_spans": d0.get("neg_false_positives"),
        "d1_neg_false_positive_spans": d1.get("neg_false_positives"),
        "d0_neg_fp_row_rate": None if d0_rate is None else round(d0_rate, 4),
        "d1_neg_fp_row_rate": None if d1_rate is None else round(d1_rate, 4),
        "max_relative_increase": max_relative_increase,
    }
    if d0_rate is None or d1_rate is None:
        return out | {"relative_increase": None, "allowed_d1_rate": None, "ok": False}
    allowed = d0_rate * (1 + max_relative_increase)
    if d0_rate:
        relative = round((d1_rate - d0_rate) / d0_rate, 4)
    else:
        relative = 0.0 if d1_rate == 0 else None
    return out | {
        "relative_increase": relative,
        "allowed_d1_rate": round(allowed, 4),
        "ok": d1_rate <= allowed + 1e-12,
    }


# ---------------------------------------------------------------- short negatives


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_optional_input(repo_dir: str, rel_path: str, expected_sha: str | None) -> dict:
    """State of an optional, hash-pinned input: absent | unpinned | mismatch | ok."""
    path = os.path.join(repo_dir, rel_path)
    if not os.path.exists(path):
        return {"state": "absent", "path": rel_path, "sha256": None}
    actual = file_sha256(path)
    if expected_sha is None:
        state = "unpinned"
    elif actual != expected_sha:
        state = "mismatch"
    else:
        state = "ok"
    return {"state": state, "path": rel_path, "sha256": actual}


def short_negatives_gate(d0: dict, d1: dict, max_fp_row_rate: float) -> dict:
    return {
        "d0_rows_with_pred": d0["rows_with_pred"],
        "d1_rows_with_pred": d1["rows_with_pred"],
        "d0_pred_spans": d0["pred_spans"],
        "d1_pred_spans": d1["pred_spans"],
        "rows": d1["rows"],
        "d0_fp_row_rate": d0["fp_row_rate"],
        "d1_fp_row_rate": d1["fp_row_rate"],
        "max_fp_row_rate": max_fp_row_rate,
        "ok": d1["fp_row_rate"] is not None and d1["fp_row_rate"] <= max_fp_row_rate,
    }


def short_negatives_report(
    rows: list[dict],
    d0_preds: list[list[tuple[int, int]]],
    d1_preds: list[list[tuple[int, int]]],
    gated_types: list[str],
    report_only_types: list[str],
    max_fp_row_rate: float,
) -> dict:
    """Per-distractor D0 vs D1 false positives; the gate uses the gated types only.

    Report-only types (homograph: names in their ordinary sense, "rose bush") are shown
    side by side with their flagged texts but can never change the gate. A row whose
    type is in neither list is an error, so a new type cannot slip in ungated.
    """
    if set(gated_types) & set(report_only_types):
        raise ValueError("a distractor type cannot be both gated and report-only")
    known = set(gated_types) | set(report_only_types)
    unknown = sorted({r["distractor"] for r in rows} - known)
    if unknown:
        raise ValueError(f"unknown distractor types: {unknown}")
    if not (len(rows) == len(d0_preds) == len(d1_preds)):
        raise ValueError("rows and predictions differ in length")

    def subset(types: set[str]) -> dict:
        idx = [i for i, r in enumerate(rows) if r["distractor"] in types]
        sub = [rows[i] for i in idx]
        return {
            "d0": negative_fp_metrics(sub, [d0_preds[i] for i in idx]),
            "d1": negative_fp_metrics(sub, [d1_preds[i] for i in idx]),
        }

    by_type = {t: subset({t}) for t in [*gated_types, *report_only_types]}
    gated = subset(set(gated_types))
    report_only = subset(set(report_only_types))
    if not gated["d1"]["rows"]:
        gate: dict = {"rows": 0, "max_fp_row_rate": max_fp_row_rate, "ok": False}
    else:
        gate = short_negatives_gate(gated["d0"], gated["d1"], max_fp_row_rate)
    gate["gated_types"] = list(gated_types)
    gate["report_only_types"] = list(report_only_types)
    return {
        "rows": len(rows),
        "by_distractor": by_type,
        "gated": gated,
        "report_only": report_only,
        "gate": gate,
    }


def short_negatives_check(resolved: dict, gate: dict | None = None) -> dict:
    """Check record for the optional short-negatives slot.

    Absent -> skipped (optional, recorded). Present but unpinned or hash-mismatched ->
    a REQUIRED failure: a file that exists but cannot be trusted must not be ignored.
    """
    state = resolved["state"]
    if state == "absent":
        return make_check(None, f"skipped: not provided ({resolved['path']})", required=False)
    if state == "unpinned":
        return make_check(
            False, f"{resolved['path']} exists but its sha256 is not pinned in the config"
        )
    if state == "mismatch":
        return make_check(False, f"{resolved['path']} sha256 {resolved['sha256']} != pinned")
    if gate is None:
        return make_check(False, "file verified but not scored")
    return make_check(gate["ok"], gate)


# ---------------------------------------------------------------- probe


def probe_summary(per_temp: dict[str, dict], band: list[float]) -> dict[str, dict]:
    """Per temperature: miss rate, whether it is inside the band, and the go/no-go."""
    lo, hi = band
    out = {}
    for temp, metrics in per_temp.items():
        miss = metrics.get("outcome_rates", {}).get("complete_miss")
        out[temp] = {
            "miss_rate": miss,
            "miss_rate_in_band": miss is not None and lo <= miss <= hi,
            "go": all(c["ok"] for c in metrics["thresholds"].values()),
            "failed_thresholds": sorted(n for n, c in metrics["thresholds"].items() if not c["ok"]),
        }
    return out


def probe_gate(per_temp_d1: dict[str, dict], band: list[float], require_band: bool) -> dict:
    """M07b.4: the M07a go/no-go table passes against D1 at one temperature at least.

    With `require_band`, that temperature's miss rate must also sit inside the band
    (the M07b acceptance gate reads "miss rate inside the pre-declared band"; the M07a
    table alone also accepts a high live-group share instead).
    """
    summary = probe_summary(per_temp_d1, band)
    passing = [
        t for t, s in summary.items() if s["go"] and (s["miss_rate_in_band"] or not require_band)
    ]
    return {
        "band": list(band),
        "require_miss_rate_in_band": require_band,
        "per_temperature": summary,
        "passing_temperatures": passing,
        "ok": bool(passing),
    }


def probe_comparison(d0_per_temp: dict, d1_per_temp: dict, band: list[float]) -> dict:
    """D0 vs D1 on the same probe texts, per temperature."""
    d0, d1 = probe_summary(d0_per_temp, band), probe_summary(d1_per_temp, band)
    if d0.keys() != d1.keys():
        raise ValueError("D0 and D1 probe metrics cover different temperatures")
    return {
        t: {
            "d0_miss_rate": d0[t]["miss_rate"],
            "d1_miss_rate": d1[t]["miss_rate"],
            "d0_in_band": d0[t]["miss_rate_in_band"],
            "d1_in_band": d1[t]["miss_rate_in_band"],
            "d0_go": d0[t]["go"],
            "d1_go": d1[t]["go"],
        }
        for t in d1
    }


# ---------------------------------------------------------------- manifest + copy


def dir_file_hashes(root: str, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    """sha256 of every file under root, keyed by its relative path (sorted)."""
    hashes = {}
    for base, _, files in os.walk(root):
        for name in files:
            path = os.path.join(base, name)
            rel = os.path.relpath(path, root)
            if rel not in exclude:
                hashes[rel] = file_sha256(path)
    return dict(sorted(hashes.items()))


def frozen_dir_for(frozen_root: str, source_run_id: str) -> str:
    return os.path.join(frozen_root, f"d1-frozen-{source_run_id}")


def build_freeze_manifest(
    *,
    files: dict[str, str],
    source_run_id: str,
    source_dir: str,
    freeze_run_id: str,
    repo_commit: str | None,
    source_repo_commit: str | None,
    locks: dict,
    checks: dict[str, dict],
    gates: dict,
    created_utc: str | None = None,
) -> dict:
    if "model.safetensors" not in files:
        raise ValueError("a frozen detector needs model.safetensors")
    return {
        "kind": "frozen-detector",
        "model": "D1",
        "status": "frozen",
        "created_utc": created_utc or datetime.datetime.now(datetime.UTC).isoformat(),
        "source_run_id": source_run_id,
        "source_dir": source_dir,
        "freeze_run_id": freeze_run_id,
        "repo_commit": repo_commit,
        "source_repo_commit": source_repo_commit,
        "d1_model_sha256": files["model.safetensors"],
        "locks": locks,
        "files": files,
        "checks": checks,
        "gates": gates,
    }


def freeze_candidate(
    candidate_dir: str, frozen_dir: str, expected_model_sha: str, manifest_fields: dict
) -> dict:
    """Copy the candidate to `frozen_dir`, verify every file, write the manifest.

    Never overwrites: an existing `frozen_dir` raises FileExistsError. The copy is built
    in `<frozen_dir>.partial` and renamed into place only once verified, so a crash
    leaves no directory that looks frozen.
    """
    if os.path.exists(frozen_dir):
        raise FileExistsError(f"frozen dir already exists, refusing to overwrite: {frozen_dir}")
    source_hashes = dir_file_hashes(candidate_dir)
    if source_hashes.get("model.safetensors") != expected_model_sha:
        raise ValueError("candidate model.safetensors does not match the expected sha256")
    partial = frozen_dir + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(candidate_dir, partial)
    copied = dir_file_hashes(partial)
    if copied != source_hashes:
        shutil.rmtree(partial, ignore_errors=True)
        raise OSError("copied files differ from the candidate")
    manifest = build_freeze_manifest(files=copied, **manifest_fields)
    with open(os.path.join(partial, MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if os.path.exists(frozen_dir):
        shutil.rmtree(partial, ignore_errors=True)
        raise FileExistsError(f"frozen dir appeared during the copy: {frozen_dir}")
    os.rename(partial, frozen_dir)
    return manifest
