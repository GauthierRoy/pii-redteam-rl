"""M07b.3/.4: final-eval no-regression, short negatives, re-probe, then freeze D1.

Order: locks -> single-use final-eval (D0 vs D1) -> optional short negatives ->
M07a probe re-run against D1 (D0 rescored on the same texts) -> freeze decision.
D1 is copied to its frozen directory only if every required check passes; otherwise the
report says `freeze_blocked` and lists the failing checks. Gate logic lives in
`freeze_gates.py`; the final-eval scoring is `train_d1.run_d1_final_eval` and the probe
is `probe.run_probe`, both reused unchanged.
"""

from __future__ import annotations

import copy
import datetime
import json
import os
import sys

import torch
import transformers
import yaml
from transformers import AutoModelForTokenClassification, AutoTokenizer

from pii_redteam.d1_metrics import load_negative_rows, no_regression_gate
from pii_redteam.freeze_gates import (
    check_counts,
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
    short_negatives_report,
)
from pii_redteam.probe import _device_dtype, _finalize_samples, _score_with_d0, run_probe
from pii_redteam.probe_metrics import per_temperature_metrics
from pii_redteam.train_d1 import (
    CANDIDATE_DIR,
    FINAL_EVAL_REPORT_FILE,
    REPORT_FILE,
    _final_eval_ids,
    _load_detector,
    _rebuild_splits,
    _verify_detector,
    _write_json,
    predict_spans,
    run_d1_final_eval,
)

FREEZE_REPORT_FILE = "d1_freeze_report.json"
PROBE_REPORT_FILE = "probe_report.json"
PROBE_SAMPLES_FILE = "samples.jsonl"


def _load_pinned_yaml(repo_dir: str, rel_path: str, expected_sha: str) -> tuple[dict, bool, str]:
    path = os.path.join(repo_dir, rel_path)
    actual = file_sha256(path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f), actual == expected_sha, actual


def _rescore_probe_with_d0(samples: list[dict], probe_cfg: dict, device: str) -> dict:
    """Score the D1 probe's own texts with frozen D0 through the same probe scorer."""
    copies = []
    for s in samples:
        if s["temperature"] is None:  # greedy reference: not part of the go/no-go table
            continue
        copies.append(
            {
                "request_id": s["request_id"],
                "group_id": s["group_id"],
                "sample_idx": s["sample_idx"],
                "temperature": s["temperature"],
                "status": s["status"],
                "text": s["text"],
                "raw_text": s["raw_text"],
                "json_tolerant_ok": s["json_tolerant_ok"],
                "target_span": tuple(s["target_span"]) if s["target_span"] else None,
            }
        )
    det = probe_cfg["detector"]
    d0_tok = AutoTokenizer.from_pretrained(det["path"])
    d0 = AutoModelForTokenClassification.from_pretrained(det["path"]).to(device)
    d0.eval()
    _score_with_d0(d0, d0_tok, [s for s in copies if s["status"] == "valid"], det["max_length"])
    lost = sum(1 for s in copies if s.get("d0_truncated_target"))
    _finalize_samples(copies, probe_cfg["rewards"]["logit_clip"])
    del d0
    return per_temperature_metrics(
        copies,
        probe_cfg["generation"]["temperatures"],
        probe_cfg["rewards"]["logit_clip"],
        probe_cfg["thresholds"],
        lost,
    )


def run_freeze_d1(config: dict, repo_dir: str) -> dict:
    started = datetime.datetime.now(datetime.UTC)
    freeze_run_id = f"{config['run_id_prefix']}-{started:%Y%m%d-%H%M%S}"
    work = os.path.join(config["work_dir"], freeze_run_id)
    os.makedirs(work, exist_ok=True)
    report_path = os.path.join(work, FREEZE_REPORT_FILE)
    source_dir = config["source"]["run_dir"]
    locks = config["locks"]
    gates_cfg = config["gates"]
    device, gpu, _ = _device_dtype()
    checks: dict[str, dict] = {}
    gates: dict[str, dict] = {}
    report: dict = {
        "run_id": freeze_run_id,
        "kind": "detector-freeze",
        "status": "running",
        "started_utc": started.isoformat(),
        "config": config,
        "source_run_dir": source_dir,
        "checks": checks,
        "gates": gates,
        "environment": {
            "repo_commit": config.get("repo_commit"),
            "device": gpu,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": sys.version.split()[0],
        },
    }

    def check(name: str, ok: bool | None, detail: object = "", required: bool = True) -> None:
        checks[name] = make_check(ok, detail, required=required)
        tag = checks[name]["status"].upper()
        print(f"[{tag}] {name}" + (f" -- {detail}" if detail != "" else ""))

    def flush() -> None:
        _write_json(report_path, report)

    def blocked_so_far() -> bool:
        return not freeze_decision(checks)["freeze_allowed"]

    def finish(status: str) -> dict:
        decision = freeze_decision(checks)
        report["decision"] = decision
        report["status"] = status
        report["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        report["experiments_entry"] = _experiments_entry(report, config, work, gpu)
        flush()
        print("=" * 70)
        print(f"STATUS: {status}")
        if decision["failing_required"]:
            print(f"failing required checks: {decision['failing_required']}")
        if decision["skipped_optional"]:
            print(f"skipped optional checks: {decision['skipped_optional']}")
        print(f"report: {report_path}")
        return report

    print(f"freeze run_id: {freeze_run_id} -> {work}")
    flush()

    # ---- 1. locks
    train_cfg, train_cfg_ok, train_cfg_sha = _load_pinned_yaml(
        repo_dir, config["source"]["train_config"], config["source"]["train_config_sha256"]
    )
    check("training config unchanged since the run", train_cfg_ok, train_cfg_sha[:16])
    lock_pairs = {
        "final_eval_sha256": train_cfg["locks"]["final_eval_sha256"],
        "d0_sha256": train_cfg["detector"]["sha256"],
    }
    same_locks = all(locks[k] == v for k, v in lock_pairs.items())
    check("freeze locks match the training config", same_locks, lock_pairs)

    with open(os.path.join(source_dir, REPORT_FILE), encoding="utf-8") as f:
        source_report = json.load(f)
    source_run_id = source_report["run_id"]
    report["source_run_id"] = source_run_id
    check(
        "source run is the configured run",
        source_run_id == config["source"]["run_id"]
        and os.path.basename(os.path.normpath(source_dir)) == source_run_id,
        source_run_id,
    )
    check(
        "source run awaits final-eval",
        source_report.get("status") == "trained_pending_final_eval",
        source_report.get("status"),
    )
    candidate = os.path.join(source_dir, CANDIDATE_DIR)
    d1_sha = file_sha256(os.path.join(candidate, "model.safetensors"))
    report["d1_model_sha256"] = d1_sha
    check(
        "D1 candidate sha256 matches the training report and the lock",
        d1_sha == source_report.get("d1_model_sha256") == locks["d1_model_sha256"],
        d1_sha[:16],
    )
    try:
        d0_sha = _verify_detector(train_cfg)
    except SystemExit as exc:
        d0_sha = None
        check("frozen D0 verified", False, exc)
    else:
        check("frozen D0 verified", d0_sha == locks["d0_sha256"], (d0_sha or "")[:16])

    splits_dir = os.path.join(repo_dir, "artifacts", "splits")
    if config.get("rebuild_locks"):
        _rebuild_splits(train_cfg, repo_dir, splits_dir)
    try:
        _final_eval_ids(train_cfg, splits_dir)
        fe_sha = file_sha256(os.path.join(splits_dir, "final_eval.jsonl"))
    except (SystemExit, FileNotFoundError) as exc:
        fe_sha = None
        check("final-eval partition lock reproduces", False, exc)
    else:
        check(
            "final-eval partition lock reproduces",
            fe_sha == locks["final_eval_sha256"],
            (fe_sha or "")[:16],
        )

    source_commit = source_report.get("environment", {}).get("repo_commit")
    report["source_repo_commit"] = source_commit
    check(
        "repo commits recorded (this run and the training run)",
        repo_commit_recorded(config.get("repo_commit")) and repo_commit_recorded(source_commit),
        f"freeze {config.get('repo_commit')}, training {source_commit}",
    )
    frozen_dir = frozen_dir_for(config["frozen_root"], source_run_id)
    report["frozen_dir"] = frozen_dir
    check(
        "frozen dir does not exist yet (never overwrite)",
        not os.path.exists(frozen_dir),
        frozen_dir,
    )
    flush()
    if blocked_so_far():
        print("A lock failed: the final-eval partition is NOT touched.")
        return finish("freeze_blocked")

    # ---- 2. final-eval: single use per candidate
    fe_report_path = os.path.join(source_dir, FINAL_EVAL_REPORT_FILE)
    if os.path.exists(fe_report_path):
        with open(fe_report_path, encoding="utf-8") as f:
            fe = json.load(f)
        reused = True
        print(f"final-eval already ran for this candidate; reusing {fe_report_path}")
    else:
        fe = run_d1_final_eval(dict(train_cfg, rebuild_locks=False), repo_dir, source_dir)
        reused = False
    report["final_eval"] = {
        "report": fe_report_path,
        "reused_existing_single_use_report": reused,
        "d0": fe["d0"],
        "d1": fe["d1"],
        "paired": fe["paired"],
    }
    check(
        "final-eval report is for this candidate and partition",
        fe["d1_model_sha256"] == d1_sha and fe["final_eval_sha256"] == locks["final_eval_sha256"],
        f"reused={reused}",
    )
    gates["final_eval_f1"] = final_eval_f1_gate(
        fe["d0"], fe["d1"], gates_cfg["final_eval_f1_max_drop"]
    )
    check(
        "final-eval span-F1 within max drop of D0",
        gates["final_eval_f1"]["ok"],
        gates["final_eval_f1"],
    )
    gates["final_eval_precision_recall"] = no_regression_gate(
        fe["d0"], fe["d1"], train_cfg["gates"]["final_eval_max_drop"]
    )
    check(
        "final-eval precision and recall drops within the pre-registered max (train config)",
        gates["final_eval_precision_recall"]["ok"],
        gates["final_eval_precision_recall"],
    )
    gates["precision_guard"] = precision_guard(
        fe["d0"], fe["d1"], gates_cfg["precision_guard"]["max_relative_neg_fp_rate_increase"]
    )
    check(
        "precision guard: name-free-row FP rate (PROPOSED threshold)",
        gates["precision_guard"]["ok"],
        gates["precision_guard"],
    )
    _print_final_eval(fe, gates["precision_guard"])
    flush()

    # ---- 3. optional short negatives
    neg_cfg = config["short_negatives"]
    resolved = resolve_optional_input(repo_dir, neg_cfg["path"], neg_cfg.get("sha256"))
    neg_gate = None
    if resolved["state"] == "ok":
        rows = load_negative_rows(os.path.join(repo_dir, neg_cfg["path"]))
        tok = AutoTokenizer.from_pretrained(candidate)
        bs = train_cfg["training"]["eval_batch_size"]
        max_length = train_cfg["detector"]["max_length"]
        preds = {}
        for name, path in (("d0", train_cfg["detector"]["path"]), ("d1", candidate)):
            model = _load_detector(path, device)
            preds[name] = predict_spans(model, tok, rows, max_length, bs, device)
            del model
        neg_section = short_negatives_report(
            rows,
            preds["d0"],
            preds["d1"],
            neg_cfg["gated_distractors"],
            neg_cfg["report_only_distractors"],
            neg_cfg["max_fp_row_rate"],
        )
        neg_gate = neg_section["gate"]
        report["short_negatives"] = {"input": resolved, **neg_section}
        _print_short_negatives(neg_section)
    else:
        report["short_negatives"] = {"input": resolved}
    if neg_gate is not None:
        gates["short_negatives"] = neg_gate
    check(
        "short negatives: D1 false-positive row rate (PROPOSED threshold)",
        **_unpack(short_negatives_check(resolved, neg_gate)),
    )
    flush()

    # ---- 4. M07a probe re-run against D1 (same table), D0 rescored on the same texts
    probe_cfg_section = config["probe"]
    m07a_cfg, m07a_ok, m07a_sha = _load_pinned_yaml(
        repo_dir, probe_cfg_section["config"], probe_cfg_section["config_sha256"]
    )
    check("M07a probe config unchanged (same go/no-go table)", m07a_ok, m07a_sha[:16])
    check(
        "M07a probe config scores with the same frozen D0",
        m07a_cfg["detector"]["sha256"] == locks["d0_sha256"],
        m07a_cfg["detector"]["sha256"][:16],
    )
    if blocked_so_far() and not probe_cfg_section.get("run_if_blocked", False):
        check(
            "M07a probe go/no-go passes against D1",
            None,
            "skipped: an earlier required check failed (probe.run_if_blocked is false)",
        )
        return _decide_and_freeze(report, checks, gates, config, locks, finish)

    probe_cfg = copy.deepcopy(m07a_cfg)
    probe_cfg["detector"] = dict(m07a_cfg["detector"], path=candidate, sha256=d1_sha)
    probe_cfg["run_id_prefix"] = probe_cfg_section["run_id_prefix"]
    probe_cfg["work_dir"] = config["work_dir"]
    probe_cfg["cache_dir"] = config["cache_dir"]
    probe_cfg["repo_commit"] = config.get("repo_commit")
    reuse_dir = probe_cfg_section.get("reuse_run_dir")
    if reuse_dir:
        with open(os.path.join(reuse_dir, PROBE_REPORT_FILE), encoding="utf-8") as f:
            d1_probe = json.load(f)
        reused_ok = (
            d1_probe["config"]["detector"]["sha256"] == d1_sha
            and d1_probe["config"]["thresholds"] == m07a_cfg["thresholds"]
            and d1_probe["config"]["generation"] == m07a_cfg["generation"]
        )
        check("reused probe run scored this D1 under the M07a protocol", reused_ok, reuse_dir)
        probe_dir = reuse_dir
    else:
        d1_probe = run_probe(probe_cfg, repo_dir)
        probe_dir = os.path.join(config["work_dir"], d1_probe["run_id"])
    probe_locks_ok = all(c["ok"] for c in d1_probe["checks"].values())
    check("probe run's own checks pass", probe_locks_ok, sorted(d1_probe["checks"]))
    with open(os.path.join(probe_dir, PROBE_SAMPLES_FILE), encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    d0_probe_cfg = dict(probe_cfg, detector=m07a_cfg["detector"])  # M07a's own D0 entry
    d0_per_temp = _rescore_probe_with_d0(samples, d0_probe_cfg, device)
    d1_per_temp = d1_probe["per_temperature"]
    band = m07a_cfg["thresholds"]["miss_rate_range"]
    gates["probe"] = probe_gate(d1_per_temp, band, probe_cfg_section["require_miss_rate_in_band"])
    report["probe"] = {
        "run_dir": probe_dir,
        "probe_run_id": d1_probe["run_id"],
        "d1_per_temperature": d1_per_temp,
        "d0_rescored_per_temperature": d0_per_temp,
        "d0_vs_d1": probe_comparison(d0_per_temp, d1_per_temp, band),
    }
    check("M07a probe go/no-go passes against D1", gates["probe"]["ok"], gates["probe"])
    _print_probe(report["probe"]["d0_vs_d1"], band)
    flush()

    return _decide_and_freeze(report, checks, gates, config, locks, finish)


def _unpack(record: dict) -> dict:
    return {"ok": record["ok"], "detail": record["detail"], "required": record["required"]}


def _decide_and_freeze(report, checks, gates, config, locks, finish) -> dict:
    """Step 5: copy D1 to its frozen dir only if every required check passed."""
    source_run_id = report["source_run_id"]
    frozen_dir = report["frozen_dir"]
    if not freeze_decision(checks)["freeze_allowed"]:
        return finish("freeze_blocked")
    manifest = freeze_candidate(
        os.path.join(report["source_run_dir"], CANDIDATE_DIR),
        frozen_dir,
        report["d1_model_sha256"],
        {
            "source_run_id": source_run_id,
            "source_dir": report["source_run_dir"],
            "freeze_run_id": report["run_id"],
            "repo_commit": config.get("repo_commit"),
            "source_repo_commit": report.get("source_repo_commit"),
            "locks": locks,
            "checks": checks,
            "gates": gates,
        },
    )
    report["frozen_manifest"] = {
        "path": os.path.join(frozen_dir, "freeze_manifest.json"),
        "files": manifest["files"],
    }
    print(f"D1 FROZEN -> {frozen_dir}")
    return finish("frozen")


def _print_final_eval(fe: dict, guard: dict) -> None:
    print("=" * 70)
    print("Final-eval partition, D0 vs D1")
    for name in ("d0", "d1"):
        m = fe[name]
        print(
            f"  {name}: P {m['span_precision']} R {m['span_recall']} F1 {m['span_f1']} "
            f"name-free rows flagged {m.get('neg_rows_with_pred')}/{m['neg_rows']} "
            f"(FP spans {m['neg_false_positives']}) extra-pred rows {m['rows_with_extra_pred']}"
        )
    print(
        f"  name-free FP row rate: D0 {guard['d0_neg_fp_row_rate']} -> "
        f"D1 {guard['d1_neg_fp_row_rate']} (allowed <= {guard['allowed_d1_rate']})"
    )


def _print_short_negatives(section: dict) -> None:
    print("=" * 70)
    print("Short negatives (name-free), rows flagged / rows, D0 -> D1")
    for dtype, res in section["by_distractor"].items():
        d0, d1 = res["d0"], res["d1"]
        tag = "gated" if dtype in section["gate"]["gated_types"] else "REPORT ONLY"
        print(
            f"  {dtype:12s} [{tag}]: {d0['rows_with_pred']}/{d0['rows']} "
            f"({d0['pred_spans']} spans) -> {d1['rows_with_pred']}/{d1['rows']} "
            f"({d1['pred_spans']} spans)"
        )
    for dtype in section["gate"]["report_only_types"]:
        for name in ("d0", "d1"):
            for f in section["by_distractor"][dtype][name]["flagged"]:
                print(f"    {dtype} {name} {f['id']}: {f['spans']} in {f['text'][:120]!r}")


def _print_probe(table: dict, band: list[float]) -> None:
    print("=" * 70)
    print(f"M07a probe, same texts, miss-rate band {band}")
    for temp, row in table.items():
        print(
            f"  t={temp}: D0 miss {row['d0_miss_rate']} (band {row['d0_in_band']}, "
            f"go {row['d0_go']}) | D1 miss {row['d1_miss_rate']} "
            f"(band {row['d1_in_band']}, go {row['d1_go']})"
        )


def _experiments_entry(report: dict, config: dict, work: str, gpu: str) -> dict:
    fe = report.get("final_eval") or {}
    metrics: dict = {"d1_model_sha256": report.get("d1_model_sha256")}
    if fe:
        metrics["final_eval"] = {
            name: {
                k: fe[name].get(k)
                for k in (
                    "span_precision",
                    "span_recall",
                    "span_f1",
                    "neg_rows",
                    "neg_rows_with_pred",
                    "neg_false_positives",
                )
            }
            for name in ("d0", "d1")
        }
    if "short_negatives" in report.get("gates", {}):
        g = report["gates"]["short_negatives"]
        metrics["short_negatives_fp_row_rate"] = {
            "gated_types": g["gated_types"],
            "d0": g["d0_fp_row_rate"],
            "d1": g["d1_fp_row_rate"],
        }
        by_type = report["short_negatives"]["by_distractor"]
        metrics["short_negatives_rows_flagged_by_distractor"] = {
            t: {
                "rows": res["d1"]["rows"],
                "d0": res["d0"]["rows_with_pred"],
                "d1": res["d1"]["rows_with_pred"],
                "gated": t in g["gated_types"],
            }
            for t, res in by_type.items()
        }
    if report.get("probe"):
        metrics["probe_miss_rate"] = {
            t: {"d0": r["d0_miss_rate"], "d1": r["d1_miss_rate"]}
            for t, r in report["probe"]["d0_vs_d1"].items()
        }
        metrics["probe_d1_go"] = {t: r["d1_go"] for t, r in report["probe"]["d0_vs_d1"].items()}
    decision = report.get("decision", {})
    if report["status"] == "frozen":
        outcome = f"D1 FROZEN at {report['frozen_dir']}"
    else:
        outcome = f"FREEZE BLOCKED: failing required checks {decision.get('failing_required')}"
    if decision.get("skipped_optional"):
        outcome += f"; skipped optional: {decision['skipped_optional']}"
    artifacts = f"{work} ({FREEZE_REPORT_FILE})"
    if report.get("probe"):
        artifacts += f"; probe {report['probe']['run_dir']}"
    if report["status"] == "frozen":
        artifacts += f"; frozen {report['frozen_dir']}"
    return {
        "run_id": report["run_id"],
        "kind": "detector-freeze",
        "milestone": "M07b.3/M07b.4",
        "executed": report["started_utc"][:10],
        "where": gpu,
        "repo_commit": config.get("repo_commit"),
        "source_run_id": report.get("source_run_id"),
        "artifacts": artifacts,
        "checks": check_counts(report["checks"]),
        "metrics": metrics,
        "outcome": outcome,
    }
