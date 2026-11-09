"""M07b.3/.4: final-eval no-regression, short negatives, re-probe, then freeze D1.

Order: locks + candidate choice -> single-use final-eval per candidate (D0 vs each) ->
optional short negatives -> M07a probe re-run against the chosen candidate (D0 rescored
on the same texts) -> freeze decision. With several saved epochs, the one that may be
frozen is fixed by the pre-registered dev-only rule BEFORE anything else is scored; the
others are scored for a side-by-side report only.
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
    candidate_table,
    check_counts,
    file_sha256,
    final_eval_f1_gate,
    freeze_candidate,
    freeze_decision,
    frozen_dir_for,
    make_check,
    plan_candidates,
    precision_guard,
    probe_comparison,
    probe_gate,
    repo_commit_recorded,
    resolve_optional_input,
    short_negatives_check,
    short_negatives_report,
)
from pii_redteam.overnight import find_reusable_probe_run
from pii_redteam.probe import _device_dtype, _finalize_samples, _score_with_d0, run_probe
from pii_redteam.probe_metrics import per_temperature_metrics
from pii_redteam.train_d1 import (
    REPORT_FILE,
    _final_eval_ids,
    _load_detector,
    _rebuild_splits,
    _verify_detector,
    _write_json,
    final_eval_report_name,
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


def _final_eval_gates(fe: dict, gates_cfg: dict, train_cfg: dict) -> dict:
    return {
        "final_eval_f1": final_eval_f1_gate(
            fe["d0"], fe["d1"], gates_cfg["final_eval_f1_max_drop"]
        ),
        "final_eval_precision_recall": no_regression_gate(
            fe["d0"], fe["d1"], train_cfg["gates"]["final_eval_max_drop"]
        ),
        "precision_guard": precision_guard(
            fe["d0"], fe["d1"], gates_cfg["precision_guard"]["max_relative_neg_fp_rate_increase"]
        ),
    }


def run_freeze_d1(config: dict, repo_dir: str) -> dict:
    started = datetime.datetime.now(datetime.UTC)
    freeze_run_id = f"{config['run_id_prefix']}-{started:%Y%m%d-%H%M%S}"
    source_dir = config["source"]["run_dir"]
    if not source_dir:
        raise SystemExit("source.run_dir is not set: pass --source-run-dir")
    work = os.path.join(config["work_dir"], freeze_run_id)
    os.makedirs(work, exist_ok=True)
    report_path = os.path.join(work, FREEZE_REPORT_FILE)
    locks = config["locks"]
    gates_cfg = config["gates"]
    device, gpu, _ = _device_dtype()
    checks: dict[str, dict] = {}
    gates: dict[str, dict] = {}
    results: dict[str, dict] = {}
    report: dict = {
        "run_id": freeze_run_id,
        "kind": "detector-freeze",
        "status": "running",
        "started_utc": started.isoformat(),
        "config": config,
        "source_run_dir": source_dir,
        "checks": checks,
        "gates": gates,
        "candidates": results,
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
        if "selection" in report:
            report["side_by_side"] = candidate_table(report["selection"], results)
        _write_json(report_path, report)

    def blocked_so_far() -> bool:
        return not freeze_decision(checks)["freeze_allowed"]

    def finish(status: str) -> dict:
        decision = freeze_decision(checks)
        report["decision"] = decision
        report["status"] = status
        report["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        flush()
        report["experiments_entry"] = _experiments_entry(report, config, work, gpu)
        flush()
        print("=" * 70)
        _print_side_by_side(report.get("side_by_side") or [])
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
    expected_run_id = config["source"].get("run_id")
    check(
        "source run is the configured run",
        (expected_run_id is None or source_run_id == expected_run_id)
        and os.path.basename(os.path.normpath(source_dir)) == source_run_id,
        f"{source_run_id} (configured {expected_run_id or 'via --source-run-dir'})",
    )
    check(
        "source run awaits final-eval",
        source_report.get("status") == "trained_pending_final_eval",
        source_report.get("status"),
    )

    # Which candidate may be frozen is fixed HERE, from the training report alone.
    plan = plan_candidates(source_report, train_cfg)
    report["selection"] = plan
    chosen = plan["chosen"]
    report["chosen_candidate"] = chosen
    check(
        "freeze rule chose a candidate (pre-registered, dev only)",
        chosen is not None,
        f"rule {plan['rule']}, chosen {chosen}",
    )
    if plan["selection"] is not None:
        recorded = (source_report.get("freeze_selection") or {}).get("chosen_epoch")
        check(
            "freeze rule reproduces the training run's recorded choice",
            recorded == plan["selection"]["chosen_epoch"],
            f"training {recorded}, recomputed {plan['selection']['chosen_epoch']}",
        )
    pins = locks.get("d1_candidate_sha256")
    sha_problems = []
    for cand in plan["candidates"]:
        path = os.path.join(source_dir, cand["name"], "model.safetensors")
        actual = file_sha256(path) if os.path.exists(path) else None
        if actual != cand["sha256"]:
            sha_problems.append(f"{cand['name']}: file {actual} != training report")
        if pins is not None and pins.get(cand["name"]) != actual:
            sha_problems.append(f"{cand['name']}: file {actual} != config pin")
    if pins is not None and set(pins) != {c["name"] for c in plan["candidates"]}:
        sha_problems.append(f"config pins {sorted(pins)} != candidates")
    check(
        "candidate sha256s match the training report"
        + (" and the config pins" if pins is not None else " (not pinned in config)"),
        not sha_problems,
        sha_problems or {c["name"]: (c["sha256"] or "")[:16] for c in plan["candidates"]},
    )
    report["d1_model_sha256"] = next(
        (c["sha256"] for c in plan["candidates"] if c["name"] == chosen), None
    )
    try:
        d0_sha = _verify_detector(train_cfg)
    except SystemExit as exc:
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

    # Chosen first; the others are scored for the side-by-side report only.
    ordered = [c for c in plan["candidates"] if c["name"] == chosen] + [
        c for c in plan["candidates"] if c["name"] != chosen
    ]
    for cand in ordered:
        results[cand["name"]] = {
            "epoch": cand["epoch"],
            "sha256": cand["sha256"],
            "chosen": cand["name"] == chosen,
        }

    # ---- 2. final-eval: single use per candidate
    for cand in ordered:
        name = cand["name"]
        fe_report_path = os.path.join(source_dir, final_eval_report_name(name))
        if os.path.exists(fe_report_path):
            with open(fe_report_path, encoding="utf-8") as f:
                fe = json.load(f)
            reused = True
            print(f"final-eval already ran for {name}; reusing {fe_report_path}")
        else:
            fe = run_d1_final_eval(dict(train_cfg, rebuild_locks=False), repo_dir, source_dir, name)
            reused = False
        cand_gates = _final_eval_gates(fe, gates_cfg, train_cfg)
        matches = (
            fe["d1_model_sha256"] == cand["sha256"]
            and fe["final_eval_sha256"] == locks["final_eval_sha256"]
        )
        results[name]["final_eval"] = {
            "report": fe_report_path,
            "reused_existing_single_use_report": reused,
            "report_matches_candidate_and_partition": matches,
            "d0": fe["d0"],
            "d1": fe["d1"],
            "paired": fe["paired"],
        }
        results[name]["gates"] = cand_gates
        _print_final_eval(name, fe, cand_gates["precision_guard"])
        if name != chosen:
            continue
        report["final_eval"] = results[name]["final_eval"]
        gates.update(cand_gates)
        check(
            "final-eval report is for the chosen candidate and partition",
            matches,
            f"{name}, reused={reused}",
        )
        check(
            "final-eval span-F1 within max drop of D0",
            cand_gates["final_eval_f1"]["ok"],
            cand_gates["final_eval_f1"],
        )
        check(
            "final-eval precision and recall drops within the pre-registered max (train config)",
            cand_gates["final_eval_precision_recall"]["ok"],
            cand_gates["final_eval_precision_recall"],
        )
        check(
            "precision guard: name-free-row FP rate (owner decision 2026-11-09)",
            cand_gates["precision_guard"]["ok"],
            cand_gates["precision_guard"],
        )
        flush()
    flush()

    # ---- 3. optional short negatives (every candidate; the gate uses the chosen one)
    neg_cfg = config["short_negatives"]
    resolved = resolve_optional_input(repo_dir, neg_cfg["path"], neg_cfg.get("sha256"))
    neg_gate = None
    report["short_negatives"] = {"input": resolved}
    if resolved["state"] == "ok":
        rows = load_negative_rows(os.path.join(repo_dir, neg_cfg["path"]))
        tok = AutoTokenizer.from_pretrained(os.path.join(source_dir, chosen))
        bs = train_cfg["training"]["eval_batch_size"]
        max_length = train_cfg["detector"]["max_length"]
        model = _load_detector(train_cfg["detector"]["path"], device)
        d0_preds = predict_spans(model, tok, rows, max_length, bs, device)
        del model
        for cand in ordered:
            model = _load_detector(os.path.join(source_dir, cand["name"]), device)
            preds = predict_spans(model, tok, rows, max_length, bs, device)
            del model
            section = short_negatives_report(
                rows,
                d0_preds,
                preds,
                neg_cfg["gated_distractors"],
                neg_cfg["report_only_distractors"],
                neg_cfg["max_fp_row_rate"],
            )
            results[cand["name"]]["short_negatives"] = section
            _print_short_negatives(cand["name"], section)
            if cand["name"] == chosen:
                neg_gate = section["gate"]
                report["short_negatives"] = {"input": resolved, **section}
    if neg_gate is not None:
        gates["short_negatives"] = neg_gate
    check(
        "short negatives: D1 false-positive row rate (PROPOSED threshold)",
        **_unpack(short_negatives_check(resolved, neg_gate)),
    )
    flush()

    # ---- 4. M07a probe re-run against the chosen candidate (others only if configured)
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

    probe_all = probe_cfg_section.get("probe_all_candidates", False)
    to_probe = ordered if probe_all else ordered[:1]
    for cand in to_probe:
        is_chosen = cand["name"] == chosen
        section, gate, own_checks_ok, reuse_ok = _probe_candidate(
            config,
            repo_dir,
            m07a_cfg,
            os.path.join(source_dir, cand["name"]),
            cand["sha256"],
            probe_cfg_section.get("reuse_run_dir") if is_chosen else None,
            device,
        )
        results[cand["name"]]["probe"] = section
        results[cand["name"]]["probe_gate"] = gate
        _print_probe(cand["name"], section["d0_vs_d1"], gate["band"])
        if not is_chosen:
            flush()
            continue
        if reuse_ok is not None:
            check(
                "reused probe run scored this D1 under the M07a protocol",
                reuse_ok,
                section["run_dir"],
            )
        check("probe run's own checks pass", own_checks_ok, section["probe_run_id"])
        report["probe"] = section
        gates["probe"] = gate
        check("M07a probe go/no-go passes against D1", gate["ok"], gate)
        flush()
    for cand in ordered:
        results[cand["name"]].setdefault("probe", "not run (probe.probe_all_candidates is false)")
    flush()

    return _decide_and_freeze(report, checks, gates, config, locks, finish)


def _probe_candidate(config, repo_dir, m07a_cfg, cand_dir, sha, reuse_dir, device):
    """Run (or reuse) the unchanged M07a probe with one candidate as the scorer."""
    probe_cfg_section = config["probe"]
    probe_cfg = copy.deepcopy(m07a_cfg)
    probe_cfg["detector"] = dict(m07a_cfg["detector"], path=cand_dir, sha256=sha)
    probe_cfg["run_id_prefix"] = probe_cfg_section["run_id_prefix"]
    probe_cfg["work_dir"] = config["work_dir"]
    probe_cfg["cache_dir"] = config["cache_dir"]
    probe_cfg["repo_commit"] = config.get("repo_commit")
    reuse_ok = None
    if reuse_dir == "auto":
        # Resume after a disconnect: a FINISHED probe run for this sha is reused; a
        # partial one (samples.partial.jsonl only) cannot be resumed and is regenerated.
        reuse_dir = find_reusable_probe_run(
            config["work_dir"], probe_cfg_section["run_id_prefix"], sha, m07a_cfg
        )
        print(f"probe reuse (auto): {reuse_dir or 'none finished for this sha, running it'}")
    if reuse_dir:
        with open(os.path.join(reuse_dir, PROBE_REPORT_FILE), encoding="utf-8") as f:
            d1_probe = json.load(f)
        reuse_ok = (
            d1_probe["config"]["detector"]["sha256"] == sha
            and d1_probe["config"]["thresholds"] == m07a_cfg["thresholds"]
            and d1_probe["config"]["generation"] == m07a_cfg["generation"]
        )
        probe_dir = reuse_dir
    else:
        d1_probe = run_probe(probe_cfg, repo_dir)
        probe_dir = os.path.join(config["work_dir"], d1_probe["run_id"])
    own_checks_ok = all(c["ok"] for c in d1_probe["checks"].values())
    with open(os.path.join(probe_dir, PROBE_SAMPLES_FILE), encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    d0_probe_cfg = dict(probe_cfg, detector=m07a_cfg["detector"])  # M07a's own D0 entry
    d0_per_temp = _rescore_probe_with_d0(samples, d0_probe_cfg, device)
    d1_per_temp = d1_probe["per_temperature"]
    band = m07a_cfg["thresholds"]["miss_rate_range"]
    gate = probe_gate(d1_per_temp, band, probe_cfg_section["require_miss_rate_in_band"])
    section = {
        "run_dir": probe_dir,
        "probe_run_id": d1_probe["run_id"],
        "d1_per_temperature": d1_per_temp,
        "d0_rescored_per_temperature": d0_per_temp,
        "d0_vs_d1": probe_comparison(d0_per_temp, d1_per_temp, band),
    }
    return section, gate, own_checks_ok, reuse_ok


def _unpack(record: dict) -> dict:
    return {"ok": record["ok"], "detail": record["detail"], "required": record["required"]}


def _decide_and_freeze(report, checks, gates, config, locks, finish) -> dict:
    """Step 5: copy D1 to its frozen dir only if every required check passed."""
    source_run_id = report["source_run_id"]
    frozen_dir = report["frozen_dir"]
    if not freeze_decision(checks)["freeze_allowed"]:
        return finish("freeze_blocked")
    chosen = report["chosen_candidate"]
    manifest = freeze_candidate(
        os.path.join(report["source_run_dir"], chosen),
        frozen_dir,
        report["d1_model_sha256"],
        {
            "source_run_id": source_run_id,
            "source_dir": report["source_run_dir"],
            "source_candidate": chosen,
            "selection": report["selection"],
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
    print(f"D1 FROZEN ({chosen}) -> {frozen_dir}")
    return finish("frozen")


def _print_final_eval(name: str, fe: dict, guard: dict) -> None:
    print("=" * 70)
    print(f"Final-eval partition, D0 vs {name}")
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


def _print_short_negatives(name: str, section: dict) -> None:
    print("=" * 70)
    print(f"Short negatives (name-free), rows flagged / rows, D0 -> {name}")
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


def _print_probe(name: str, table: dict, band: list[float]) -> None:
    print("=" * 70)
    print(f"M07a probe scored by {name}, D0 rescored on the same texts, band {band}")
    for temp, row in table.items():
        print(
            f"  t={temp}: D0 miss {row['d0_miss_rate']} (band {row['d0_in_band']}, "
            f"go {row['d0_go']}) | D1 miss {row['d1_miss_rate']} "
            f"(band {row['d1_in_band']}, go {row['d1_go']})"
        )


def _print_side_by_side(rows: list[dict]) -> None:
    if not rows:
        return
    print("Candidates side by side (the chosen one is fixed by the dev-only rule)")
    for r in rows:
        mark = "CHOSEN" if r["chosen"] else "      "
        print(
            f"  {mark} {r['name']} (epoch {r['epoch']}): final-eval F1 {r['final_eval_f1']} "
            f"P {r['final_eval_precision']} name-free rows flagged "
            f"{r['final_eval_neg_rows_with_pred']} | gates {r['gates_ok']} | short-neg "
            f"{r['short_negatives_gated_fp_row_rate']} | probe {r['probe_ok']}"
        )


def _experiments_entry(report: dict, config: dict, work: str, gpu: str) -> dict:
    fe = report.get("final_eval") or {}
    metrics: dict = {
        "d1_model_sha256": report.get("d1_model_sha256"),
        "chosen_candidate": report.get("chosen_candidate"),
        "selection_rule": (report.get("selection") or {}).get("rule"),
    }
    if report.get("side_by_side"):
        metrics["candidates"] = report["side_by_side"]
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
        outcome = f"D1 FROZEN ({report['chosen_candidate']}) at {report['frozen_dir']}"
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
