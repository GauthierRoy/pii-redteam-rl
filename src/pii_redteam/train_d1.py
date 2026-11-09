"""M07b.2/.3: fine-tune frozen D0 into candidate D1 and report D0 vs D1.

`run_train_d1` trains, selects on development data only, then scores the short_eval
slices. `run_d1_final_eval` is a separate, single-use step for the final-eval
no-regression check. Freezing D1 stays an owner decision.
"""

from __future__ import annotations

import datetime
import json
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer

from pii_redteam.d1_metrics import (
    FREEZE_RULE,
    is_better_candidate,
    load_synthetic_rows,
    no_regression_gate,
    paired_changes,
    select_freeze_epoch,
    short_eval_gate,
    span_metrics,
    split_by_name,
    strip_per_row,
)
from pii_redteam.data import QuarantinedRecord, to_canonical
from pii_redteam.detector import bio_labels_from_spans, spans_from_bio_labels, trim_span_whitespace
from pii_redteam.probe import _device_dtype, _sha256

LABEL2ID = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
TRAIN_FILE = "1english_openpii_30k.jsonl"
DEV_FILE = "1english_openpii_8k.jsonl"
CANDIDATE_DIR = "d1-candidate"
REPORT_FILE = "d1_report.json"
FINAL_EVAL_REPORT_FILE = "d1_final_eval_report.json"


def epoch_dir_name(epoch: int) -> str:
    return f"d1-epoch{epoch}"


def final_eval_report_name(candidate: str) -> str:
    """One single-use final-eval report per candidate directory."""
    if candidate == CANDIDATE_DIR:
        return FINAL_EVAL_REPORT_FILE
    return f"d1_final_eval_report.{candidate}.json"


def _save_checkpoint(model, tok, target: str) -> None:
    tmp = target + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    tok.save_pretrained(tmp)
    shutil.rmtree(target, ignore_errors=True)
    os.replace(tmp, target)


def _write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _load_detector(path: str, device: str):
    # fp32 weights: required by GradScaler, and scoring matches the D0/probe pipeline.
    model = AutoModelForTokenClassification.from_pretrained(
        path, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID, dtype=torch.float32
    )
    return model.to(device)


@torch.no_grad()
def predict_spans(model, tok, rows, max_length, batch_size, device) -> list[list[tuple[int, int]]]:
    """Decoded PERSON spans: the same fp32, argmax, BIO and trim path as D0's evaluation."""
    model.eval()
    preds: list[list[tuple[int, int]]] = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        enc = tok(
            [r["text"] for r in chunk],
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        logits = model(
            input_ids=torch.tensor(enc["input_ids"]).to(device),
            attention_mask=torch.tensor(enc["attention_mask"]).to(device),
        ).logits
        for j, row in enumerate(chunk):
            labels = [ID2LABEL[int(p)] for p in logits[j].argmax(-1).tolist()]
            decoded = spans_from_bio_labels(enc["offset_mapping"][j], labels)
            preds.append(
                [(sp["start"], sp["end"]) for sp in trim_span_whitespace(row["text"], decoded)]
            )
    return preds


def encode_rows(tok, rows, max_length) -> tuple[list[dict], int]:
    """Tokenize in this process, so the count of spans lost to truncation is accurate."""
    enc = tok(
        [r["text"] for r in rows],
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    encoded, lost = [], 0
    for i, row in enumerate(rows):
        offsets = [tuple(o) for o in enc["offset_mapping"][i]]
        labels = bio_labels_from_spans(offsets, row["spans"])
        encoded.append(
            {
                "input_ids": enc["input_ids"][i],
                "labels": [
                    -100 if off == (0, 0) else LABEL2ID[lab]
                    for off, lab in zip(offsets, labels, strict=True)
                ],
            }
        )
        visible_end = max((end for _, end in offsets), default=0)
        lost += sum(sp["end"] > visible_end for sp in row["spans"])
    return encoded, lost


def _collate(batch, pad_id, device) -> dict:
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attention, labels = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        attention.append([1] * len(b["input_ids"]) + [0] * n)
        labels.append(b["labels"] + [-100] * n)
    return {
        "input_ids": torch.tensor(input_ids).to(device),
        "attention_mask": torch.tensor(attention).to(device),
        "labels": torch.tensor(labels).to(device),
    }


def _rebuild_splits(config: dict, repo_dir: str, splits_dir: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/carve_splits.py",
            "--cache-dir",
            config["cache_dir"],
            "--out",
            splits_dir,
        ],
        cwd=repo_dir,
        env=dict(os.environ, PYTHONPATH=os.path.join(repo_dir, "src")),
        check=True,
    )


def _final_eval_ids(config: dict, splits_dir: str) -> set[str]:
    with open(os.path.join(splits_dir, "final_eval_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest["final_eval_sha256"] != config["locks"]["final_eval_sha256"]:
        raise SystemExit(f"final-eval lock mismatch: {manifest['final_eval_sha256']}")
    return {r["example_id"] for r in manifest["rows"]}


def _load_ai4privacy(path: str, excluded_ids: set[str]) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = to_canonical(json.loads(line))
            except QuarantinedRecord:
                continue
            if row["example_id"] not in excluded_ids:
                rows.append(row)
    return rows


def _locked_path(repo_dir: str, candidates: dict[str, str], label: str) -> str:
    """First existing candidate path, which must have its recorded sha256."""
    for rel, sha in candidates.items():
        path = os.path.join(repo_dir, rel)
        if os.path.exists(path):
            actual = _sha256(path)
            if actual != sha:
                raise SystemExit(f"{label} lock mismatch at {rel}: {actual}")
            return path
    raise SystemExit(f"{label}: none of {sorted(candidates)} exist")


def _verify_detector(config: dict) -> str | None:
    path, expected = config["detector"]["path"], config["detector"]["sha256"]
    if expected is None:
        if not config.get("limits"):
            raise SystemExit("detector.sha256 may only be null in a limited smoke config.")
        return None
    actual = _sha256(os.path.join(path, "model.safetensors"))
    if actual != expected:
        raise SystemExit(f"D0 lock mismatch: {actual}")
    return actual


def run_train_d1(config: dict, repo_dir: str) -> dict:
    started = datetime.datetime.now(datetime.UTC)
    run_id = f"{config['run_id_prefix']}-{started:%Y%m%d-%H%M%S}"
    work = os.path.join(config["work_dir"], run_id)
    os.makedirs(work, exist_ok=True)
    report_path = os.path.join(work, REPORT_FILE)
    train_cfg = config["training"]
    max_length = config["detector"]["max_length"]
    limits = config.get("limits") or {}
    seed = config["seed"]
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device, gpu, dtype = _device_dtype()
    checks: dict[str, dict] = {}
    report: dict = {
        "run_id": run_id,
        "status": "running",
        "started_utc": started.isoformat(),
        "config": config,
        "checks": checks,
        "environment": {
            "repo_commit": config.get("repo_commit"),
            "device": gpu,
            "train_dtype": str(dtype),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": sys.version.split()[0],
        },
    }

    def check(name: str, ok: bool, detail: object = "") -> None:
        checks[name] = {"ok": bool(ok), "detail": str(detail)}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail != "" else ""))

    def flush() -> None:
        _write_json(report_path, report)

    print(f"run_id: {run_id} -> {work}")
    flush()

    # ---- locks
    splits_dir = os.path.join(repo_dir, "artifacts", "splits")
    if config.get("rebuild_locks"):
        _rebuild_splits(config, repo_dir, splits_dir)
    final_eval_ids = _final_eval_ids(config, splits_dir)
    check("final-eval lock reproduces", True, config["locks"]["final_eval_sha256"][:16])
    d0_sha = _verify_detector(config)
    check("frozen D0 verified", d0_sha is not None or bool(limits), d0_sha or "smoke: no D0 lock")
    aug_path = _locked_path(repo_dir, config["data"]["augmentation"], "d1_aug_v1")
    slice_paths = {
        name: _locked_path(repo_dir, candidates, name)
        for name, candidates in config["data"]["short_eval"].items()
    }
    check("augmentation and short_eval hashes verified", True, sorted(slice_paths))

    # ---- data
    excluded_aug = set(config["data"].get("aug_exclude_ids") or [])
    aug_rows = load_synthetic_rows(aug_path)
    missing_exclusions = excluded_aug - {r["example_id"] for r in aug_rows}
    if missing_exclusions:
        raise SystemExit(f"aug_exclude_ids not in the data: {sorted(missing_exclusions)}")
    aug_rows = [r for r in aug_rows if r["example_id"] not in excluded_aug]
    aug_train, aug_dev = split_by_name(aug_rows, config["data"]["aug_dev_fraction"], seed)
    replay_pool = _load_ai4privacy(os.path.join(config["cache_dir"], TRAIN_FILE), final_eval_ids)
    dev_rows = _load_ai4privacy(os.path.join(config["cache_dir"], DEV_FILE), final_eval_ids)
    expected = config["data"].get("expected_rows")
    if expected:
        counts_ok = len(replay_pool) == expected["train_side"] and (
            len(dev_rows) == expected["dev_calibration"]
        )
        check("AI4Privacy rows match D0's", counts_ok, f"{len(replay_pool)} / {len(dev_rows)}")
        if not counts_ok:
            raise SystemExit("AI4Privacy row counts differ from the D0 run.")
    slices = {name: load_synthetic_rows(path) for name, path in slice_paths.items()}

    aug_names = {r["name"].casefold() for r in aug_rows}
    aug_texts = {r["text"] for r in aug_rows}
    eval_rows = [r for rows in slices.values() for r in rows]
    name_leaks = sorted({r["name"] for r in eval_rows if r["name"].casefold() in aug_names})
    text_leaks = [r["example_id"] for r in eval_rows if r["text"] in aug_texts]
    leak_free = not name_leaks and not text_leaks
    check(
        "short_eval disjoint from augmentation",
        leak_free,
        f"name overlaps {name_leaks[:5]}, text overlaps {text_leaks[:5]}",
    )
    if not leak_free:
        flush()
        raise SystemExit("short_eval overlaps the augmentation data; refusing to train.")

    if limits:
        print(f"SMOKE limits: {limits}")
        aug_train = aug_train[: limits["aug_train"]]
        aug_dev = aug_dev[: limits["aug_dev"]]
        replay_pool = replay_pool[: limits["replay_pool"]]
        dev_rows = dev_rows[: limits["dev"]]
        slices = {name: rows[: limits["short_eval"]] for name, rows in slices.items()}
    replay_per_epoch = min(
        round(len(aug_train) * train_cfg["replay_per_aug_row"]), len(replay_pool)
    )
    report["data"] = {
        "aug_excluded": sorted(excluded_aug),
        "aug_train": len(aug_train),
        "aug_dev": len(aug_dev),
        "replay_pool": len(replay_pool),
        "replay_per_epoch": replay_per_epoch,
        "dev_calibration": len(dev_rows),
        "short_eval": {name: len(rows) for name, rows in slices.items()},
    }
    print(json.dumps(report["data"], indent=2))

    # ---- baseline: D0 on the selection sets, before any update
    tok = AutoTokenizer.from_pretrained(config["detector"]["path"])
    model = _load_detector(config["detector"]["path"], device)
    eval_bs = train_cfg["eval_batch_size"]

    def score(detector, rows) -> dict:
        return span_metrics(rows, predict_spans(detector, tok, rows, max_length, eval_bs, device))

    d0_dev = score(model, dev_rows)
    d0_aug_dev = score(model, aug_dev)
    report["d0_baseline"] = {
        "dev_calibration": strip_per_row(d0_dev),
        "aug_dev": strip_per_row(d0_aug_dev),
    }
    expected_f1 = config["detector"].get("expected_dev_f1")
    if expected_f1 is not None:
        reproduced = abs(d0_dev["span_f1"] - expected_f1) <= 0.001
        check("D0 dev span-F1 reproduces", reproduced, f"{d0_dev['span_f1']} vs {expected_f1}")
        if not reproduced:
            flush()
            raise SystemExit("The evaluation pipeline does not reproduce D0's recorded dev F1.")
    print(
        f"D0 baseline: dev F1 {d0_dev['span_f1']}, aug-dev F1 {d0_aug_dev['span_f1']}, "
        f"aug-dev miss {d0_aug_dev['complete_miss_rate']}"
    )
    flush()

    # ---- fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp and dtype == torch.float16)
    pad_id = tok.pad_token_id or 0  # ty: ignore[unresolved-attribute]  # stub union
    encoded_aug, _ = encode_rows(tok, aug_train, max_length)
    best_aug_f1: float | None = None
    candidate = os.path.join(work, CANDIDATE_DIR)
    curve: list[dict] = []
    report["curve"] = curve
    report["best_epoch"] = None
    save_epochs = sorted(set(train_cfg.get("save_epochs") or []))
    if any(not 1 <= e <= train_cfg["epochs"] for e in save_epochs):
        raise SystemExit(f"save_epochs {save_epochs} outside 1..{train_cfg['epochs']}")
    epoch_checkpoints: dict[str, dict] = {}
    if save_epochs:
        report["epoch_checkpoints"] = epoch_checkpoints
    lost_spans = 0
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for epoch in range(1, train_cfg["epochs"] + 1):
        epoch_start = time.time()
        encoded_replay, lost = encode_rows(
            tok, rng.sample(replay_pool, replay_per_epoch), max_length
        )
        lost_spans += lost
        epoch_rows = encoded_aug + encoded_replay
        rng.shuffle(epoch_rows)
        model.train()
        losses = []
        for i in range(0, len(epoch_rows), train_cfg["batch_size"]):
            batch = _collate(epoch_rows[i : i + train_cfg["batch_size"]], pad_id, device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            if not np.isfinite(losses[-1]):
                report["status"] = "non_finite_loss"
                flush()
                raise SystemExit(f"Non-finite loss at epoch {epoch}, batch offset {i}, {dtype}.")
        train_seconds = time.time() - epoch_start
        dev = score(model, dev_rows)
        aug = score(model, aug_dev)
        improved = is_better_candidate(
            dev["span_f1"],
            aug["span_f1"],
            best_aug_f1,
            d0_dev["span_f1"],
            config["selection"]["dev_f1_max_drop"],
        )
        if improved:
            best_aug_f1 = aug["span_f1"]
            _save_checkpoint(model, tok, candidate)
            report["best_epoch"] = epoch
        if epoch in save_epochs:
            name = epoch_dir_name(epoch)
            _save_checkpoint(model, tok, os.path.join(work, name))
            epoch_checkpoints[str(epoch)] = {
                "epoch": epoch,
                "dir": name,
                "sha256": _sha256(os.path.join(work, name, "model.safetensors")),
                "dev_span_f1": dev["span_f1"],
                "dev_span_precision": dev["span_precision"],
                "dev_neg_rows": dev["neg_rows"],
                "dev_neg_rows_with_pred": dev["neg_rows_with_pred"],
                "dev_neg_false_positives": dev["neg_false_positives"],
                "aug_dev_span_f1": aug["span_f1"],
            }
        curve.append(
            {
                "epoch": epoch,
                "steps": len(losses),
                "mean_loss": round(sum(losses) / len(losses), 4),
                "train_seconds": round(train_seconds, 1),
                "seconds_including_eval": round(time.time() - epoch_start, 1),
                "dev_calibration": strip_per_row(dev),
                "aug_dev": strip_per_row(aug),
                "selected_so_far": improved,
            }
        )
        print(
            f"epoch {epoch}: loss {curve[-1]['mean_loss']}, dev F1 {dev['span_f1']}, "
            f"aug-dev F1 {aug['span_f1']}, aug-dev miss {aug['complete_miss_rate']}, "
            f"{curve[-1]['seconds_including_eval']:.0f}s" + (" [saved]" if improved else "")
        )
        flush()
    report["training"] = {
        "total_seconds": round(time.time() - t_start, 1),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if use_amp else None,
        "replay_spans_lost_to_truncation": lost_spans,
    }
    if save_epochs:
        _record_freeze_selection(report, config, d0_dev["span_f1"], check)
    has_candidate = report["best_epoch"] is not None
    check(
        "an epoch stayed within the dev F1 floor",
        has_candidate,
        f"best epoch {report['best_epoch']}, floor D0 - {config['selection']['dev_f1_max_drop']}",
    )
    if not has_candidate:
        report["status"] = "no_candidate"
        flush()
        print("No epoch met the dev F1 floor. short_eval was not scored and nothing was saved.")
        return report

    # ---- short_eval: only now, from the saved artifact, after selection is final
    del model, optimizer
    if use_amp:
        torch.cuda.empty_cache()
    report["d1_model_sha256"] = _sha256(os.path.join(candidate, "model.safetensors"))
    d1 = _load_detector(candidate, device)
    d1_scores = {name: score(d1, rows) for name, rows in slices.items()}
    del d1
    d0 = _load_detector(config["detector"]["path"], device)
    d0_scores = {name: score(d0, rows) for name, rows in slices.items()}
    del d0
    report["short_eval"] = {
        name: {
            "d0": strip_per_row(d0_scores[name]),
            "d1": strip_per_row(d1_scores[name]),
            "paired": paired_changes(d0_scores[name], d1_scores[name]),
        }
        for name in slices
    }
    primary = config["gates"]["primary_short_eval_slice"]
    gate = short_eval_gate(
        d0_scores[primary], d1_scores[primary], config["gates"]["short_eval_recall_gain_min"]
    )
    report["short_eval_gate"] = {"slice": primary, **gate}
    check(f"short_eval recall gain on {primary}", gate["ok"], gate)
    if epoch_checkpoints:
        # Every saved epoch on short_eval too, side by side (reported, never selected on).
        by_epoch = {}
        for key, ckpt in epoch_checkpoints.items():
            model_e = _load_detector(os.path.join(work, ckpt["dir"]), device)
            by_epoch[key] = {}
            for name, rows in slices.items():
                scored = score(model_e, rows)
                by_epoch[key][name] = {
                    "d1": strip_per_row(scored),
                    "paired": paired_changes(d0_scores[name], scored),
                }
            del model_e
        report["short_eval_by_epoch"] = by_epoch
    report["status"] = "trained_pending_final_eval"
    report["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    report["experiments_entry"] = {
        "run_id": run_id,
        "kind": "detector-training",
        "milestone": "M07b.2",
        "executed": started.date().isoformat(),
        "where": gpu,
        "repo_commit": config.get("repo_commit"),
        "artifacts": f"{work} ({REPORT_FILE}, {CANDIDATE_DIR}/)",
        "metrics": {
            "best_epoch": report["best_epoch"],
            "d1_model_sha256": report["d1_model_sha256"],
            "short_eval_recall": {
                name: {
                    "d0": d0_scores[name]["span_recall"],
                    "d1": d1_scores[name]["span_recall"],
                }
                for name in slices
            },
        },
    }
    if epoch_checkpoints:
        report["experiments_entry"]["metrics"]["epoch_checkpoints"] = {
            key: {
                k: ckpt[k]
                for k in ("sha256", "dev_span_f1", "dev_neg_rows_with_pred", "dev_span_precision")
            }
            for key, ckpt in epoch_checkpoints.items()
        }
        report["experiments_entry"]["metrics"]["freeze_selection_chosen_epoch"] = report[
            "freeze_selection"
        ]["chosen_epoch"]
        if "reproduction" in report:
            report["experiments_entry"]["metrics"]["reproduction"] = report["reproduction"]
    flush()

    print("=" * 70)
    print("D0 vs D1 on short_eval (selection used development data only)")
    for name, res in report["short_eval"].items():
        for model_name in ("d0", "d1"):
            m = res[model_name]
            print(
                f"  {name} {model_name}: P {m['span_precision']} R {m['span_recall']} "
                f"F1 {m['span_f1']} miss {m['complete_miss_rate']} "
                f"extra-pred rows {m['rows_with_extra_pred']}/{m['rows']}"
            )
        paired = res["paired"]
        print(f"  {name} paired: +{paired['gained_exact']} / -{paired['lost_exact']} exact hits")
    print(f"candidate: {candidate}  sha256 {report['d1_model_sha256']}")
    print(f"report:    {report_path}")
    if epoch_checkpoints:
        for key, ckpt in epoch_checkpoints.items():
            print(
                f"saved epoch {key}: {ckpt['dir']} sha256 {ckpt['sha256'][:16]} dev F1 "
                f"{ckpt['dev_span_f1']} name-free dev rows flagged "
                f"{ckpt['dev_neg_rows_with_pred']}/{ckpt['dev_neg_rows']}"
            )
        print(
            f"freeze rule ({FREEZE_RULE}, dev only) chose epoch "
            f"{report['freeze_selection']['chosen_epoch']}"
        )
    print("Final-eval NOT run. D1 is NOT frozen: run freeze-d1 for that.")
    return report


def _record_freeze_selection(report: dict, config: dict, d0_dev_f1: float, check) -> None:
    """Apply the pre-registered dev-only freeze rule and the reproduction note."""
    rule_cfg = config["selection"]["freeze_rule"]
    if rule_cfg["rule"] != FREEZE_RULE:
        raise SystemExit(f"unknown freeze rule {rule_cfg['rule']!r}")
    selection = select_freeze_epoch(
        list(report["epoch_checkpoints"].values()), d0_dev_f1, rule_cfg["dev_f1_max_drop"]
    )
    report["freeze_selection"] = selection
    check(
        "freeze rule found an eligible saved epoch (dev only)",
        selection["chosen_epoch"] is not None,
        f"chosen {selection['chosen_epoch']}, eligible {selection['eligible_epochs']}",
    )
    ref = config.get("reference")
    if ref:
        new = report["epoch_checkpoints"].get(str(ref["epoch"]), {}).get("sha256")
        same = new == ref["sha256"]
        report["reproduction"] = {
            "reference_run_id": ref["run_id"],
            "epoch": ref["epoch"],
            "reference_sha256": ref["sha256"],
            "new_sha256": new,
            "reproduced": same,
            "note": "bit-identical weights"
            if same
            else "weights differ from the reference run (recorded, not a failure: T4 fp16 "
            "training may be nondeterministic)",
        }
        print(
            f"NOTE reproduction of {ref['run_id']} epoch {ref['epoch']}: "
            f"{report['reproduction']['note']}"
        )


def candidate_sha_from_report(train_report: dict, candidate: str) -> str | None:
    """The sha256 the training run recorded for a candidate directory."""
    if candidate == CANDIDATE_DIR:
        return train_report.get("d1_model_sha256")
    for ckpt in (train_report.get("epoch_checkpoints") or {}).values():
        if ckpt["dir"] == candidate:
            return ckpt["sha256"]
    return None


def run_d1_final_eval(
    config: dict, repo_dir: str, run_dir: str, candidate_name: str = CANDIDATE_DIR
) -> dict:
    """Single-use final-eval no-regression check for one saved candidate."""
    out_path = os.path.join(run_dir, final_eval_report_name(candidate_name))
    if os.path.exists(out_path):
        raise SystemExit(f"Final-eval already ran for this candidate: {out_path}")
    with open(os.path.join(run_dir, REPORT_FILE), encoding="utf-8") as f:
        train_report = json.load(f)
    if train_report.get("status") != "trained_pending_final_eval":
        raise SystemExit(f"No candidate ready: status {train_report.get('status')}")
    candidate = os.path.join(run_dir, candidate_name)
    d1_sha = _sha256(os.path.join(candidate, "model.safetensors"))
    if d1_sha != candidate_sha_from_report(train_report, candidate_name):
        raise SystemExit(f"Candidate changed since training: {d1_sha}")
    _verify_detector(config)

    splits_dir = os.path.join(repo_dir, "artifacts", "splits")
    if config.get("rebuild_locks"):
        _rebuild_splits(config, repo_dir, splits_dir)
    _final_eval_ids(config, splits_dir)
    path = os.path.join(splits_dir, "final_eval.jsonl")
    if _sha256(path) != config["locks"]["final_eval_sha256"]:
        raise SystemExit("final_eval.jsonl does not match its lock.")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if config.get("limits"):
        rows = rows[: config["limits"]["final_eval"]]

    device, gpu, _ = _device_dtype()
    max_length = config["detector"]["max_length"]
    eval_bs = config["training"]["eval_batch_size"]
    tok = AutoTokenizer.from_pretrained(candidate)
    scores = {}
    for name, model_path in (("d0", config["detector"]["path"]), ("d1", candidate)):
        model = _load_detector(model_path, device)
        preds = predict_spans(model, tok, rows, max_length, eval_bs, device)
        scores[name] = span_metrics(rows, preds)
        del model
    gate = no_regression_gate(scores["d0"], scores["d1"], config["gates"]["final_eval_max_drop"])
    report = {
        "run_id": train_report["run_id"],
        "candidate": candidate_name,
        "executed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "device": gpu,
        "d1_model_sha256": d1_sha,
        "final_eval_sha256": config["locks"]["final_eval_sha256"],
        "d0": strip_per_row(scores["d0"]),
        "d1": strip_per_row(scores["d1"]),
        "paired": paired_changes(scores["d0"], scores["d1"]),
        "no_regression_gate": gate,
    }
    _write_json(out_path, report)
    print(json.dumps({k: report[k] for k in ("d0", "d1", "no_regression_gate")}, indent=2))
    print(f"report: {out_path}")
    print("No-regression", "PASS" if gate["ok"] else "FAIL", "- freezing D1 is an owner decision.")
    return report
