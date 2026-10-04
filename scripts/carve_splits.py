"""Carve locked split roles and name pools from the pinned AI4Privacy files (M02.3–M02.4).

Usage: uv run python scripts/carve_splits.py
Writes artifacts/splits/{final_eval.jsonl,final_eval_manifest.json,
train_side_index.jsonl,validation_index.jsonl,name_pools.json,report.json}.

Policy (docs/DATA_CARD.md, "Split roles"): validation file is development/calibration
only; a seeded carve of the train file becomes the locked final-eval partition;
remaining train rows carry detector-training / generator-SFT / attack-development
roles. Exact-text duplicates are removed; template-skeleton reuse is measured and
reported, not banned. Held-out names occur only in final-eval rows and never in
train-side name pools.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pii_redteam.data import (
    REVISION,
    SOURCE,
    QuarantinedRecord,
    carve_final_eval,
    carve_heldout_names,
    dedup_by_text,
    person_name_counts,
    template_skeleton,
    text_sha256,
    to_canonical,
)

TRAIN_FILE = "data/train/1english_openpii_30k.jsonl"
VALIDATION_FILE = "data/validation/1english_openpii_8k.jsonl"
BASE = "https://huggingface.co/datasets/ai4privacy/pii-masking-300k/resolve"
SPLIT_SEED = 20260920
FINAL_EVAL_N = 1000
HELDOUT_NAMES_N = 200


def fetch_cached(path: str, cache_dir: str) -> str:
    """Download the pinned file once into cache_dir; return the local path."""
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, os.path.basename(path))
    if os.path.exists(local):
        return local
    url = f"{BASE}/{REVISION}/{path}"
    print(f"downloading {url}")
    part = local + ".part"
    with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(part, local)
    return local


def scan_file(local_path: str) -> dict:
    """One pass: canonical rows, quarantine/duplicate counts, hashes, name counts."""
    canonical_rows: list[dict] = []
    quarantined: list[str] = []
    skeleton_hashes: dict[str, str] = {}
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                canonical = to_canonical(record)
            except QuarantinedRecord:
                quarantined.append(record.get("id", "?"))
                continue
            canonical_rows.append(canonical)
            skeleton_hashes[canonical["example_id"]] = hashlib.sha256(
                template_skeleton(record["source_text"], record["privacy_mask"]).encode("utf-8")
            ).hexdigest()
    kept, dupes = dedup_by_text(canonical_rows)
    return {
        "kept": kept,
        "skeleton_hashes": skeleton_hashes,
        "n_rows": len(canonical_rows) + len(quarantined),
        "n_quarantined": len(quarantined),
        "quarantined_ids": quarantined,
        "n_exact_duplicates": len(dupes),
        "duplicate_ids": dupes,
        "n_person_rows": sum(1 for r in kept if r["spans"]),
    }


def index_lines(rows: list[dict], role: str, skeletons: dict[str, str]) -> list[dict]:
    return [
        {
            "example_id": r["example_id"],
            "role": role,
            "text_sha256": text_sha256(r["text"]),
            "template_sha256": skeletons[r["example_id"]],
            "n_person_spans": len(r["spans"]),
            "person_names": [r["text"][s["start"] : s["end"]] for s in r["spans"]],
        }
        for r in rows
    ]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/tmp/pii-redteam-cache")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--final-eval-n", type=int, default=FINAL_EVAL_N)
    parser.add_argument("--heldout-n", type=int, default=HELDOUT_NAMES_N)
    parser.add_argument("--out", default="artifacts/splits")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train = scan_file(fetch_cached(TRAIN_FILE, args.cache_dir))
    validation = scan_file(fetch_cached(VALIDATION_FILE, args.cache_dir))
    print(
        f"train rows={train['n_rows']} quarantined={train['n_quarantined']} "
        f"dupes={train['n_exact_duplicates']} person_rows={train['n_person_rows']}"
    )
    print(
        f"validation rows={validation['n_rows']} quarantined={validation['n_quarantined']} "
        f"dupes={validation['n_exact_duplicates']} person_rows={validation['n_person_rows']}"
    )

    # Cross-file exact-duplicate check: a train-side text also present in the
    # development file must never enter the final-eval partition.
    dev_text_hashes = {text_sha256(r["text"]) for r in validation["kept"]}
    candidates = [r for r in train["kept"] if text_sha256(r["text"]) not in dev_text_hashes]
    n_cross_file_dupes = len(train["kept"]) - len(candidates)

    final_eval_ids = set(
        carve_final_eval([r["example_id"] for r in candidates], seed=args.seed, n=args.final_eval_n)
    )
    final_eval_rows = [r for r in candidates if r["example_id"] in final_eval_ids]
    train_side_rows = [r for r in candidates if r["example_id"] not in final_eval_ids]
    assert len(final_eval_rows) == args.final_eval_n, "final-eval carve came up short"

    # Name pools: train-side pool draws only from train-side rows; held-out names
    # occur only in final-eval rows (never anywhere in train-side texts).
    train_side_names = person_name_counts(train_side_rows)
    final_eval_names = person_name_counts(final_eval_rows)
    heldout = carve_heldout_names(
        final_eval_names, train_side_names, seed=args.seed, heldout_n=args.heldout_n
    )
    train_pool = sorted(
        (name, count) for name, count in train_side_names.items() if name not in set(heldout)
    )
    assert not set(heldout) & {name for name, _ in train_pool}, "heldout/train-pool overlap"

    # Leakage checks (acceptance gate M02): id/text disjointness by construction,
    # cross-file exact duplicates excluded, held-out names disjoint from the pool.
    train_side_hashes = {text_sha256(r["text"]) for r in train_side_rows}
    final_eval_hashes = {text_sha256(r["text"]) for r in final_eval_rows}
    assert not (train_side_hashes & final_eval_hashes)
    assert not (final_eval_hashes & dev_text_hashes)
    # Template reuse is expected in this synthetic dataset: measured, not banned.
    train_side_templates = {train["skeleton_hashes"][r["example_id"]] for r in train_side_rows}
    template_leak = sum(
        1
        for r in final_eval_rows
        if train["skeleton_hashes"][r["example_id"]] in train_side_templates
    )
    dev_templates = set(validation["skeleton_hashes"].values())
    template_leak_dev = sum(
        1 for r in final_eval_rows if train["skeleton_hashes"][r["example_id"]] in dev_templates
    )

    write_jsonl(
        os.path.join(args.out, "final_eval.jsonl"),
        [{**r, "role": "final_eval"} for r in final_eval_rows],
    )
    final_eval_sha = hashlib.sha256(
        open(os.path.join(args.out, "final_eval.jsonl"), "rb").read()
    ).hexdigest()
    write_jsonl(
        os.path.join(args.out, "train_side_index.jsonl"),
        index_lines(train_side_rows, "train_side", train["skeleton_hashes"]),
    )
    write_jsonl(
        os.path.join(args.out, "validation_index.jsonl"),
        index_lines(validation["kept"], "dev_calibration", validation["skeleton_hashes"]),
    )
    name_pools = {
        "version": 1,
        "seed": args.seed,
        "provenance": {"source": SOURCE, "revision": REVISION, "train_file": TRAIN_FILE},
        "train_side_pool": [{"value": n, "count": c} for n, c in train_pool],
        "heldout_names": [
            {"value": n, "count": final_eval_names[n], "origin": "final_eval_unique"}
            for n in heldout
        ],
        "heldout_rule": "names occurring only in final-eval rows; disjoint from train_side_pool",
    }
    with open(os.path.join(args.out, "name_pools.json"), "w", encoding="utf-8") as f:
        json.dump(name_pools, f, ensure_ascii=False, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "final_eval_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": SOURCE,
                "revision": REVISION,
                "train_file": TRAIN_FILE,
                "validation_file": VALIDATION_FILE,
                "seed": args.seed,
                "final_eval_n": len(final_eval_rows),
                "rows": [
                    {"example_id": r["example_id"], "text_sha256": text_sha256(r["text"])}
                    for r in final_eval_rows
                ],
                "final_eval_sha256": final_eval_sha,
                "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "locked": True,
                "policy": "no generator training, prompt authoring, or reward tuning from these rows",
            },
            f,
            indent=2,
            sort_keys=True,
        )

    report = {
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "seed": args.seed,
        "train": {
            k: train[k] for k in ("n_rows", "n_quarantined", "n_exact_duplicates", "n_person_rows")
        },
        "validation": {
            k: validation[k]
            for k in ("n_rows", "n_quarantined", "n_exact_duplicates", "n_person_rows")
        },
        "roles": {
            "final_eval": len(final_eval_rows),
            "train_side": len(train_side_rows),
            "dev_calibration": len(validation["kept"]),
        },
        "final_eval_person_rows": sum(1 for r in final_eval_rows if r["spans"]),
        "cross_file_exact_duplicates_excluded": n_cross_file_dupes,
        "template_reuse": {
            "final_eval_rows_sharing_template_with_train_side": template_leak,
            "final_eval_rows_sharing_template_with_dev": template_leak_dev,
            "policy": "measured and reported; exact-text duplication is banned, template reuse is not",
        },
        "names": {
            "train_side_unique_names": len(train_side_names),
            "final_eval_unique_names": len(final_eval_names),
            "names_unique_to_final_eval": len(
                [n for n in final_eval_names if n not in train_side_names]
            ),
            "heldout_selected": len(heldout),
            "heldout_in_train_pool": len(set(heldout) & {n for n, _ in train_pool}),
            "dev_names": len(person_name_counts(validation["kept"])),
        },
        "leakage_checks": {
            "final_eval_train_side_text_disjoint": True,
            "final_eval_dev_text_disjoint": True,
            "heldout_train_pool_disjoint": True,
        },
    }
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(
        f"final_eval={len(final_eval_rows)} train_side={len(train_side_rows)} heldout_names={len(heldout)}"
    )
    print(f"names unique to final_eval={report['names']['names_unique_to_final_eval']}")


if __name__ == "__main__":
    main()
