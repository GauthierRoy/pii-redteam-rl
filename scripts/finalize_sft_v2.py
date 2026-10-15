"""Finalize the spark-authored SFT v2 dataset (train/dev/short_eval + manifest).

Reads authored batch JSONL (quota fields + text), rebuilds v2 rows with
code-computed spans, re-validates everything (fail hard), and writes:
  data/synthetic/spark_sft_v1/train.jsonl        (1195: idx 0-1149 + 45 gap)
  data/synthetic/spark_sft_v1/dev.jsonl          (100, dev-100 subset)
  data/synthetic/spark_sft_v1/short_eval_v1.jsonl (300, held-out names only)
  data/synthetic/spark_sft_v1/manifest.json      (sha256 + generation spec)

Usage: PYTHONPATH=src python3 scripts/finalize_sft_v2.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pii_redteam import sft_synth_v2 as v2  # noqa: E402

BATCH_DIR = "/tmp/sft_v2_batches"
DEV_QUOTA_DIR = "/tmp/sft_v2_batches_dev100"
OUT_DIR = "data/synthetic/spark_sft_v1"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_authored(patterns: list[str]) -> list[dict]:
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def build_row(a: dict) -> dict:
    name, text = a["name"], a["text"]
    span = v2.compute_span(text, name)
    assert span is not None, f"no exact-once span for {(a['split'], a['idx'])}"
    return {
        "id": f"spark-sft-v1-{a['split']}-{a['idx']:04d}",
        "split": a["split"],
        "request": {
            "language": "en",
            "context_type": a["context_type"],
            "target_length": a["target_length"],
            "name_position": a["name_position"],
            "entities": [{"type": "PERSON", "value": name}],
        },
        "target": {"text": text, "span": [span[0], span[1]]},
        "provenance": "spark_authored",
        "name_form": a["name_form"],
    }


def main() -> None:
    quotas: dict[tuple[str, int], dict] = {}
    for path in glob.glob(f"{BATCH_DIR}/quota_*.jsonl") + [f"{DEV_QUOTA_DIR}/quota_dev.jsonl"]:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                q = json.loads(line)
                quotas[(q["split"], q["idx"])] = q

    groups = {
        "train": [f"{BATCH_DIR}/authored_train_{i:02d}.jsonl" for i in range(1, 24)]
        + [f"{BATCH_DIR}/authored_train_gap.jsonl"],
        "dev": [f"{BATCH_DIR}/authored_dev_01.jsonl", f"{BATCH_DIR}/authored_dev_02.jsonl"],
        "short_eval": [f"{BATCH_DIR}/authored_eval_{i:02d}.jsonl" for i in range(1, 7)],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    files, all_rows = {}, []
    for split, pats in groups.items():
        authored = []
        for pat in pats:
            assert os.path.exists(pat), f"missing batch file {pat}"
            for line in open(pat, encoding="utf-8"):
                if line.strip():
                    authored.append(json.loads(line))
        # quota consistency
        for a in authored:
            q = quotas.get((a["split"], a["idx"]))
            assert q is not None, f"no quota for {(a['split'], a['idx'])}"
            for f_ in (
                "split",
                "idx",
                "context_type",
                "target_length",
                "name_position",
                "name_form",
                "name",
            ):
                assert a[f_] == q[f_], f"quota drift {(a['split'], a['idx'])}.{f_}"
        rows = [build_row(a) for a in authored]
        errs = [(r["id"], v2.validate_row(r)) for r in rows]
        bad = [(i, e) for i, e in errs if e]
        assert not bad, f"{split} validation failures: {bad[:5]}"
        ids = Counter(r["id"] for r in rows)
        assert all(c == 1 for c in ids.values()), "duplicate ids"
        rows.sort(key=lambda r: r["id"])
        out = (
            f"{OUT_DIR}/train.jsonl"
            if split == "train"
            else f"{OUT_DIR}/dev.jsonl"
            if split == "dev"
            else f"{OUT_DIR}/short_eval_v1.jsonl"
        )
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[split] = {"path": out, "rows": len(rows), "sha256": sha256_file(out)}
        all_rows += rows
        print(f"{split}: {len(rows)} rows -> {out} sha256={files[split]['sha256'][:12]}...")

    assert files["train"]["rows"] == 1195, files["train"]
    assert files["dev"]["rows"] == 100, files["dev"]
    assert files["short_eval"]["rows"] == 300, files["short_eval"]

    dup = v2.find_duplicates(all_rows)
    assert not dup["exact"], dup["exact"][:5]
    openings = Counter(v2.opening(r["target"]["text"]) for r in all_rows)
    top10 = sum(c for _, c in openings.most_common(10)) / len(all_rows)
    fracs = [
        v2.position_fraction(tuple(r["target"]["span"]), r["target"]["text"]) for r in all_rows
    ]
    name_at_start = sum(1 for f in fracs if f < 0.25) / len(fracs)
    spec = json.load(open(f"{BATCH_DIR}/quota_spec.json", encoding="utf-8"))

    manifest = {
        "dataset": "spark-sft-v1",
        "creator": "spark_authored",
        "seed": 20260920,
        "files": files,
        "counts": {k: v["rows"] for k, v in files.items()},
        "generation_spec": {
            "method": "template-bank generation (161 templates, 40 context x position slots) "
            "+ seeded synonym paraphrase + filler padding to mid-bucket + "
            "Jaccard acceptance rewrite for near-dupes + hand-written "
            "replacements for stubborn clusters",
            "quota_summary": spec["counts"],
            "pool_provenance": {
                "name_pools": "artifacts/splits/name_pools.json",
                "requestable_train_side": spec["pool"]["requestable_train_side"],
                "heldout_usable": spec["pool"]["heldout_usable"],
                "heldout_excluded_artifacts": spec["pool"]["heldout_dropped_artifacts"],
                "pool_filter": "R07 is_requestable_name (imported read-only)",
            },
            "train_gap": "idx 0-1149 from quota_train + 45 gap rows (tail cells, idx>=1150); "
            "tail quota idx 1153/1169/1173/1189/1196 unused",
            "dev_subset": "dev-100: 1/cell from 30 double cells + 70/90 singletons "
            "(drop 2/context, seed 20260920); dropped idx in "
            "/tmp/sft_v2_batches_dev100/dev100_dropped_idxs.json",
            "name_disjointness": "train/dev names disjoint by quota construction; "
            "short_eval uses held-out names only (asserted in quota script)",
        },
        "validation": {
            "validator": "scripts/validate_sft_v2.py over all authored batches",
            "failures": 0,
            "exact_dup_pairs": 0,
            "near_dup_pairs_gt_0.7": len(dup["near"]),
            "top10_opening_share": round(top10, 4),
            "name_at_start_share": round(name_at_start, 4),
            "d0_extra_name_flags": "pending (frozen checkpoint unavailable at finalize time; "
            "/tmp/d0zip wiped; flag script ready at "
            "scripts/flag_extra_names_d0.py)",
        },
    }
    with open(f"{OUT_DIR}/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(
        f"manifest -> {OUT_DIR}/manifest.json "
        f"(near_dup_pairs={len(dup['near'])}, top10_open={top10:.4f})"
    )


if __name__ == "__main__":
    main()
