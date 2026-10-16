"""Finalize d1_aug_v1 (LLM-authored-direct stage-1 augmentation set; M07b.2).

Reads the authored batch files (data/synthetic/d1_aug_v1_batches/), builds
final rows with code-computed spans, hard-fails on ANY validation error
(including filler and placeholder-span checks), requires zero template reuse
within the set and zero 8-word overlap with the evaluation sets (short_eval_v2,
short_eval_human_v1 when present, deprecated short_eval_v1) and with
spark_sft_v1 train/dev, then writes:

  data/synthetic/d1_aug_v1.jsonl       (2000 rows, split sft_train)
  data/synthetic/d1_aug_v1_manifest.json

This set is training-side augmentation data for D1 (M07b.2). Names come from
the training-side pool only; held-out/final-eval names are excluded.

Usage: PYTHONPATH=src python3 scripts/finalize_d1_aug_v1.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pii_redteam import sft_synth_v2 as v2  # noqa: E402

BATCH_GLOB = "data/synthetic/d1_aug_v1_batches/batch_*.jsonl"
OUT = "data/synthetic/d1_aug_v1.jsonl"
MANIFEST = "data/synthetic/d1_aug_v1_manifest.json"
PROVENANCE = "llm_authored_direct:deepseek-v1-flash"
DATASET = "d1-aug-v1"
ROWS = 2000

EVAL_OVERLAP_PATHS = {
    "short_eval_v2": ["data/synthetic/short_eval_v2.jsonl"],
    "short_eval_human_v1": ["data/synthetic/short_eval_human_v1.jsonl"],
    "deprecated_short_eval_v1": ["data/deprecated/spark_sft_v1_short_eval_v1.jsonl"],
}
TRAIN_OVERLAP_PATHS = {
    "spark_sft_v1_train_dev": [
        "data/synthetic/spark_sft_v1/train.jsonl",
        "data/synthetic/spark_sft_v1/dev.jsonl",
    ],
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_texts(paths: list[str]) -> list[str]:
    texts = []
    for path in paths:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                texts.append(r["target"]["text"])
    return texts


def overlap_rows(rows: list[dict], shingles: set[tuple[str, ...]]) -> list[tuple[str, int]]:
    out = []
    for r in rows:
        shared = v2._shingles(r["target"]["text"], v2.TEMPLATE_REUSE_WORDS) & shingles
        if shared:
            out.append((r["id"], len(shared)))
    return out


def text_shingles(texts: list[str]) -> set[tuple[str, ...]]:
    sh: set[tuple[str, ...]] = set()
    for t in texts:
        sh |= v2._shingles(t, v2.TEMPLATE_REUSE_WORDS)
    return sh


def main() -> None:
    auth = []
    for path in sorted(glob.glob(BATCH_GLOB)):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                auth.append(json.loads(line))
    assert len(auth) == ROWS, len(auth)
    ids = [a["idx"] for a in auth]
    assert ids == list(range(ROWS)), "idx must be 0..1999 exactly once"

    rows = []
    for a in auth:
        span = v2.compute_span(a["text"], a["name"])
        assert span is not None, f"no exact-once span for idx {a['idx']}"
        row = {
            "id": f"{DATASET}-{a['idx']:05d}",
            "split": "sft_train",
            "request": {
                "language": "en",
                "context_type": a["context_type"],
                "target_length": a["target_length"],
                "name_position": a["name_position"],
                "entities": [{"type": "PERSON", "value": a["name"]}],
            },
            "target": {"text": a["text"], "span": [span[0], span[1]]},
            "provenance": PROVENANCE,
            "name_form": a["name_form"],
        }
        errs = v2.validate_row(row)
        assert not errs, f"idx {a['idx']} validation failures: {errs}"
        rows.append(row)

    # duplicates + within-set template reuse must be zero
    dup = v2.find_duplicates(rows)
    assert not dup["exact"], dup["exact"][:5]
    assert not dup["near"], [(rows[i]["id"], rows[j]["id"]) for i, j, _ in dup["near"][:5]]
    reuse = v2.find_template_reuse(rows)
    assert not reuse["pairs"], f"within-set template reuse: {reuse['pairs'][:5]}"

    # zero 8-word overlap with evaluation sets (hard) and spark_sft_v1 train/dev
    overlap_report: dict[str, int] = {}
    for name, paths in EVAL_OVERLAP_PATHS.items():
        present = [p for p in paths if os.path.exists(p)]
        if not present:
            overlap_report[name] = -1  # not present in this checkout; verify on the PR
            continue
        hits = overlap_rows(rows, text_shingles(load_texts(present)))
        assert not hits, f"template overlap with {name}: {hits[:10]}"
        overlap_report[name] = 0
    for name, paths in TRAIN_OVERLAP_PATHS.items():
        hits = overlap_rows(rows, text_shingles(load_texts(paths)))
        assert not hits, f"template overlap with {name}: {hits[:10]}"
        overlap_report[name] = 0

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    openings = Counter(v2.opening(r["target"]["text"]) for r in rows)
    top10 = sum(c for _, c in openings.most_common(10)) / len(rows)
    fracs = [v2.position_fraction(tuple(r["target"]["span"]), r["target"]["text"]) for r in rows]
    name_at_start = sum(1 for f in fracs if f < 0.25) / len(fracs)

    by_ctx: Counter = Counter(r["request"]["context_type"] for r in rows)
    by_pos: Counter = Counter(r["request"]["name_position"] for r in rows)
    by_bucket: Counter = Counter(tuple(r["request"]["target_length"]) for r in rows)
    by_form: Counter = Counter(r["name_form"] for r in rows)
    names = [r["request"]["entities"][0]["value"] for r in rows]

    manifest = {
        "dataset": DATASET,
        "provenance": PROVENANCE,
        "authoring": "each text written directly by the LLM (DeepSeek V1 Flash via opencode, "
        "2026-10-15/16); no templates, paraphrase tools, synonym swaps, or filler padding; "
        "authored drafts preserved in data/synthetic/d1_aug_v1_batches/ (40 files x 50); "
        "batch assignments (context/position/length/name) pre-generated deterministically "
        "(seed 20261015) and each text authored around the assigned entity",
        "names": "training-side names only (artifacts/splits/name_pools.json train_side_pool); "
        f"{len(set(names))} distinct names across {len(rows)} rows; zero overlap with "
        "held-out (final-eval-only) names; not evaluation data",
        "role": "M07b.2 stage-1 augmentation data: plain short-message SFT rows used to "
        "fine-tune D0 -> D1 (same task schema / request distribution as short_eval); "
        "not to be used as evaluation data",
        "files": {
            DATASET.replace("-", "_"): {"path": OUT, "rows": len(rows), "sha256": sha256_file(OUT)}
        },
        "counts": {"rows": len(rows)},
        "coverage": {
            "per_context_type": dict(sorted(by_ctx.items())),
            "per_name_position": dict(sorted(by_pos.items())),
            "per_length_bucket": {f"{lo}-{hi}": n for (lo, hi), n in sorted(by_bucket.items())},
            "per_name_form": dict(sorted(by_form.items())),
        },
        "validation": {
            "validator": "pii_redteam.sft_synth_v2.validate_row (all checks incl. filler)",
            "failures": 0,
            "exact_dup_pairs": 0,
            "near_dup_pairs_gt_0.7": 0,
            "within_set_template_reuse_pairs": 0,
            "cross_dataset_8gram_overlap": overlap_report,
            "top10_opening_share": round(top10, 4),
            "name_at_start_share": round(name_at_start, 4),
        },
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"rows: {len(rows)} -> {OUT} sha256={manifest['files']['d1_aug_v1']['sha256']}")
    print(f"coverage: ctx={dict(sorted(by_ctx.items()))}")
    print(
        f"  pos={dict(sorted(by_pos.items()))} bucket={ {f'{lo}-{hi}': n for (lo, hi), n in sorted(by_bucket.items())} } form={dict(sorted(by_form.items()))}"
    )
    print(f"top10_opening_share={top10:.4f} name_at_start={name_at_start:.4f}")
    print(f"cross-dataset overlap: {overlap_report}")
    rng = random.Random(20261015)
    print("--- 20 random rows for owner review ---")
    for r in rng.sample(rows, 20):
        print(
            f"  [{r['id']} {r['request']['context_type']}/{r['request']['name_position']}/{r['name_form']}]"
        )
        print(f"    {r['target']['text'][:220]!r}")


if __name__ == "__main__":
    main()
