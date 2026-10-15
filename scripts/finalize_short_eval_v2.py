"""Finalize short_eval_v2 (LLM-authored-direct eval set; STEP 2, D16 follow-up).

Reads the hand-authored batch files (data/synthetic/short_eval_v2_batches/),
builds final v2 rows with code-computed spans, hard-fails on ANY validation
error (including the post-audit filler checks), requires zero template reuse
within the set and zero 8-word overlap with spark_sft_v1 train/dev and the
deprecated short_eval_v1, then writes:

  data/synthetic/short_eval_v2.jsonl   (150 rows, provenance llm_authored_direct)
  data/synthetic/short_eval_v2_manifest.json

A ~50-row human-written slice slot is reserved at
data/synthetic/short_eval_human_v1.jsonl (same schema, provenance
human_authored; owner-provided, validated by the same checks when it lands).

Usage: PYTHONPATH=src python3 scripts/finalize_short_eval_v2.py
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

BATCH_GLOB = "data/synthetic/short_eval_v2_batches/batch_*.jsonl"
OUT = "data/synthetic/short_eval_v2.jsonl"
MANIFEST = "data/synthetic/short_eval_v2_manifest.json"
HUMAN_SLOT = "data/synthetic/short_eval_human_v1.jsonl"
PROVENANCE = "llm_authored_direct"
DATASET = "short-eval-v2"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    auth = []
    for path in sorted(glob.glob(BATCH_GLOB)):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                auth.append(json.loads(line))
    assert len(auth) == 150, len(auth)
    ids = [a["idx"] for a in auth]
    assert ids == list(range(150)), "idx must be 0..149 exactly once"

    rows = []
    for a in auth:
        span = v2.compute_span(a["text"], a["name"])
        assert span is not None, f"no exact-once span for idx {a['idx']}"
        row = {
            "id": f"{DATASET}-{a['idx']:04d}",
            "split": "short_eval",
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
    names = [r["request"]["entities"][0]["value"] for r in rows]
    assert len(set(names)) == len(rows), "held-out names must be distinct, one row each"
    dup = v2.find_duplicates(rows)
    assert not dup["exact"], dup["exact"][:5]
    assert not dup["near"], [(rows[i]["id"], rows[j]["id"]) for i, j, _ in dup["near"][:5]]
    reuse = v2.find_template_reuse(rows)
    assert not reuse["pairs"], f"within-set template reuse: {reuse['pairs'][:5]}"

    # zero 8-word overlap with spark_sft_v1 train/dev and deprecated short_eval_v1
    spark = []
    for path in (
        "data/synthetic/spark_sft_v1/train.jsonl",
        "data/synthetic/spark_sft_v1/dev.jsonl",
        "data/deprecated/spark_sft_v1_short_eval_v1.jsonl",
    ):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                spark.append(json.loads(line))
    spark_shingles: set[tuple[str, ...]] = set()
    for r in spark:
        spark_shingles |= v2._shingles(r["target"]["text"], v2.TEMPLATE_REUSE_WORDS)
    overlaps = []
    for r in rows:
        shared = v2._shingles(r["target"]["text"], v2.TEMPLATE_REUSE_WORDS) & spark_shingles
        if shared:
            overlaps.append((r["id"], len(shared)))
    assert not overlaps, f"template overlap with spark_sft_v1: {overlaps[:10]}"

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

    manifest = {
        "dataset": DATASET,
        "provenance": PROVENANCE,
        "authoring": "each text written directly by an LLM (Mistral Vibe session, 2026-10-15); "
        "no templates, paraphrase tools, synonym swaps, or filler padding; "
        "authored drafts preserved in data/synthetic/short_eval_v2_batches/",
        "names": "held-out names only (artifacts/splits/name_pools.json heldout_names, "
        "R07 is_requestable_name filter); 150 distinct names, each used exactly once; "
        "never training-side names; no final-eval content",
        "files": {"short_eval_v2": {"path": OUT, "rows": len(rows), "sha256": sha256_file(OUT)}},
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
            "cross_dataset_8gram_overlap_with_spark_sft_v1": 0,
            "top10_opening_share": round(top10, 4),
            "name_at_start_share": round(name_at_start, 4),
            "independent_extra_person_check": "spaCy pass run flag-only (see STATE.md); "
            "LLM-judge pass pending weights (scripts/person_check_v2.py)",
        },
        "human_slice_slot": {
            "path": HUMAN_SLOT,
            "provenance": "human_authored",
            "rows": "~50 (owner-provided; same schema; validated by the same checks)",
        },
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"rows: {len(rows)} -> {OUT} sha256={manifest['files']['short_eval_v2']['sha256']}")
    print(f"coverage: ctx={dict(sorted(by_ctx.items()))}")
    print(
        f"  pos={dict(sorted(by_pos.items()))} bucket={ {f'{lo}-{hi}': n for (lo, hi), n in sorted(by_bucket.items())} } form={dict(sorted(by_form.items()))}"
    )
    print(f"top10_opening_share={top10:.4f} name_at_start={name_at_start:.4f}")
    rng = random.Random(20261015)
    print("--- 20 random rows for owner review ---")
    for r in rng.sample(rows, 20):
        print(
            f"  [{r['id']} {r['request']['context_type']}/{r['request']['name_position']}/{r['name_form']}]"
        )
        print(f"    {r['target']['text'][:220]!r}")


if __name__ == "__main__":
    main()
