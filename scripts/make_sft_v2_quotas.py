"""Build seeded cell quotas for the spark-authored SFT v2 dataset.

Reads artifacts/splits/name_pools.json, applies the R07 requestable filter
(imported, never reimplemented), partitions disjoint train/dev names, and
emits per-row quotas that Spark authors texts for in ~50-row batches.

Coverage: context_type x name_position x length bucket x name form.
  - sft_train: 1200 rows (360 cells x3 + 120 evenly spread extras)
  - sft_dev:    150 rows over context x position cells, disjoint names
  - short_eval: 300 rows over the requestable held-out names only

name_form is usage-based: 'full' rows use multi-token values; 'first' and
'surname' rows use single-token values in casual vs. titled/formal slots
(single-token pool values are ~86% given-name-like; documented limitation).

Usage: python3 scripts/make_sft_v2_quotas.py --out /tmp/sft_v2_batches
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pii_redteam.requests import is_requestable_name  # noqa: E402  (read-only R07 import)
from pii_redteam.sft_synth_v2 import CONTEXT_TYPES, LENGTH_BUCKETS, NAME_POSITIONS  # noqa: E402

SEED = 20260920
N_TRAIN, N_DEV, N_EVAL = 1200, 150, 300


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", default="artifacts/splits/name_pools.json")
    parser.add_argument("--out", default="/tmp/sft_v2_batches")
    args = parser.parse_args()

    with open(args.pools, encoding="utf-8") as f:
        pools = json.load(f)
    train_pool = [e["value"] for e in pools["train_side_pool"] if is_requestable_name(e["value"])]
    heldout = [e["value"] for e in pools["heldout_names"] if is_requestable_name(e["value"])]
    heldout_dropped = [
        e["value"] for e in pools["heldout_names"] if not is_requestable_name(e["value"])
    ]
    print(
        f"requestable train-side: {len(train_pool)}; heldout usable: {len(heldout)} "
        f"(dropped artifacts: {heldout_dropped})"
    )

    multi = sorted({v for v in train_pool if len(v.split()) > 1})
    single = sorted({v for v in train_pool if len(v.split()) == 1})
    print(f"multi-token: {len(multi)}; single-token: {len(single)}")

    rng_dev = random.Random(SEED + 1)
    rng_train = random.Random(SEED + 2)
    rng_eval = random.Random(SEED + 3)

    # ---- dev names first (disjoint from train by construction)
    dev_multi = rng_dev.sample(multi, 50)
    dev_single = rng_dev.sample(single, 100)
    dev_names = set(dev_multi) | set(set(dev_single))
    multi_rest = [v for v in multi if v not in dev_names]
    single_rest = [v for v in single if v not in dev_names]

    # ---- dev quotas: 40 context x position cells, 30x4 + 10x3 = 150
    dev_rows = []
    forms = ("first", "full", "surname")
    buckets = list(LENGTH_BUCKETS)
    k = 0
    for ci, ctx in enumerate(CONTEXT_TYPES):
        for pi, pos in enumerate(NAME_POSITIONS):
            n = 3 if (ci + pi) % 4 == 0 else 4
            for _ in range(n):
                dev_rows.append(
                    {
                        "split": "sft_dev",
                        "idx": len(dev_rows),
                        "context_type": ctx,
                        "target_length": list(buckets[k % 3]),
                        "name_position": pos,
                        "name_form": forms[k % 3],
                    }
                )
                k += 1
    assert len(dev_rows) == N_DEV, len(dev_rows)
    # assign dev names per form
    need_full = sum(1 for r in dev_rows if r["name_form"] == "full")
    need_single = N_DEV - need_full
    assert need_full <= len(dev_multi) and need_single <= len(dev_single)
    rng_dev.shuffle(dev_multi)
    rng_dev.shuffle(dev_single)
    fi = si = 0
    for r in dev_rows:
        if r["name_form"] == "full":
            r["name"] = dev_multi[fi]
            fi += 1
        else:
            r["name"] = dev_single[si]
            si += 1

    # ---- train quotas: 360 cells x3 + extras where (ci+pi+bi+fi)%3==0 (120)
    train_rows = []
    for ci, ctx in enumerate(CONTEXT_TYPES):
        for pi, pos in enumerate(NAME_POSITIONS):
            for bi, bucket in enumerate(LENGTH_BUCKETS):
                for fi, form in enumerate(forms):
                    n = 3 + (1 if (ci + pi + bi + fi) % 3 == 0 else 0)
                    for _ in range(n):
                        train_rows.append(
                            {
                                "split": "sft_train",
                                "idx": len(train_rows),
                                "context_type": ctx,
                                "target_length": list(bucket),
                                "name_position": pos,
                                "name_form": form,
                            }
                        )
    assert len(train_rows) == N_TRAIN, len(train_rows)
    t_full = [r for r in train_rows if r["name_form"] == "full"]
    t_single = [r for r in train_rows if r["name_form"] != "full"]
    print(f"train needs full={len(t_full)} single-usage={len(t_single)}")
    assert len(t_full) <= len(multi_rest) and len(t_single) <= len(single_rest)
    train_multi = rng_train.sample(multi_rest, len(t_full))
    train_single = rng_train.sample(single_rest, len(t_single))
    for r, name in zip(t_full, train_multi, strict=True):
        r["name"] = name
    for r, name in zip(t_single, train_single, strict=True):
        r["name"] = name

    # ---- leakage asserts
    train_names = {r["name"] for r in train_rows}
    dev_name_set = {r["name"] for r in dev_rows}
    assert not (train_names & dev_name_set), "train/dev name overlap"
    assert not (train_names & set(heldout)), "train uses held-out name"
    assert not (dev_name_set & set(heldout)), "dev uses held-out name"

    # ---- short_eval quotas: every usable held-out name >=1; +1 for the first
    # 102 in seeded order (different context/position/bucket), forms follow the
    # name (multi->full, single alternates first/surname usage).
    held_shuf = list(heldout)
    rng_eval.shuffle(held_shuf)
    ctx_cycle = list(CONTEXT_TYPES) * (N_EVAL // len(CONTEXT_TYPES) + 1)
    pos_cycle = [p for p in NAME_POSITIONS for _ in range(N_EVAL // len(NAME_POSITIONS) + 1)]
    bucket_cycle = [b for b in LENGTH_BUCKETS for _ in range(N_EVAL // len(LENGTH_BUCKETS) + 1)]
    rng_eval.shuffle(ctx_cycle)
    rng_eval.shuffle(pos_cycle)
    rng_eval.shuffle(bucket_cycle)
    eval_rows = []
    sis = 0
    for i, name in enumerate(held_shuf):
        reps = 2 if i < (N_EVAL - len(held_shuf)) else 1
        for _rep in range(reps):
            j = len(eval_rows)
            if len(name.split()) > 1:
                form = "full"
            else:
                form = ("first", "surname")[sis % 2]
                sis += 1
            eval_rows.append(
                {
                    "split": "short_eval",
                    "idx": j,
                    "context_type": ctx_cycle[j],
                    "target_length": list(bucket_cycle[j]),
                    "name_position": pos_cycle[j],
                    "name_form": form,
                    "name": name,
                }
            )
    assert len(eval_rows) == N_EVAL, len(eval_rows)
    # Same-name pairs must not share the full cell (avoid near-duplicate twins).
    by_name: dict[str, list[int]] = {}
    for j, r in enumerate(eval_rows):
        name = str(r["name"])
        by_name.setdefault(name, []).append(j)
    for js in by_name.values():
        if len(js) == 2 and all(
            eval_rows[js[0]][k] == eval_rows[js[1]][k]
            for k in ("context_type", "name_position", "target_length")
        ):
            a, b = js
            for c in range(len(eval_rows)):
                if c not in js and eval_rows[c]["name"] != eval_rows[a]["name"]:
                    eval_rows[b]["context_type"], eval_rows[c]["context_type"] = (
                        eval_rows[c]["context_type"],
                        eval_rows[b]["context_type"],
                    )
                    break

    os.makedirs(args.out, exist_ok=True)
    spec = {
        "seed": SEED,
        "counts": {
            "sft_train": len(train_rows),
            "sft_dev": len(dev_rows),
            "short_eval": len(eval_rows),
        },
        "pool": {
            "requestable_train_side": len(train_pool),
            "multi": len(multi),
            "single": len(single),
            "heldout_usable": len(heldout),
            "heldout_dropped_artifacts": heldout_dropped,
        },
        "dev_name_sample": sorted(dev_name_set)[:5],
    }
    for split, rows in (("train", train_rows), ("dev", dev_rows), ("short_eval", eval_rows)):
        path = os.path.join(args.out, f"quota_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} quotas -> {path}")
    with open(os.path.join(args.out, "quota_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    print(json.dumps(spec["counts"]))


if __name__ == "__main__":
    main()
