"""Pick balanced dev-100 subset from quota_dev (150 rows) - deterministic.

150 rows form 120 cells (90x1 + 30x2) keyed (ctx,pos,bucket,form).
Take 1 from each double cell (30) + 70 of 90 singles (drop 2 per context).
Result: 100 rows, 100 cells, 10 per context. Seed 20260920.
"""

import json
import os
import random
from collections import Counter, defaultdict

SRC = "/tmp/sft_v2_batches/quota_dev.jsonl"
D100 = "/tmp/sft_v2_batches_dev100"
os.makedirs(D100, exist_ok=True)

rows = [json.loads(line) for line in open(SRC) if line.strip()]
assert len(rows) == 150
cells = defaultdict(list)
for r in rows:
    cells[
        (r["context_type"], r["name_position"], tuple(r["target_length"]), r["name_form"])
    ].append(r)
singles = {k: v for k, v in cells.items() if len(v) == 1}
doubles = {k: v for k, v in cells.items() if len(v) == 2}
assert len(singles) == 90 and len(doubles) == 30

rng = random.Random(20260920)
picked = [rng.choice(doubles[k]) for k in sorted(doubles)]
by_ctx = defaultdict(list)
for k, v in singles.items():
    by_ctx[k[0]].append(v[0])
dropped = []
for ctx in sorted(by_ctx):
    lst = by_ctx[ctx][:]
    rng.shuffle(lst)
    dropped.extend(lst[:2])
    picked.extend(lst[2:])
assert len(picked) == 100
print("contexts:", dict(Counter(r["context_type"] for r in picked)))
print("positions:", dict(Counter(r["name_position"] for r in picked)))
print("buckets:", dict(Counter(tuple(r["target_length"]) for r in picked)))
print("forms:", dict(Counter(r["name_form"] for r in picked)))
with open(f"{D100}/quota_dev.jsonl", "w") as f:
    for r in sorted(picked, key=lambda r: r["idx"]):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for name in [
    "quota_train.jsonl",
    "quota_train_gap.jsonl",
    "quota_short_eval.jsonl",
    "quota_spec.json",
]:
    dst = f"{D100}/{name}"
    if not os.path.exists(dst):
        os.symlink(f"/tmp/sft_v2_batches/{name}", dst)
json.dump(sorted(r["idx"] for r in dropped), open(f"{D100}/dev100_dropped_idxs.json", "w"))
print("dropped:", sorted(r["idx"] for r in dropped))
