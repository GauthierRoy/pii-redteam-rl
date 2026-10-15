"""Rewrite residual near-dupe rows with Jaccard acceptance sampling.

For each row involved in a near pair (later index first), draw fresh
candidates (forced different template + fresh swap/filler seeds) and accept
the first that (a) passes v2 checks and (b) has J<=0.65 vs every other row.
Iterates until clean or attempts exhausted.
"""

import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_v2 as G

from pii_redteam import sft_synth_v2 as v2

BATCH_DIR = "/tmp/sft_v2_batches"
rep = json.load(open(f"{BATCH_DIR}/validation_full.json"))
pairs = [tuple(p[:2]) for p in rep["near_dup_pairs"]]
print("pairs:", len(pairs))

order, built, keys = [], [], []
for pat in [
    f"{BATCH_DIR}/authored_train_*.jsonl",
    f"{BATCH_DIR}/authored_eval_*.jsonl",
    f"{BATCH_DIR}/authored_dev_*.jsonl",
]:
    for p in sorted(glob.glob(pat)):
        for line in open(p):
            if line.strip():
                a = json.loads(line)
                keys.append((p, a["split"], a["idx"]))
                built.append(a)
                order.append(len(built) - 1)

quotas_by_key = {}
for qp in [
    f"{BATCH_DIR}/quota_train.jsonl",
    f"{BATCH_DIR}/quota_train_gap.jsonl",
    f"{BATCH_DIR}/quota_short_eval.jsonl",
    "/tmp/sft_v2_batches_dev100/quota_dev.jsonl",
]:
    for line in open(qp):
        if line.strip():
            q = json.loads(line)
            quotas_by_key[(q["split"], q["idx"])] = q

grams = [v2.word_trigrams(a["text"]) for a in built]


def max_j(j, exclude_self=True):
    gj = grams[j]
    m = 0.0
    for k, gk in enumerate(grams):
        if k == j:
            continue
        m = max(m, v2.jaccard3(gj, gk))
    return m


todo = sorted({j for p in pairs for j in p[1:]}, reverse=False)
# process most-connected first

deg = Counter(i for p in pairs for i in p)
todo = sorted(set(i for p in pairs for i in p), key=lambda i: -deg[i])
print("rows to rewrite:", len(todo))

# current template index per row is unknown; force rotation by big boost steps
rewrites = {}
for j in todo:
    a = built[j]
    q = quotas_by_key[(a["split"], a["idx"])]
    accepted = None
    for draw in range(60):
        boost = 777000 + draw * 131
        text, errs = G.author_one(q, boost=boost)
        if errs:
            continue
        g = v2.word_trigrams(text)
        ok = True
        for k, gk in enumerate(grams):
            if k == j:
                continue
            if v2.jaccard3(g, gk) > 0.65:
                ok = False
                break
        if ok:
            accepted = text
            break
    if accepted is None:
        print(f"  NO CLEAN DRAW for {(a['split'], a['idx'])} (best effort kept)")
        continue
    a["text"] = accepted
    grams[j] = v2.word_trigrams(accepted)
    rewrites[(a["split"], a["idx"])] = accepted

for p in sorted({k[0] for k in keys}):
    rows = [json.loads(line) for line in open(p) if line.strip()]
    changed = False
    for r in rows:
        if (r["split"], r["idx"]) in rewrites:
            r["text"] = rewrites[(r["split"], r["idx"])]
            changed = True
    if changed:
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"rewrote {len(rewrites)} rows")
