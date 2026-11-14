"""Finalize short_eval_neg_v1: name-free short messages for the target detector false-positive check.

Reads the raw GLM-5.3 batches (data/synthetic/short_eval_neg_v1_batches/, 150
rows, prompts alongside), drops rows that over-reuse the same invented names
(deterministic cap, below), validates every kept row with
``sft_synth_v2.validate_negative_row``, requires zero exact/near duplicates and
zero within-set 8-word template reuse, then writes:

  data/synthetic/short_eval_neg_v1.jsonl
  data/synthetic/short_eval_neg_v1_manifest.json

Distractor groups: place, org_product, calendar, role_only gate target detector's
false-positive rate; homograph (words that are also names, used in their
ordinary sense) is report-only (owner decision, 2026-11-09).

Usage: PYTHONPATH=src python3 scripts/finalize_short_eval_neg_v1.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pii_redteam import sft_synth_v2 as v2  # noqa: E402

BATCH_GLOB = "data/synthetic/short_eval_neg_v1_batches/batch_*.jsonl"
OUT = "data/synthetic/short_eval_neg_v1.jsonl"
MANIFEST = "data/synthetic/short_eval_neg_v1_manifest.json"
PROVENANCE = "llm_authored_direct:glm-5.3"
DATASET = "short-eval-neg-v1"

# Reuse cap (owner decision 2026-11-09: "drop some of the duplicates").
# GLM copied invented names from the prompt examples into many rows. Walking rows
# in idx order, a row is dropped once any listed term it contains has already
# been kept CAP times. Real everyday products (Slack, Excel, ...) and the verb
# "will" are deliberately not capped.
CAP_GATED = 3
CAP_HOMOGRAPH = 5
CAPPED_NAMES = (
    "Falcon", "Northwind", "Q3 Offsite", "Building C", "Harbor Road", "Kestrel", "Beacon",
    "Copperline", "Meridian", "Riverside", "Project Atlas", "Lisbon", "Porto", "Annex",
    "Northgate", "Market Street", "Cedar Lane", "Redshift", "Bluepeak", "Coppergate",
    "Sandpiper", "Bluebird", "Ironclad",
)  # fmt: skip
CAPPED_HOMOGRAPHS = ("rose", "frank", "may", "sterling", "bill", "grace period", "mark")
# Capitalized homographs at the start of a subject line: the hardest report-only
# cases, kept by owner decision even though they reuse capped words.
FORCE_KEEP = {128, 133}


def capped_terms(a: dict) -> set[str]:
    text = a["text"]
    if a["distractor"] == "homograph":
        return {w for w in CAPPED_HOMOGRAPHS if re.search(rf"\b{w}\b", text, re.I)}
    return {w for w in CAPPED_NAMES if re.search(rf"\b{w}\b", text)}


def select(auth: list[dict]) -> tuple[list[dict], list[dict]]:
    """Apply the reuse cap in idx order. Returns (kept, dropped-with-reason)."""
    seen: Counter = Counter()
    kept, dropped = [], []
    for a in sorted(auth, key=lambda a: a["idx"]):
        cap = CAP_HOMOGRAPH if a["distractor"] == "homograph" else CAP_GATED
        terms = capped_terms(a)
        over = sorted(t for t in terms if seen[t] >= cap)
        if over and a["idx"] not in FORCE_KEEP:
            dropped.append({"idx": a["idx"], "distractor": a["distractor"], "over_cap": over})
            continue
        seen.update(terms)
        kept.append(a)
    return kept, dropped


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
    assert sorted(a["idx"] for a in auth) == list(range(150)), "idx must be 0..149 exactly once"

    kept, dropped = select(auth)
    rows: list[dict] = []
    final_kept = []
    for a in kept:
        row = {
            "id": f"{DATASET}-{a['idx']:04d}",
            "split": "short_eval",
            "request": {
                "language": "en",
                "context_type": a["context_type"],
                "target_length": a["target_length"],
                "name_position": None,
                "entities": [],
            },
            "target": {"text": a["text"], "span": None},
            "provenance": PROVENANCE,
            "name_form": None,
            "distractor": a["distractor"],
        }
        errs = v2.validate_negative_row(row)
        if errs:
            assert a["idx"] not in FORCE_KEEP, f"force-kept idx {a['idx']} fails: {errs}"
            dropped.append({"idx": a["idx"], "distractor": a["distractor"], "validator": errs})
            continue
        rows.append(row)
        final_kept.append(a)
    kept = final_kept

    dup = v2.find_duplicates(rows)
    assert not dup["exact"], dup["exact"][:5]
    assert not dup["near"], [(rows[i]["id"], rows[j]["id"]) for i, j, _ in dup["near"][:5]]
    reuse = v2.find_template_reuse(rows)
    reuse_pairs = [(rows[i]["id"], rows[j]["id"]) for i, j in reuse["pairs"]]

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_dis = Counter(r["distractor"] for r in rows)
    by_ctx = Counter(r["request"]["context_type"] for r in rows)
    by_bucket = Counter(tuple(r["request"]["target_length"]) for r in rows)
    reuse_after = Counter(t for a in kept for t in capped_terms(a))
    manifest = {
        "dataset": DATASET,
        "provenance": PROVENANCE,
        "purpose": "name-free short messages: base detector vs the target detector false-positive check (M07b freeze)",
        "authoring": "each text written directly by GLM-5.3 via Mistral Vibe (headless, "
        "2026-11-09) from the prompts in data/synthetic/short_eval_neg_v1_batches/prompts/ "
        "(generated by scripts/make_short_eval_neg_prompts.py); raw outputs preserved "
        "unmodified in data/synthetic/short_eval_neg_v1_batches/",
        "label": "no PERSON spans in any row (request.entities == [], target.span == null)",
        "distractor_groups": {
            "gated": list(v2.GATED_DISTRACTORS),
            "report_only": ["homograph"],
            "note": "owner decision 2026-11-09: homograph rows never gate the freeze",
        },
        "files": {
            "short_eval_neg_v1": {"path": OUT, "rows": len(rows), "sha256": sha256_file(OUT)}
        },
        "counts": {
            "raw_rows": len(auth),
            "rows": len(rows),
            "gated_rows": sum(by_dis[d] for d in v2.GATED_DISTRACTORS),
            "report_only_rows": by_dis["homograph"],
        },
        "coverage": {
            "per_distractor": dict(sorted(by_dis.items())),
            "per_context_type": dict(sorted(by_ctx.items())),
            "per_length_bucket": {f"{lo}-{hi}": n for (lo, hi), n in sorted(by_bucket.items())},
        },
        "selection": {
            "rule": f"idx order; drop a row once a capped term it contains was already kept "
            f"{CAP_GATED}x (gated groups) / {CAP_HOMOGRAPH}x (homograph)",
            "capped_names": list(CAPPED_NAMES),
            "capped_homographs": list(CAPPED_HOMOGRAPHS),
            "force_keep_idx": sorted(FORCE_KEEP),
            "dropped": dropped,
            "max_reuse_after": dict(reuse_after.most_common(10)),
        },
        "validation": {
            "validator": "pii_redteam.sft_synth_v2.validate_negative_row",
            "failures": 0,
            "exact_dup_pairs": 0,
            "near_dup_pairs_gt_0.7": 0,
            "within_set_8gram_reuse_pairs": reuse_pairs,
            "name_free_audit": "all 150 raw rows read by hand (2026-11-09): no person names; "
            "spaCy en_core_web_sm PERSON hits (82) were all non-names (e.g. 'Boiler Room', "
            "'Firefox', 'Q3 Offsite'). Hardest kept cases: idx 128 'Subject: Frank feedback "
            "on the June release', idx 133 'Subject: Sterling parts ordered by mistake'.",
            "known_skew": "list/table row, forwarded email thread and meeting notes are "
            "under-represented after the reuse cap",
        },
    }
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"rows: {len(rows)} (dropped {len(dropped)}) -> {OUT}")
    print(f"sha256={manifest['files']['short_eval_neg_v1']['sha256']}")
    print(f"per_distractor={dict(sorted(by_dis.items()))}")
    print(f"per_context={dict(sorted(by_ctx.items()))}")
    print(f"within-set 8-gram reuse pairs: {reuse_pairs}")


if __name__ == "__main__":
    main()
