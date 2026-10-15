"""Validate authored SFT v2 batches (code, not self-report).

Reads authored batch JSONL files (quota fields + text), rebuilds v2 rows with
code-computed spans, and reports: pass rate per check, final counts per cell,
top-10 opening share, name-at-start share, and 20 seeded random rows for
manual review. D0 extra-name flags are merged from a flags JSON file when
present (flag only — never a rejection); otherwise reported as pending.

Usage:
  python3 scripts/validate_sft_v2.py --batches /tmp/sft_v2_batches/authored_train_*.jsonl ...
  python3 scripts/validate_sft_v2.py --batches '/tmp/sft_v2_batches/authored_*.jsonl' ...
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter

sys.path.insert(0, "src")

from pii_redteam import sft_synth_v2 as v2

QUOTA_FIELDS = (
    "split",
    "idx",
    "context_type",
    "target_length",
    "name_position",
    "name_form",
    "name",
)


def build_row(authored: dict) -> dict:
    """Rebuild a v2 row from authored fields with a code-computed span."""
    name, text = authored["name"], authored["text"]
    span = v2.compute_span(text, name)
    return {
        "id": f"spark-sft-v1-{authored['split']}-{authored['idx']:04d}",
        "split": authored["split"],
        "request": {
            "language": "en",
            "context_type": authored["context_type"],
            "target_length": authored["target_length"],
            "name_position": authored["name_position"],
            "entities": [{"type": "PERSON", "value": name}],
        },
        "target": {"text": text, "span": [span[0], span[1]] if span else None},
        "provenance": "template_paraphrase",
        "name_form": authored["name_form"],
    }


def check_quota_consistency(authored: dict, quota: dict) -> list[str]:
    """Authored fields must match the quota line exactly (no drift)."""
    errors = []
    for field in QUOTA_FIELDS:
        if authored.get(field) != quota.get(field):
            errors.append(
                f"quota drift on {field}: {authored.get(field)!r} != {quota.get(field)!r}"
            )
    if not isinstance(authored.get("text"), str) or not authored["text"].strip():
        errors.append("missing text")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", required=True)
    parser.add_argument("--quota-dir", default="/tmp/sft_v2_batches")
    parser.add_argument("--flags", default=None, help="d0 flags JSON (flag-only merge)")
    parser.add_argument("--report", default="/tmp/sft_v2_batches/validation_report.json")
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    quotas: dict[tuple[str, int], dict] = {}
    for path in glob.glob(f"{args.quota_dir}/quota_*.jsonl"):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                q = json.loads(line)
                quotas[(q["split"], q["idx"])] = q

    paths: list[str] = []
    for pat in args.batches:
        paths += sorted(glob.glob(pat))
    authored_rows = []
    for path in paths:
        for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
            if line.strip():
                try:
                    authored_rows.append((path, lineno, json.loads(line)))
                except json.JSONDecodeError as e:
                    print(f"BAD JSON {path}:{lineno}: {e}")

    flags: dict[str, int] = {}
    if args.flags:
        flags = {k: v for k, v in json.load(open(args.flags, encoding="utf-8")).items()}

    per_check: Counter = Counter()
    per_check_pass: Counter = Counter()
    failures: list[dict] = []
    built: list[dict] = []
    seen_ids: Counter = Counter()

    for path, lineno, a in authored_rows:
        key = (a.get("split"), a.get("idx"))
        quota = quotas.get(key)
        loc = f"{path}:{lineno} split={a.get('split')} idx={a.get('idx')}"
        row_errors: list[str] = []
        if quota is None:
            row_errors.append("no matching quota line")
        else:
            row_errors += check_quota_consistency(a, quota)
        if not row_errors:
            row = build_row(a)
            seen_ids[row["id"]] += 1
            row_errors += v2.validate_row(row)
            built.append(row)
        checks = {
            "schema": not any(
                "split" in e
                or "context_type" in e
                or "target_length" in e
                or "name_position" in e
                or "entities" in e
                or "provenance" in e
                or "name_form" in e
                or "quota" in e
                or "matching quota" in e
                for e in row_errors
            ),
            "exact_once": not any(
                "exactly once" in e or "missing (no word" in e for e in row_errors
            ),
            "spans": not any("span" in e for e in row_errors),
            "refusal_meta": not any("refusal/meta" in e for e in row_errors),
            "pii_containment": not any("PII token" in e for e in row_errors),
            "length": not any("word count" in e for e in row_errors),
            "position": not any("name_position" in e and "unknown" not in e for e in row_errors),
            "filler": not any("filler:" in e for e in row_errors),
        }
        for name, ok in checks.items():
            per_check[name] += 1
            per_check_pass[name] += ok
        if row_errors:
            failures.append({"loc": loc, "errors": row_errors})

    dup = v2.find_duplicates(built)
    exact_ids = {built[j]["id"] for _, j in dup["exact"]}
    near_ids = {built[j]["id"] for _, j, _ in dup["near"]}

    # template reuse: within-split rates + cross-split overlap with eval (hard fail)
    reuse = v2.find_template_reuse(built)
    involved_ids = {built[i]["id"] for i in reuse["rows_involved"]}
    by_split: Counter = Counter(r["split"] for r in built)
    reuse_by_split: Counter = Counter(r["split"] for r in built if r["id"] in involved_ids)
    print("--- template reuse (>= 8 consecutive shared words) ---")
    for split in sorted(by_split):
        n_inv = reuse_by_split.get(split, 0)
        print(f"  {split}: {n_inv}/{by_split[split]} rows involved in a reuse pair")
    overlap = v2.cross_split_template_overlap(built)
    print(f"cross-split overlap with eval rows (must be ZERO): {len(overlap)} row pairs")
    for a, b, c in overlap[:10]:
        print(f"  overlap {c} shingles: {a} <-> {b}")

    openings = Counter(v2.opening(r["target"]["text"]) for r in built)
    top10_share = sum(c for _, c in openings.most_common(10)) / len(built) if built else 0.0
    fracs = [
        v2.position_fraction(tuple(r["target"]["span"]), r["target"]["text"])  # type: ignore[arg-type]
        for r in built
        if r["target"]["span"]
    ]
    name_at_start = sum(1 for f in fracs if f < 0.25) / len(fracs) if fracs else 0.0

    flagged = sum(1 for r in built if flags.get(r["id"], 0) > 0)

    print(f"rows: {len(authored_rows)} built: {len(built)} failures: {len(failures)}")
    print("--- pass rate per check ---")
    for name in per_check:
        rate = per_check_pass[name] / per_check[name]
        print(f"  {name:15s} {per_check_pass[name]:5d}/{per_check[name]:5d} = {rate:.4f}")
    print(f"exact dup pairs: {len(dup['exact'])} rows-involved: {len(exact_ids)}")
    print(f"near dup pairs (>0.7): {len(dup['near'])} rows-involved: {len(near_ids)}")
    for i, j, s in dup["near"][:10]:
        print(f"  near {s}: {built[i]['id']} <-> {built[j]['id']}")
    print(f"top-10 opening share: {top10_share:.4f}")
    print(f"name-at-start share: {name_at_start:.4f}")
    if flags:
        print(
            f"D0 extra-name flags (flag-only): {flagged}/{len(built)} = {flagged / len(built):.4f}"
        )
    else:
        print("D0 extra-name flags: PENDING (no flags file)")
    dup_ids = sorted({r["id"] for r in built if seen_ids[r["id"]] > 1})
    print(f"duplicate ids across batch files: {len(dup_ids)}")
    for f_ in failures[:25]:
        print(f"  FAIL {f_['loc']}: {'; '.join(f_['errors'])}")

    rng = random.Random(args.seed)
    sample = rng.sample(built, min(20, len(built)))
    print("--- 20 random rows for manual review ---")
    for r in sample:
        print(
            f"  [{r['id']} {r['request']['context_type']}/{r['request']['name_position']}] "
            f"{r['target']['text'][:160]!r}"
        )

    report = {
        "rows": len(authored_rows),
        "built": len(built),
        "failures": len(failures),
        "failure_details": failures,
        "per_check": {k: {"pass": per_check_pass[k], "total": per_check[k]} for k in per_check},
        "exact_dup_pairs": dup["exact"],
        "near_dup_pairs": dup["near"],
        "top10_opening_share": round(top10_share, 4),
        "name_at_start_share": round(name_at_start, 4),
        "d0_flags": {"flagged": flagged, "total": len(built)} if flags else "pending",
        "duplicate_ids": dup_ids,
        "template_reuse_pairs": len(reuse["pairs"]),
        "template_reuse_rows_by_split": {
            s: {"involved": reuse_by_split.get(s, 0), "total": by_split[s]}
            for s in sorted(by_split)
        },
        "cross_split_eval_overlap": len(overlap),
        "review_sample_ids": [r["id"] for r in sample],
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"report -> {args.report}")
    return_code = 0 if not failures and not dup["exact"] and not dup_ids and not overlap else 1
    sys.exit(return_code)


if __name__ == "__main__":
    main()
