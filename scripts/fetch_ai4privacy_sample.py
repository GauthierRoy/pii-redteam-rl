"""Fetch a small pinned-revision AI4Privacy sample (stdlib only, no storage of full files).

Usage: uv run python scripts/fetch_ai4privacy_sample.py --max-records 200
Writes artifacts/ai4privacy_sample.jsonl + artifacts/ai4privacy_sample_manifest.json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import urllib.request

REVISION = "c8c77895a005822682b66ab547fc0422579bc1d3"
BASE = "https://huggingface.co/datasets/ai4privacy/pii-masking-300k/resolve"


def fetch_sample(path: str, max_records: int, revision: str = REVISION) -> tuple[list[dict], str]:
    url = f"{BASE}/{revision}/{path}"
    digest = hashlib.sha256()
    records: list[dict] = []
    buf = b""
    offset = 0
    while len(records) < max_records:
        req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset + 999999}"})
        try:
            chunk = urllib.request.urlopen(req, timeout=120).read()
        except Exception:
            break
        if not chunk:
            break
        offset += len(chunk)
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for line in lines:
            if not line.strip() or len(records) >= max_records:
                continue
            digest.update(line + b"\n")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records, url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/train/1english_openpii_30k.jsonl")
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--out", default="artifacts/ai4privacy_sample.jsonl")
    args = parser.parse_args()
    records, url = fetch_sample(args.file, args.max_records)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "source": "ai4privacy/pii-masking-300k",
        "revision": REVISION,
        "url": url,
        "records": len(records),
        "sha256": hashlib.sha256(open(args.out, "rb").read()).hexdigest(),
        "fetched_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    with open(os.path.splitext(args.out)[0] + "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"sample: {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
