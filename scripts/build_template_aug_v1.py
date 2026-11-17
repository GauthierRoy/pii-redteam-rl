"""M06.1 template baseline: one template text per rl_aug_v1 row (same names, same context types).

python scripts/build_template_aug_v1.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from pii_redteam.baselines import TEMPLATES, template_rows

SOURCE = "data/synthetic/rl_aug_v1.jsonl"
OUT = "data/synthetic/template_aug_v1.jsonl"
SEED = 20261117


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    with open(SOURCE, encoding="utf-8") as f:
        rl_rows = [json.loads(line) for line in f if line.strip()]
    rows = template_rows(rl_rows, SEED)
    with open(OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "file": OUT,
        "sha256": sha256(OUT),
        "rows": len(rows),
        "names": len({r["request"]["entities"][0]["value"] for r in rows}),
        "names_and_contexts_from": {"file": SOURCE, "sha256": sha256(SOURCE)},
        "templates": sum(len(t) for t in TEMPLATES.values()),
        "seed": SEED,
        "by_context": dict(Counter(r["request"]["context_type"] for r in rows)),
        "distinct_texts_ignoring_name": len(
            {r["target"]["text"].replace(r["request"]["entities"][0]["value"], "") for r in rows}
        ),
        "opens_with_dear": sum(r["target"]["text"].lower().startswith("dear") for r in rows),
    }
    with open(OUT.replace(".jsonl", "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
