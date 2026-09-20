"""CPU smoke entrypoint: config -> isolated run dir with manifest + report.

Usage: python3 scripts/run_smoke.py --config configs/smoke/smoke.yaml --out artifacts/smoke-cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pii_redteam import experiments  # noqa: E402
from pii_redteam.config import load_config, resolve_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke/smoke.yaml")
    parser.add_argument("--out", default="artifacts/smoke-cpu")
    args = parser.parse_args()
    cfg = resolve_config(load_config(args.config))
    fixture_path = os.path.join(ROOT, "tests", "fixtures", "examples.jsonl")
    with open(fixture_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    report = experiments.run_smoke(cfg, args.out, examples)
    det = report["detector_eval"]
    print(
        f"smoke done: {report['counts']['candidates']} candidates "
        f"({report['counts']['valid']} valid), "
        f"fake-detector P={det['precision']:.3f} R={det['recall']:.3f} "
        f"on {det['n_examples']} fixtures -> {args.out}"
    )


if __name__ == "__main__":
    main()
