"""CLI entry points. Usage: PYTHONPATH=src python -m pii_redteam.cli probe-base --config CONFIG."""

from __future__ import annotations

import argparse
import json
import os

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(prog="pii_redteam")
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser(
        "probe-base", help="M07a base-sampling probe (diversity + reward signal)"
    )
    probe.add_argument("--config", required=True, help="path to a probe YAML config")
    compare = sub.add_parser(
        "compare-prompts",
        help="M07b prompt-variant comparison (P0/P1/P2 on the request bank v2)",
    )
    compare.add_argument("--config", required=True, help="path to a comparison YAML config")
    train_d1 = sub.add_parser(
        "train-d1", help="M07b fine-tune D0 into candidate D1, then score short_eval"
    )
    train_d1.add_argument("--config", required=True, help="path to a train-d1 YAML config")
    final_eval = sub.add_parser(
        "d1-final-eval", help="M07b.3 single-use final-eval no-regression check for a D1 run"
    )
    final_eval.add_argument("--config", required=True, help="path to a train-d1 YAML config")
    final_eval.add_argument("--run-dir", required=True, help="the train-d1 run directory")
    final_eval.add_argument(
        "--candidate", default="d1-candidate", help="candidate dir, e.g. d1-epoch2"
    )
    freeze = sub.add_parser(
        "freeze-d1",
        help="M07b.3/.4 final-eval no-regression, re-probe, then freeze D1 if every gate passes",
    )
    freeze.add_argument("--config", required=True, help="path to a freeze-d1 YAML config")
    freeze.add_argument(
        "--source-run-dir",
        default=None,
        help="training run dir; required when the config leaves source.run_dir null",
    )
    args = parser.parse_args()

    if args.command == "probe-base":
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        repo_dir = config.get("repo_dir") or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config.setdefault("repo_commit", os.environ.get("PROBE_REPO_COMMIT", "unrecorded"))
        from pii_redteam.probe import run_probe

        report = run_probe(config, repo_dir)
        print(json.dumps({"run_id": report["run_id"], "go_no_go": report["go_no_go"]}, indent=2))
    elif args.command == "compare-prompts":
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        repo_dir = config.get("repo_dir") or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config.setdefault("repo_commit", os.environ.get("PROBE_REPO_COMMIT", "unrecorded"))
        from pii_redteam.prompt_compare import run_compare

        report = run_compare(config, repo_dir)
        print(
            json.dumps(
                {"run_id": report["run_id"], "pass_by_cell": report["pass_by_cell"]}, indent=2
            )
        )
    elif args.command in ("train-d1", "d1-final-eval"):
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        repo_dir = config.get("repo_dir") or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config.setdefault("repo_commit", os.environ.get("PROBE_REPO_COMMIT", "unrecorded"))
        from pii_redteam.train_d1 import run_d1_final_eval, run_train_d1

        if args.command == "train-d1":
            report = run_train_d1(config, repo_dir)
            print(json.dumps({"run_id": report["run_id"], "status": report["status"]}, indent=2))
        else:
            run_d1_final_eval(config, repo_dir, args.run_dir, args.candidate)
    elif args.command == "freeze-d1":
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        repo_dir = config.get("repo_dir") or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config.setdefault("repo_commit", os.environ.get("PROBE_REPO_COMMIT", "unrecorded"))
        if args.source_run_dir:
            if config["source"].get("run_dir"):
                raise SystemExit("source.run_dir is set in the config; do not also pass it")
            config["source"]["run_dir"] = args.source_run_dir
        from pii_redteam.freeze_d1 import run_freeze_d1

        report = run_freeze_d1(config, repo_dir)
        print(
            json.dumps(
                {
                    "run_id": report["run_id"],
                    "status": report["status"],
                    "failing_required": report["decision"]["failing_required"],
                },
                indent=2,
            )
        )
        if report["status"] != "frozen":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
