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
    smoke = sub.add_parser(
        "rl-smoke", help="M08.3 capped RL smoke run (TRL GRPO vs the frozen target detector)"
    )
    smoke.add_argument("--config", required=True, help="path to an RL smoke YAML config")
    train_detector = sub.add_parser(
        "train-detector",
        help="fine-tune a frozen detector on an augmentation set, then score short_eval",
    )
    train_detector.add_argument("--config", required=True, help="path to a train-detector config")
    generate = sub.add_parser(
        "rl-generate", help="sample a trained RL attacker on fresh names and filter the texts"
    )
    generate.add_argument("--config", required=True, help="path to an RL generation YAML config")
    llm_attack = sub.add_parser(
        "llm-attack", help="non-RL arm: an API model writes texts with the detector's feedback"
    )
    llm_attack.add_argument("--config", required=True, help="path to an LLM attack YAML config")
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
    elif args.command == "rl-smoke":
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config.setdefault("repo_commit", os.environ.get("PROBE_REPO_COMMIT", "unrecorded"))
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pii_redteam.training.rl_smoke import run_rl_smoke

        report = run_rl_smoke(config, repo_dir)
        print(json.dumps({"run_id": report["run_id"], "checks": report["checks"]}, indent=2))
    elif args.command == "train-detector":
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        repo_dir = config.get("repo_dir") or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        config.setdefault("repo_commit", os.environ.get("PROBE_REPO_COMMIT", "unrecorded"))
        from pii_redteam.train_detector import run_train_d1

        report = run_train_d1(config, repo_dir)
        print(json.dumps({"run_id": report["run_id"], "status": report["status"]}, indent=2))
    elif args.command == "rl-generate":
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config.setdefault("repo_commit", os.environ.get("PROBE_REPO_COMMIT", "unrecorded"))
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pii_redteam.training.rl_generate import run_rl_generate

        report = run_rl_generate(config, repo_dir)
        print(json.dumps({"run_id": report["run_id"], "counts": report["counts"]}, indent=2))
    elif args.command == "llm-attack":
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config.setdefault("repo_commit", os.environ.get("PROBE_REPO_COMMIT", "unrecorded"))
        repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from pii_redteam.training.llm_attack import run_llm_attack

        report = run_llm_attack(config, repo_dir)
        print(json.dumps({"run_id": report["run_id"], "stop": report["stop_reason"]}, indent=2))


if __name__ == "__main__":
    main()
