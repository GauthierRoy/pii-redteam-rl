"""M08.3 RL smoke run: TRL GRPO + LoRA on the generator against the frozen D1 reward.

Capped by design (step cap, wall-clock cap, checkpoints to the work dir). It measures, it
does not claim a result: reward signal on real generations before any update (how sparse
detector misses are per group), training health (finite loss, non-zero gradients, memory,
seconds per step) and a pre/post sample comparison on held-out requests with every raw
output saved for manual fluency/diversity review. Colab-only in practice; the logic lives
here so the notebook stays a thin launcher.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import GRPOTrainer

from ..prompt_compare import _device_dtype, _generate
from ..requests import load_person_name_pool
from ..run_io import write_jsonl
from .d1_scorer import D1Scorer
from .grpo_compat import build_grpo_config
from .rl_bank import bank_names, build_rl_requests
from .rl_dataset import RL_PROMPT_VARIANT, rl_rows
from .rl_reward import D1Reward

DETECTOR_MAX_LENGTH = 512


class WallClockStop(TrainerCallback):
    """Hard cap: stop training (and save) once `max_minutes` of wall clock have elapsed."""

    def __init__(self, max_minutes: float) -> None:
        self.deadline = time.time() + max_minutes * 60
        self.triggered = False

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() > self.deadline:
            self.triggered = True
            control.should_save = True
            control.should_training_stop = True
        return control


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def resolve(path: str, repo_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_dir, path)


def save_report(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def build_requests(config: dict, repo_dir: str) -> tuple[list[dict], list[dict], dict]:
    """(train requests, held-out signal requests, provenance) with the final-eval lock asserted."""
    splits_dir = resolve(config["splits_dir"], repo_dir)
    manifest_path = os.path.join(splits_dir, "final_eval_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        final_eval_sha = json.load(f)["final_eval_sha256"]
    if final_eval_sha != config["locks"]["final_eval_sha256"]:
        raise SystemExit(f"final-eval lock mismatch: {final_eval_sha}")
    pool, pool_provenance = load_person_name_pool(os.path.join(splits_dir, "name_pools.json"))
    excluded: set[str] = set()
    for path in config["bank"]["exclude_banks"]:
        excluded |= bank_names(resolve(path, repo_dir))
    n_signal, n_train = config["bank"]["n_signal"], config["bank"]["n_train"]
    requests = build_rl_requests(
        pool,
        n=n_signal + n_train,
        seed=config["bank"]["seed"],
        exclude=frozenset(excluded),
    )
    provenance = {
        "pool": pool_provenance,
        "excluded_names": len(excluded),
        "n_signal": n_signal,
        "n_train": n_train,
        "seed": config["bank"]["seed"],
    }
    return requests[n_signal:], requests[:n_signal], provenance


def load_policy(config: dict, dtype: torch.dtype):
    """Qwen policy in 4-bit nf4 (QLoRA, as the M05 SFT run) or plain dtype for CPU checks."""
    tok = AutoTokenizer.from_pretrained(config["model_id"])
    if tok.pad_token_id is None:  # ty: ignore[unresolved-attribute]
        tok.pad_token = tok.eos_token  # ty: ignore[invalid-assignment,unresolved-attribute]
    if config["model"]["quantize_4bit"]:
        model = AutoModelForCausalLM.from_pretrained(
            config["model_id"],
            dtype=dtype,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            ),
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(config["model_id"], dtype=dtype)
    return tok, model


def load_scorer(config: dict, device: str) -> D1Scorer:
    detector = config["detector"]
    weights_sha = file_sha256(os.path.join(detector["path"], "model.safetensors"))
    if weights_sha != detector["sha256"]:
        raise SystemExit(f"detector lock mismatch: {weights_sha} != {detector['sha256']}")
    model = AutoModelForTokenClassification.from_pretrained(detector["path"]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(detector["path"])
    return D1Scorer(model, tokenizer, DETECTOR_MAX_LENGTH)


def sample_and_score(tok, model, requests, config: dict, reward: D1Reward, device: str, label: str):
    """Sample G rollouts per request with the RL sampling settings and score them."""
    sampling = config["sampling"]
    samples = _generate(
        tok,
        model,
        requests,
        variant=RL_PROMPT_VARIANT,
        temperature=sampling["temperature"],
        n=config["grpo"]["num_generations"],
        top_p=sampling["top_p"],
        max_new_tokens=config["grpo"]["max_completion_length"],
        batch_prompts=sampling["batch_prompts"],
        log_prefix=label,
        device=device,
    )
    extras: dict = {}
    metrics: dict = {}
    eos = tok.eos_token_id
    rewards = reward(
        [s["raw_text"] for s in samples],
        [s["name"] for s in samples],
        completion_ids=[[] if s["truncated"] else [eos] for s in samples],
        log_extra=extras.__setitem__,
        log_metric=metrics.__setitem__,
    )
    for sample, r, status, p in zip(
        samples, rewards, extras["status"], extras["p_target_mean"], strict=True
    ):
        sample.update({"reward": r, "status": status, "p_target_mean": p})
    return samples, metrics


def signal_summary(samples: list[dict], metrics: dict) -> dict:
    """Group-level view of the reward signal: is there anything for GRPO to learn from?"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[sample["request_id"]].append(sample)
    group_std = [statistics.pstdev([s["reward"] for s in g]) for g in groups.values()]
    group_has_miss = [any(s["reward"] > 0.5 for s in g) for g in groups.values()]
    group_unique = [len({s["raw_text"] for s in g}) / len(g) for g in groups.values()]
    valid = [s for s in samples if s["status"] == "valid"]
    return {
        "samples": len(samples),
        "groups": len(groups),
        **metrics,
        "groups_with_detector_miss": sum(group_has_miss) / len(groups),
        "mean_group_reward_std": statistics.mean(group_std),
        "mean_valid_reward": statistics.mean(s["reward"] for s in valid) if valid else None,
        "mean_unique_text_share_per_group": statistics.mean(group_unique),
        "mean_valid_words": statistics.mean(len(s["text"].split()) for s in valid)
        if valid
        else None,
    }


def smoke_checks(config: dict, summary: dict) -> dict[str, dict]:
    """Health checks on the finished run (not learning claims)."""
    thresholds = config["thresholds"]
    history = [e for e in summary["log_history"] if "loss" in e]
    finite = all(math.isfinite(e[k]) for e in history for k in ("loss", "grad_norm") if k in e)
    pre, post = summary["pre"], summary.get("post")
    checks = {
        "pre-RL valid rate": (
            pre["valid_rate"] >= thresholds["pre_valid_rate_min"],
            f"{pre['valid_rate']:.3f} (min {thresholds['pre_valid_rate_min']})",
        ),
        "training reached the step cap": (
            summary["steps_done"] == config["grpo"]["max_steps"] and not summary["wall_clock_stop"],
            f"{summary['steps_done']}/{config['grpo']['max_steps']} steps, "
            f"wall-clock stop={summary['wall_clock_stop']}",
        ),
        "loss and grad_norm finite": (finite and bool(history), f"{len(history)} logged steps"),
        "some update was non-zero": (
            any(e.get("grad_norm", 0.0) > 0.0 for e in history),
            f"max grad_norm {max((e.get('grad_norm', 0.0) for e in history), default=0.0):.3g}",
        ),
        "adapter saved": (os.path.exists(summary["adapter_path"]), summary["adapter_path"]),
        "peak GPU memory": (
            summary["peak_memory_gb"] <= thresholds["peak_memory_fraction_max"] * summary["gpu_gb"],
            f"{summary['peak_memory_gb']:.1f} of {summary['gpu_gb']:.1f} GB",
        ),
    }
    if post is not None:
        drop = pre["valid_rate"] - post["valid_rate"]
        checks["valid rate did not collapse"] = (
            drop <= thresholds["valid_rate_drop_max"],
            f"{pre['valid_rate']:.3f} -> {post['valid_rate']:.3f}",
        )
    return {name: {"ok": ok, "detail": detail} for name, (ok, detail) in checks.items()}


def run_rl_smoke(config: dict, repo_dir: str) -> dict:
    started = datetime.datetime.now(datetime.UTC)
    run_id = started.strftime(f"{config['run_id_prefix']}-%Y%m%d-%H%M%S")
    work_dir = os.path.join(resolve(config["work_dir"], repo_dir), run_id)
    os.makedirs(work_dir, exist_ok=True)
    report_path = os.path.join(work_dir, "rl_smoke_report.json")
    device, gpu, dtype = _device_dtype()
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if device == "cuda" else 0.0
    report: dict = {
        "kind": "rl-smoke",
        "run_id": run_id,
        "started_utc": started.isoformat(),
        "config": config,
        "environment": {
            "gpu": gpu,
            "dtype": str(dtype),
            "torch": torch.__version__,
            "python": sys.version.split()[0],
        },
    }

    train_requests, signal_requests, bank_provenance = build_requests(config, repo_dir)
    write_jsonl(os.path.join(work_dir, "requests_train.jsonl"), train_requests)
    write_jsonl(os.path.join(work_dir, "requests_signal.jsonl"), signal_requests)
    report["bank"] = bank_provenance
    save_report(report, report_path)

    tok, model = load_policy(config, dtype)
    if device == "cuda" and not config["model"]["quantize_4bit"]:
        model = model.to(device)
    scorer = load_scorer(config, device)
    reward = D1Reward(scorer, tok.eos_token_id)
    model.eval()
    torch.manual_seed(config["seed"])

    pre_samples, pre_metrics = sample_and_score(
        tok, model, signal_requests, config, reward, device, "pre-RL"
    )
    write_jsonl(os.path.join(work_dir, "samples_pre.jsonl"), pre_samples)
    pre = signal_summary(pre_samples, pre_metrics)
    report["pre"] = pre
    print("PRE-RL signal:", json.dumps(pre, indent=2))
    save_report(report, report_path)

    grpo = dict(config["grpo"])
    adapter_path = os.path.join(work_dir, "adapter")
    args = build_grpo_config(
        output_dir=os.path.join(work_dir, "checkpoints"),
        seed=config["seed"],
        temperature=config["sampling"]["temperature"],
        top_p=config["sampling"]["top_p"],
        chat_template_kwargs={"enable_thinking": False},
        report_to="none",
        logging_steps=1,
        fp16=dtype == torch.float16 and device == "cuda",
        bf16=dtype == torch.bfloat16 and device == "cuda",
        use_cpu=device == "cpu",
        **grpo,
    )
    lora = config["lora"]
    stopper = WallClockStop(config["max_minutes"])
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=args,
        train_dataset=Dataset.from_list(rl_rows(train_requests)),
        processing_class=tok,
        peft_config=LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        ),
        callbacks=[stopper],
    )
    started_training = time.time()
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    train_seconds = time.time() - started_training
    trainer.save_model(adapter_path)
    steps_done = trainer.state.global_step
    report["training"] = {
        "steps_done": steps_done,
        "train_seconds": round(train_seconds, 1),
        "sec_per_step": round(train_seconds / max(steps_done, 1), 2),
        "wall_clock_stop": stopper.triggered,
        "log_history": trainer.state.log_history,
        "adapter_path": adapter_path,
    }
    save_report(report, report_path)

    model.eval()
    model.config.use_cache = True
    post_samples, post_metrics = sample_and_score(
        tok, trainer.model, signal_requests, config, reward, device, "post-RL"
    )
    write_jsonl(os.path.join(work_dir, "samples_post.jsonl"), post_samples)
    post = signal_summary(post_samples, post_metrics)
    report["post"] = post
    print("POST-RL signal:", json.dumps(post, indent=2))

    summary = {
        "pre": pre,
        "post": post,
        "log_history": trainer.state.log_history,
        "steps_done": steps_done,
        "wall_clock_stop": stopper.triggered,
        "adapter_path": adapter_path,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0,
        "gpu_gb": gpu_gb,
    }
    report["checks"] = smoke_checks(config, summary)
    report["peak_memory_gb"] = summary["peak_memory_gb"]
    report["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    save_report(report, report_path)
    print("=" * 70)
    for name, result in report["checks"].items():
        print(f"[{'PASS' if result['ok'] else 'FAIL'}] {name}: {result['detail']}")
    print(f"report: {report_path}")
    return report
