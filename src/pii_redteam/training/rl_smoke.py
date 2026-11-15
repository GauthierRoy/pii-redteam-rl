"""M08.3 RL smoke run: TRL GRPO + LoRA on the attacker against the frozen target detector reward.

Capped by design (step cap, wall-clock cap, checkpoints to the work dir). It measures, it
does not claim a result: reward signal on real generations before any update (how sparse
detector misses are per group), training health (finite loss, non-zero gradients, memory,
seconds per step) and a pre/post sample comparison on held-out requests with every raw
output saved for manual fluency/diversity review. Colab-only in practice; the logic lives
here so the notebook stays a thin launcher.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import json
import math
import os
import statistics
import sys
import time

import torch
import trl.generation.vllm_generation as trl_vllm
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
from .detector_scorer import DetectorScorer
from .grpo_compat import build_grpo_config
from .opening_share import OpeningShare
from .person_judge import MISTRAL_CHAT_URL, PersonJudge, http_transport
from .rl_bank import bank_names, build_rl_requests
from .rl_dataset import RL_PROMPT_VARIANT, rl_rows
from .rl_reward import TargetDetectorReward
from .rl_signal import signal_summary

DETECTOR_MAX_LENGTH = 512
MULTIMODAL_QWEN_PREFIXES = ("model.language_model.", "model.visual.")
TRL_FIX_PARAM_NAME = trl_vllm.VLLMGeneration._fix_param_name_to_vllm


class WallClockStop(TrainerCallback):
    """Stop training (and save) after `max_minutes` of wall clock, or at `stop_at_step`.

    `stop_at_step` ends a run early while `max_steps` keeps defining the learning-rate
    schedule, so its first steps match a longer run with the same `max_steps`.
    """

    def __init__(self, max_minutes: float, stop_at_step: int | None = None) -> None:
        self.deadline = time.time() + max_minutes * 60
        self.stop_at_step = stop_at_step
        self.triggered = False

    def on_step_end(self, args, state, control, **kwargs):
        out_of_time = time.time() > self.deadline
        if out_of_time:
            self.triggered = True
        if out_of_time or (self.stop_at_step and state.global_step >= self.stop_at_step):
            control.should_save = True
            control.should_training_stop = True
        return control


class TimedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer that records the seconds spent generating completions (vLLM weight sync included)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generation_seconds: list[float] = []

    def _generate_single_turn(self, *args, **kwargs):
        started = time.perf_counter()
        output = super()._generate_single_turn(*args, **kwargs)
        self.generation_seconds.append(time.perf_counter() - started)
        return output


def to_multimodal_qwen_name(name: str) -> str:
    """Text-only Qwen3.5 parameter name -> the multimodal checkpoint name vLLM's loader expects."""
    if name.startswith("model.") and not name.startswith(MULTIMODAL_QWEN_PREFIXES):
        return "model.language_model." + name.removeprefix("model.")
    return name


def fix_param_name_for_multimodal_vllm(self, name: str, extra_prefixes: list[str] | None = None):
    return to_multimodal_qwen_name(TRL_FIX_PARAM_NAME(self, name, extra_prefixes))


def align_vllm_names_with_multimodal_checkpoint() -> None:
    """Sync a text-only attacker into vLLM, which only registers Qwen3.5 as a multimodal model."""
    trl_vllm.VLLMGeneration._fix_param_name_to_vllm = fix_param_name_for_multimodal_vllm


def restrict_vllm_to_language_model() -> None:
    """Make TRL's colocated vLLM skip the vision tower: TRL exposes no engine-argument pass-through."""
    trl_vllm.LLM = functools.partial(trl_vllm.LLM, language_model_only=True)


def timing_summary(step_seconds: list[float], generation_seconds: list[float]) -> dict:
    return {
        name: {
            "median": round(statistics.median(values), 2) if values else None,
            "per_step": [round(value, 2) for value in values],
        }
        for name, values in (("step", step_seconds), ("generation", generation_seconds))
    }


def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def resolve(path: str, repo_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(repo_dir, path)


def upcast_trainable_params_to_fp32(model) -> int:
    """TRL casts QLoRA adapters to bf16; the fp16 grad scaler (T4) cannot unscale bf16 grads."""
    changed = 0
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            changed += 1
    return changed


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


def load_attacker(config: dict, dtype: torch.dtype):
    """Qwen attacker in 4-bit nf4 (QLoRA, as the M05 SFT run) or plain dtype for CPU checks."""
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


def load_scorer(config: dict, device: str) -> DetectorScorer:
    detector = config["detector"]
    weights_sha = file_sha256(os.path.join(detector["path"], "model.safetensors"))
    if weights_sha != detector["sha256"]:
        raise SystemExit(f"detector lock mismatch: {weights_sha} != {detector['sha256']}")
    model = AutoModelForTokenClassification.from_pretrained(detector["path"]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(detector["path"])
    return DetectorScorer(model, tokenizer, DETECTOR_MAX_LENGTH)


def opening_share_guard(config: dict, window: int | None = None) -> OpeningShare | None:
    """The collapse guard configured under `reward.opening_share`, or None when absent."""
    settings = (config.get("reward") or {}).get("opening_share")
    if not settings:
        return None
    return OpeningShare(
        full_reward_until=settings["full_reward_until"],
        zero_reward_from=settings["zero_reward_from"],
        window=window or settings["window"],
        min_other_texts=settings["min_other_texts"],
    )


def person_judge(config: dict) -> PersonJudge | None:
    """The judge configured under `reward.person_judge`, or None when absent."""
    settings = (config.get("reward") or {}).get("person_judge")
    if not settings:
        return None
    api_key = os.environ.get(settings["api_key_env"])
    if not api_key:
        raise SystemExit(f"person judge needs the {settings['api_key_env']} environment variable")
    return PersonJudge(
        model=settings["model"],
        post=http_transport(api_key, settings.get("url", MISTRAL_CHAT_URL), settings["timeout_s"]),
        max_workers=settings["max_workers"],
    )


def sample_and_score(
    tok,
    model,
    requests,
    config: dict,
    scorer: DetectorScorer,
    judge: PersonJudge | None,
    device: str,
    label: str,
):
    """Sample G rollouts per request with the RL sampling settings and score them.

    Uses its own reward, so the opening shares are measured within this sample set only.
    """
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
    reward = TargetDetectorReward(
        scorer, eos, opening_share_guard(config, window=len(samples)), judge
    )
    rewards = reward(
        [s["raw_text"] for s in samples],
        [s["name"] for s in samples],
        completion_ids=[[] if s["truncated"] else [eos] for s in samples],
        request_id=[s["request_id"] for s in samples],
        log_extra=extras.__setitem__,
        log_metric=metrics.__setitem__,
    )
    for sample, r, status, p, share in zip(
        samples,
        rewards,
        extras["status"],
        extras["p_target_mean"],
        extras["opening_share"],
        strict=True,
    ):
        sample.update({"reward": r, "status": status, "p_target_mean": p, "opening_share": share})
    return samples, metrics


def planned_steps(config: dict) -> int:
    return config.get("stop_at_step") or config["grpo"]["max_steps"]


def smoke_checks(config: dict, summary: dict) -> dict[str, dict]:
    """Health checks on the finished run (not learning claims)."""
    thresholds = config["thresholds"]
    history = [e for e in summary["log_history"] if "loss" in e]
    grad_norms = [e["grad_norm"] for e in history if "grad_norm" in e]
    skipped = sum(1 for g in grad_norms if not math.isfinite(g))
    applied = [g for g in grad_norms if math.isfinite(g) and g > 0.0]
    skipped_max = thresholds["skipped_update_fraction_max"]
    pre, post = summary["pre"], summary.get("post")
    checks = {
        "pre-RL valid rate": (
            pre["valid_rate"] >= thresholds["pre_valid_rate_min"],
            f"{pre['valid_rate']:.3f} (min {thresholds['pre_valid_rate_min']})",
        ),
        "training reached the step cap": (
            summary["steps_done"] == planned_steps(config) and not summary["wall_clock_stop"],
            f"{summary['steps_done']}/{planned_steps(config)} steps, "
            f"wall-clock stop={summary['wall_clock_stop']}",
        ),
        "loss finite": (
            bool(history) and all(math.isfinite(e["loss"]) for e in history),
            f"{len(history)} logged steps",
        ),
        "most updates were applied": (
            bool(history) and skipped <= skipped_max * len(history) and bool(applied),
            f"{len(applied)} steps with finite non-zero grad_norm (max {max(applied, default=0.0):.3g}), "
            f"{skipped} non-finite = skipped by the fp16 grad scaler (max {skipped_max:.0%})",
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

    tok, model = load_attacker(config, dtype)
    if device == "cuda" and not config["model"]["quantize_4bit"]:
        model = model.to(device)
    scorer = load_scorer(config, device)
    judge = person_judge(config)
    reward = TargetDetectorReward(scorer, tok.eos_token_id, opening_share_guard(config), judge)
    model.eval()
    torch.manual_seed(config["seed"])

    pre_samples, pre_metrics = sample_and_score(
        tok, model, signal_requests, config, scorer, judge, device, "pre-RL"
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
    if config.get("vllm_multimodal_names"):
        align_vllm_names_with_multimodal_checkpoint()
    if config.get("vllm_language_model_only"):
        restrict_vllm_to_language_model()
    lora = config["lora"]
    stopper = WallClockStop(config["max_minutes"], config.get("stop_at_step"))
    trainer = TimedGRPOTrainer(
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
    if device == "cuda" and dtype == torch.float16:
        print(f"fp16 run: {upcast_trainable_params_to_fp32(model)} trainable tensors cast to fp32")
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
        "timing": timing_summary(
            [e["step_time"] for e in trainer.state.log_history if "step_time" in e],
            trainer.generation_seconds,
        ),
        "log_history": trainer.state.log_history,
        "adapter_path": adapter_path,
    }
    save_report(report, report_path)

    model.eval()
    model.config.use_cache = True
    post_samples, post_metrics = sample_and_score(
        tok, trainer.model, signal_requests, config, scorer, judge, device, "post-RL"
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
