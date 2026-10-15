"""M07b prompt-variant comparison (P0 vs P1 vs P2 on the v2 bank).

GPU/CPU work lives here; pure functions live in `validity_v2.py` (strict v2
validity) and `prompt_v2.py` (variants + metrics). The M07a probe scorer is
deliberately NOT imported: D0 p_target/live-group stats are computed with a
local reference-pipeline replica and every such number is flagged
"pending_scorer_verification" until the scorer fix lands.

Raw outputs are ALL saved (valid + invalid, raw completion text) so the run can
be re-scored later without regenerating.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer

from pii_redteam.probe import _score_with_d0
from pii_redteam.probe_metrics import reward_metrics
from pii_redteam.prompt_v2 import (
    PROMPT_VARIANTS,
    comparison_metrics,
    hint_compliance,
    render_prompt_v2,
)
from pii_redteam.prompt_v2 import variant_threshold_report as threshold_report
from pii_redteam.validity_v2 import classify_sample_v2

SCORER_NOTE = "official probe scorer (probe._score_with_d0), verified vs the D0 eval pipeline by the M07a scorer-sanity regression (PR #5)"


def _sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _device_dtype() -> tuple[str, str, torch.dtype]:
    """fp16 on T4 (sm < 8.0), bf16 on L4 or better, fp32 on CPU; log the choice."""
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        gpu = torch.cuda.get_device_name(0)
        if major >= 8:
            return "cuda", gpu, torch.bfloat16
        return "cuda", gpu, torch.float16
    return "cpu", "cpu", torch.float32


def _chat_text(tok, prompt: str) -> str:
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.no_grad()
def _generate(
    tok,
    model,
    bank,
    *,
    variant,
    temperature,
    n,
    top_p,
    max_new_tokens,
    batch_prompts,
    log_prefix,
    device,
):
    """Batched sampling for one (variant, temperature) cell; left padding."""
    tok.padding_side = "left"
    end_id = tok.eos_token_id
    results = []
    for i in range(0, len(bank), batch_prompts):
        chunk = bank[i : i + batch_prompts]
        texts = [_chat_text(tok, render_prompt_v2(r, variant)) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": n,
        }
        gen = model.generate(**enc, **gen_kwargs)  # stub union
        n_prompt = enc["input_ids"].shape[1]
        for j, request in enumerate(chunk):
            for k in range(n):
                seq = gen[j * n + k][n_prompt:]
                finished = bool((seq == end_id).any())
                n_tok = (
                    int((seq == end_id).nonzero()[0].item() + 1) if finished else int(seq.shape[0])
                )
                results.append(
                    {
                        "request_id": request["request_id"],
                        "variant": variant,
                        "temperature": temperature,
                        "group_id": f"{request['request_id']}|{variant}|t{temperature}",
                        "sample_idx": k,
                        "name": request["task"]["entities"][0]["value"],
                        "context_type": request["task"]["context_type"],
                        "target_length": request["task"]["target_length"],
                        "raw_text": tok.decode(seq, skip_special_tokens=True),
                        "n_generated_tokens": n_tok,
                        "truncated": not finished,
                    }
                )
        if (i // batch_prompts) % 5 == 0:
            print(f"  {log_prefix}: {i + len(chunk)}/{len(bank)} requests")
    return results


def _panels(samples: list[dict], rng: random.Random) -> None:
    print("\n" + "=" * 78)
    print("MANUAL REVIEW PANELS (t=1.0 cells)")
    print("=" * 78)
    for variant in PROMPT_VARIANTS:
        valid = [
            s
            for s in samples
            if s["variant"] == variant and s["temperature"] == 1.0 and s["status"] == "valid"
        ]
        print(f"\n--- {variant}: 10 random valid samples (t=1.0) ---")
        for s in rng.sample(valid, min(10, len(valid))):
            p = f" p_target={s['p_target_mean']:.3f}" if s.get("p_target_mean") is not None else ""
            print(f"[{s['request_id']} #{s['sample_idx']}]{p}\n  {s['text'][:220]!r}")
        invalid = [
            s
            for s in samples
            if s["variant"] == variant and s["temperature"] == 1.0 and s["status"] != "valid"
        ]
        if invalid:
            print(f"--- {variant}: up to 10 invalid samples (t=1.0) ---")
            for s in invalid[:10]:
                print(
                    f"[{s['request_id']} #{s['sample_idx']}] {s['status']}\n  {s['raw_text'][:200]!r}"
                )


def run_compare(config: dict, repo_dir: str) -> dict:
    """Run the full P0/P1/P2 comparison. Returns the report dict (also on disk)."""
    started = datetime.datetime.now(datetime.UTC)
    rng = random.Random(config["seed"])
    gen_cfg = config["generation"]
    device, gpu, dtype = _device_dtype()

    # ---- bank lock (committed v2 bank; frozen v1 bank untouched)
    bank_path = os.path.join(repo_dir, config["bank"]["path"])
    actual = _sha256(bank_path)
    if actual != config["bank"]["sha256"]:
        raise SystemExit(f"v2 bank lock mismatch: {actual} != {config['bank']['sha256']}")
    bank = [json.loads(line) for line in open(bank_path, encoding="utf-8")]
    print(f"bank v2 locked: {len(bank)} requests, sha {actual[:12]}…")
    max_req = gen_cfg.get("max_requests")
    if max_req:
        bank = bank[:max_req]
        print(f"SMOKE: limited to {len(bank)} requests")

    # ---- models
    tok = AutoTokenizer.from_pretrained(config["model_id"])
    if tok.pad_token_id is None:  # ty: ignore[unresolved-attribute]
        tok.pad_token = tok.eos_token  # ty: ignore[invalid-assignment,unresolved-attribute]
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], dtype=dtype).to(device)  # ty: ignore[invalid-argument-type]  # stub union
    model.eval()
    print(f"model loaded: {config['model_id']} on {gpu} dtype={dtype}")

    detector_cfg = config.get("detector", {})
    d0 = d0_tok = None
    if detector_cfg.get("enabled", False):
        d0_sha = _sha256(os.path.join(detector_cfg["path"], "model.safetensors"))
        if d0_sha != detector_cfg["sha256"]:
            raise SystemExit(f"D0 lock mismatch: {d0_sha} != {detector_cfg['sha256']}")
        d0_tok = AutoTokenizer.from_pretrained(detector_cfg["path"])
        d0 = AutoModelForTokenClassification.from_pretrained(detector_cfg["path"]).to(device)
        d0.eval()
        print(f"frozen D0 loaded (sha {d0_sha[:12]}…) - scoring with {SCORER_NOTE}")
    env = {
        "device": device,
        "gpu": gpu,
        "dtype": str(dtype),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": sys.version.split()[0],
    }

    # ---- generation: variants x temperatures x requests x n
    variants = config.get("variants", list(PROMPT_VARIANTS))
    samples: list[dict] = []
    total_gens = len(variants) * len(gen_cfg["temperatures"]) * len(bank) * gen_cfg["n"]
    t0 = time.time()
    done = 0
    for variant in variants:
        for temp in gen_cfg["temperatures"]:
            torch.manual_seed(config["seed"] + variants.index(variant) * 100 + int(temp * 10))
            print(f"sampling {variant} @ t={temp} ...")
            cell = _generate(
                tok,
                model,
                bank,
                variant=variant,
                temperature=temp,
                n=gen_cfg["n"],
                top_p=gen_cfg["top_p"],
                max_new_tokens=gen_cfg["max_new_tokens"],
                batch_prompts=gen_cfg["batch_prompts"],
                log_prefix=f"{variant}|t={temp}",
                device=device,
            )
            samples += cell
            done += len(cell)
            elapsed = time.time() - t0
            print(
                f"  {done}/{total_gens} generations, {elapsed / 60:.1f} min elapsed, "
                f"projected {elapsed / done * total_gens / 60:.0f} min total"
            )
    gen_seconds = time.time() - t0

    # ---- v2 validity for every sample
    for s in samples:
        v2 = classify_sample_v2(s["raw_text"], s["name"], s["truncated"])
        s.update(v2)  # status, text, target_span (v2 definitions)

    # ---- D0 scoring (flagged; every raw output is kept regardless)
    scored = 0
    if d0 is not None:
        _score_with_d0(d0, d0_tok, samples, detector_cfg["max_length"])
        scored = sum(1 for s in samples if s.get("p_target_mean") is not None)
        for s in samples:
            if s["status"] == "valid":
                s["extra_person_flags"] = sum(
                    1
                    for sp in s.get("pred_spans", [])
                    if not (sp[0] < s["target_span"][1] and sp[1] > s["target_span"][0])
                )
                s["name_position"] = (
                    (s["target_span"][0] + s["target_span"][1]) / 2 / max(len(s["text"]), 1)
                )
            s.pop("pred_spans", None)

    # ---- metrics + thresholds per (variant, temperature)
    report: dict = {
        "kind": "prompt-variant-comparison",
        "role": (
            "M07b: can prompt changes alone make the base Qwen3.5-2B produce varied, "
            "realistic short messages? If no variant passes, trigger SFT v2."
        ),
        "started_utc": started.isoformat(),
        "environment": env,
        "config": config,
        "generations_total": len(samples),
        "gen_seconds": round(gen_seconds, 1),
    }
    per_cell: dict[str, dict] = {}
    for variant in variants:
        for temp in gen_cfg["temperatures"]:
            cell_samples = [
                s for s in samples if s["variant"] == variant and s["temperature"] == temp
            ]
            metrics = comparison_metrics(cell_samples, group_size=gen_cfg["n"])
            metrics["thresholds"] = threshold_report(metrics, config["thresholds"])
            if d0 is not None:
                clip = config.get("rewards", {}).get("logit_clip", 10.0)
                valid_for_reward = [
                    s
                    for s in cell_samples
                    if s["status"] == "valid" and s.get("p_target_mean") is not None
                ]
                for s in valid_for_reward:
                    p = min(max(s["p_target_mean"], 1e-6), 1 - 1e-6)
                    s["r_prob"] = 1.0 - s["p_target_mean"]
                    s["r_logit"] = max(-clip, min(clip, -math.log(p / (1 - p))))
                metrics["reward"] = reward_metrics(
                    [
                        {
                            "group_id": s["group_id"],
                            "status": s["status"],
                            "p_target_mean": s.get("p_target_mean"),
                            "p_target_min": s.get("p_target_min"),
                            "outcome": s.get("outcome", "invalid"),
                            "n_chars": len(s["text"] or ""),
                            "name_position": s.get("name_position", 0.0),
                            "extra_person_flags": s.get("extra_person_flags", 0),
                            "r_prob": s.get("r_prob", -1.0),
                            "r_logit": s.get("r_logit", -1.0),
                        }
                        for s in cell_samples
                    ],
                    clip,
                )
                metrics["scorer_note"] = SCORER_NOTE
            per_cell[f"{variant}|t{temp}"] = metrics
    report["per_cell"] = per_cell

    # ---- pass criteria + variant pick
    def cell_pass(m: dict) -> bool:
        return all(c["ok"] for c in m["thresholds"].values())

    passed = {k: cell_pass(m) for k, m in per_cell.items()}
    report["pass_by_cell"] = passed
    hints = {r["request_id"]: r["task"].get("name_position") for r in bank}
    report["position_compliance_by_hint"] = hint_compliance(samples, hints)
    best = None
    for key, ok in passed.items():
        if not ok:
            continue
        m = per_cell[key]
        rank = (m["validity_rate"], -m["name_at_start_share"], -m["top10_opening_share"])
        if best is None or rank > best[0]:
            best = (rank, key)
    report["best_cell"] = best[1] if best else None
    if best:
        report["recommendation"] = (
            f"variant cell {best[1]} passes all pre-committed criteria - candidate prompt for M06/M08"
        )
    else:
        report["recommendation"] = (
            "no variant x temperature cell passes all criteria - prompting alone is NOT "
            "enough; this is the agreed trigger for SFT v2 on Spark-written short data "
            "(no further prompt hacking)"
        )
    print("\nPASS BY CELL:", json.dumps(passed))
    print("RECOMMENDATION:", report["recommendation"])

    # ---- artifacts: run dir + raw samples (re-scoreable)
    run_id = datetime.datetime.now(datetime.UTC).strftime(
        f"{config['run_id_prefix']}-%Y%m%d-%H%M%S"
    )
    work_dir = config["work_dir"]
    if not os.path.isabs(work_dir):
        work_dir = os.path.join(repo_dir, work_dir)
    work_dir = os.path.join(work_dir, run_id)
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "samples_raw.jsonl"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    report["run_id"] = run_id
    report["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    report["samples_raw_path"] = os.path.join(work_dir, "samples_raw.jsonl")
    report["d0_scored_samples"] = scored
    report["experiments_entry"] = {
        "run_id": run_id,
        "kind": "prompt-variant-comparison",
        "milestone": "M07b",
        "executed": report["finished_utc"][:10],
        "where": f"{gpu}-{device}",
        "repo_commit": config.get("repo_commit", "recorded-by-notebook"),
        "environment": env,
        "artifacts": f"{work_dir} (samples_raw.jsonl, prompt_compare_report.json)",
        "metrics": {
            k: {
                "validity_rate": m["validity_rate"],
                "name_at_start_share": m["name_at_start_share"],
                "top10_opening_share": m["top10_opening_share"],
                "unique_per_group_median": m["unique_per_group_median"],
                "failure_breakdown": m["failure_breakdown"],
            }
            for k, m in per_cell.items()
        },
        "outcome": report["recommendation"],
    }
    with open(os.path.join(work_dir, "prompt_compare_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _panels(samples, rng)
    print(f"\nrun_id: {run_id}\nreport: {os.path.join(work_dir, 'prompt_compare_report.json')}")
    return report
