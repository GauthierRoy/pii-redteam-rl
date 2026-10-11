"""M07a base-sampling probe: generation + frozen-D0 reward signal (thin CLI target).

Implements the M07a notebook spec: sample the BASE Qwen3.5-2B over the frozen request
bank at two temperatures plus a greedy reference, validate each sample against the
output contract, score every valid sample with frozen D0 (spans + p_target), and
report diversity + reward-signal metrics against pre-committed go/no-go thresholds.

GPU work lives here; pure metric math lives in `probe_metrics.py`.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer

from pii_redteam.detector import bio_labels_from_spans, spans_from_bio_labels, trim_span_whitespace
from pii_redteam.probe_metrics import (
    PROMPT_MARKER,
    classify_sample,
    diversity_metrics,
    outcome_for_sample,
    parse_json_tolerant,
    reward_metrics,
    reward_variants,
    threshold_report,
)
from pii_redteam.requests import render_prompt


def _sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _device_dtype() -> tuple[str, torch.dtype]:
    """fp16 on T4 (sm < 8.0), bf16 on L4 or better; log the choice."""
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        gpu = torch.cuda.get_device_name(0)
        if major >= 8:
            return gpu, torch.bfloat16
        return gpu, torch.float16
    return "cpu", torch.float32


def _load_bank(config: dict, repo_dir: str) -> list[dict]:
    """Rebuild the frozen bank on demand and assert both locks (spec 1)."""
    bank_cfg = config["bank"]
    if config.get("rebuild_locks", False):
        env = dict(os.environ, PYTHONPATH=os.path.join(repo_dir, "src"))
        subprocess.run(
            [
                sys.executable,
                "scripts/carve_splits.py",
                "--cache-dir",
                config["cache_dir"],
                "--out",
                os.path.join(repo_dir, "artifacts", "splits"),
            ],
            cwd=repo_dir,
            env=env,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/build_request_bank.py",
                "--n",
                str(bank_cfg["n"]),
                "--seed",
                str(bank_cfg["seed"]),
            ],
            cwd=repo_dir,
            env=env,
            check=True,
        )
    bank_path = os.path.join(
        repo_dir,
        "artifacts",
        "request_bank",
        f"sampler_seed{bank_cfg['seed']}_n{bank_cfg['n']}.jsonl",
    )
    actual = _sha256(bank_path)
    if actual != bank_cfg["sha256"]:
        raise SystemExit(f"bank lock mismatch: {actual} != {bank_cfg['sha256']}")
    manifest = json.load(
        open(
            os.path.join(repo_dir, "artifacts", "splits", "final_eval_manifest.json"),
            encoding="utf-8",
        )
    )
    if manifest["final_eval_sha256"] != config["locks"]["final_eval_sha256"]:
        raise SystemExit("final-eval lock mismatch - refusing to continue.")
    return [json.loads(line) for line in open(bank_path, encoding="utf-8")]


@torch.no_grad()
def _generate(
    tok, model, requests, *, temperature, n, top_p, max_new_tokens, batch_prompts, log_prefix
):
    """Batched sampling (or greedy when temperature is None); left padding."""
    tok.padding_side = "left"
    end_id = tok.eos_token_id
    results = []
    for i in range(0, len(requests), batch_prompts):
        chunk = requests[i : i + batch_prompts]
        texts = [_chat_text(tok, r) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
        if temperature is not None:
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "num_return_sequences": n,
            }
        gen = model.generate(**enc, **gen_kwargs)  # ty: ignore[invalid-argument-type]  # stub union
        n_prompt = enc["input_ids"].shape[1]
        for j, request in enumerate(chunk):
            reps = n if temperature is not None else 1
            for k in range(reps):
                seq = gen[j * reps + k][n_prompt:]
                finished = bool((seq == end_id).any())
                n_tok = (
                    int((seq == end_id).nonzero()[0].item() + 1) if finished else int(seq.shape[0])
                )
                results.append(
                    {
                        "request_id": request["request_id"],
                        "name": request["task"]["entities"][0]["value"],
                        "group_id": f"{request['request_id']}|t{temperature}",
                        "sample_idx": k,
                        "temperature": temperature,
                        "raw_text": tok.decode(seq, skip_special_tokens=True),
                        "n_generated_tokens": n_tok,
                        "truncated": not finished,
                    }
                )
        if (i // batch_prompts) % 3 == 0:
            print(f"  {log_prefix}: {i + len(chunk)}/{len(requests)} requests")
    return results


def _chat_text(tok, request):
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": render_prompt(request)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tok.apply_chat_template(
            [{"role": "user", "content": render_prompt(request)}],
            tokenize=False,
            add_generation_prompt=True,
        )


config_seed_holder = [20260920]


@torch.no_grad()
def _score_with_d0(d0, d0_tok, samples: list[dict], max_length: int) -> dict:
    """Per-sample D0 spans + p_target over target tokens; batched."""
    to_score = [s for s in samples if s["status"] == "valid"]
    spans_all = [None] * len(samples)
    index = {id(s): i for i, s in enumerate(samples)}
    for i in range(0, len(to_score), 32):
        chunk = to_score[i : i + 32]
        enc = d0_tok(
            [s["text"] for s in chunk],
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        input_ids = torch.tensor(enc["input_ids"]).to(d0.device)
        attn = torch.tensor(enc["attention_mask"]).to(d0.device)
        logits = d0(input_ids=input_ids, attention_mask=attn).logits.float()
        probs = torch.softmax(logits, dim=-1)
        for j, s in enumerate(chunk):
            offsets = enc["offset_mapping"][j]
            labels = [d0.config.id2label[p] for p in logits[j].argmax(-1).tolist()]
            pred = [
                (sp["start"], sp["end"])
                for sp in trim_span_whitespace(s["text"], spans_from_bio_labels(offsets, labels))
            ]
            gold_labels = bio_labels_from_spans(
                offsets,
                [{"label": "PERSON", "start": s["target_span"][0], "end": s["target_span"][1]}],
            )
            target_idx = [t for t, lab in enumerate(gold_labels) if lab != "O"]
            if not target_idx:
                s["d0_truncated_target"] = True
                s["outcome"] = "d0_truncated"
                s["p_target_mean"] = None
                s["p_target_min"] = None
                s["pred_spans"] = pred
                spans_all[index[id(s)]] = s
                continue
            label2id = {lab: i for i, lab in d0.config.id2label.items()}
            p_vals = [float(probs[j, t, label2id[gold_labels[t]]]) for t in target_idx]
            s["d0_truncated_target"] = False
            s["pred_spans"] = pred
            s["p_target_mean"] = sum(p_vals) / len(p_vals)
            s["p_target_min"] = min(p_vals)
            s["outcome"] = outcome_for_sample(s["target_span"], pred)
            spans_all[index[id(s)]] = s
    return {"scored": len(to_score)}


def _finalize_samples(samples: list[dict], clip: float) -> None:
    """Attach outcome, rewards, position, flags; invalids get the invalid reward."""
    for s in samples:
        s["n_chars"] = len(s["text"]) if s["text"] else 0
        if s["status"] == "valid":
            pos_frac = (s["target_span"][0] + s["target_span"][1]) / 2 / max(len(s["text"]), 1)
            s["name_position"] = pos_frac
            s["extra_person_flags"] = sum(
                1
                for sp in s.get("pred_spans", [])
                if not (sp[0] < s["target_span"][1] and sp[1] > s["target_span"][0])
            )
            rv = reward_variants(s["p_target_mean"])
            s["r_prob"], s["r_logit"] = rv["r_prob"], rv["r_logit"]
            s["outcome"] = s["d0"]["outcome"]
        else:
            s["r_prob"] = s["r_logit"] = -1.0
            s["outcome"] = "invalid"
            s["extra_person_flags"] = 0
            s["name_position"] = 0.0


def _panels(samples: list[dict], rng: random.Random) -> None:
    valid = [s for s in samples if s["status"] == "valid"]
    invalid = [s for s in samples if s["status"] != "valid"]
    print("\n" + "=" * 78)
    print("MANUAL REVIEW PANELS")
    print("=" * 78)
    print("\n--- 20 random valid samples ---")
    for s in rng.sample(valid, min(20, len(valid))):
        print(
            f"[{s['group_id']} #{s['sample_idx']}] p_target={s['p_target_mean']:.3f} "
            f"outcome={s['outcome']}\n  {s['text'][:220]!r}"
        )
    print("\n--- 20 lowest-p_target valid samples (genuinely hard, or weird?) ---")
    for s in sorted(valid, key=lambda x: x["d0"]["p_target_mean"])[:20]:
        print(
            f"[{s['group_id']} #{s['sample_idx']}] p_target={s['p_target_mean']:.3f} "
            f"outcome={s['outcome']}\n  {s['text'][:220]!r}"
        )
    print("\n--- 10 invalid samples ---")
    for s in invalid[:10]:
        print(f"[{s['group_id']} #{s['sample_idx']}] {s['status']}\n  {s['raw_text'][:220]!r}")
    print("\n--- 5 full groups per temperature ---")
    by_temp: dict[float, dict[str, list[dict]]] = {}
    for s in valid:
        by_temp.setdefault(s["temperature"], {}).setdefault(s["request_id"], []).append(s)
    for temp, groups in by_temp.items():
        for req_id, members in list(groups.items())[:5]:
            print(f"\n group {req_id} @t={temp} ({len(members)} valid samples)")
            for s in members[:8]:
                print(f"  #{s['sample_idx']} p={s['p_target_mean']:.3f} {s['text'][:120]!r}")


def _plot(work_dir: str, samples: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("plots skipped: matplotlib unavailable")
        return
    plots = os.path.join(work_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    valid = [s for s in samples if s["status"] == "valid"]
    temps = sorted({s["temperature"] for s in samples if s["temperature"] is not None})
    for temp in temps:
        tv = [s for s in valid if s["temperature"] == temp]
        if not tv:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist([s["p_target_mean"] for s in tv], bins=30)
        axes[0].set_title(f"p_target_mean @ t={temp}")
        stds = []
        groups: dict[str, list[float]] = {}
        for s in tv:
            groups.setdefault(s["request_id"], []).append(s["r_prob"])
        for vals in groups.values():
            mean = sum(vals) / len(vals)
            stds.append((sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5)
        axes[1].hist(stds, bins=20)
        axes[1].set_title(f"within-group r_prob std @ t={temp}")
        axes[2].hist([s["n_chars"] for s in tv], bins=30)
        axes[2].set_title(f"length (chars) @ t={temp}")
        fig.tight_layout()
        fig.savefig(os.path.join(plots, f"t{temp}.png"), dpi=110)
        plt.close(fig)
    print(f"plots -> {plots}")


def run_probe(config: dict, repo_dir: str) -> dict:
    """Full M07a probe per the spec. Returns the report dict (also written to disk)."""
    started = datetime.datetime.now(datetime.UTC)
    rng = random.Random(config["seed"])
    config_seed_holder[0] = config["seed"]
    gen_cfg = config["generation"]

    gpu, dtype = _device_dtype()
    checks: dict[str, dict] = {}

    def check(name, ok, detail=""):
        checks[name] = {"ok": bool(ok), "detail": str(detail)}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    # ---- locks (spec 1)
    requests_bank = _load_bank(config, repo_dir)
    check("bank + final-eval locks reproduce", True, f"bank sha {config['bank']['sha256'][:12]}…")

    d0_sha = _sha256(os.path.join(config["detector"]["path"], "model.safetensors"))
    check("frozen D0 verified", d0_sha == config["detector"]["sha256"], d0_sha[:16])
    if d0_sha != config["detector"]["sha256"]:
        raise SystemExit("D0 lock mismatch.")

    # ---- models
    tok = AutoTokenizer.from_pretrained(config["model_id"])
    # ty: ignore[next-line]  # transformers 5.x tokenizer stub union includes None
    if tok.pad_token_id is None:  # ty: ignore[unresolved-attribute]
        tok.pad_token = tok.eos_token  # ty: ignore[invalid-assignment,unresolved-attribute]
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], dtype=dtype).to("cuda")  # ty: ignore[invalid-argument-type]  # stub union
    model.eval()
    d0_tok = AutoTokenizer.from_pretrained(config["detector"]["path"])
    d0 = AutoModelForTokenClassification.from_pretrained(config["detector"]["path"]).to("cuda")
    d0.eval()
    check("models loaded (base 2B, no adapter; frozen D0 eval-mode)", True, f"{gpu} dtype={dtype}")
    env = {
        "gpu": gpu,
        "dtype": str(dtype),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": sys.version.split()[0],
    }

    # ---- timing pilot (spec 2)
    pilot_req = requests_bank[: gen_cfg["pilot"]["requests"]]
    t0 = time.time()
    _generate(
        tok,
        model,
        pilot_req,
        temperature=gen_cfg["temperatures"][0],
        n=gen_cfg["pilot"]["n"],
        top_p=gen_cfg["top_p"],
        max_new_tokens=gen_cfg["max_new_tokens"],
        batch_prompts=gen_cfg["batch_prompts"],
        log_prefix="pilot",
    )
    pilot_s = time.time() - t0
    per_gen = pilot_s / (len(pilot_req) * gen_cfg["pilot"]["n"])
    total_gens = gen_cfg["n"] * len(requests_bank) * len(gen_cfg["temperatures"]) + len(
        requests_bank
    )
    print(
        f"timing pilot: {per_gen:.2f} s/generation -> projected total {per_gen * total_gens / 60:.0f} min"
    )

    # ---- generation passes
    samples: list[dict] = []
    for temp in gen_cfg["temperatures"]:
        torch.manual_seed(config["seed"] + int(temp * 10))
        print(f"sampling at temperature {temp} ...")
        samples += _generate(
            tok,
            model,
            requests_bank,
            temperature=temp,
            n=gen_cfg["n"],
            top_p=gen_cfg["top_p"],
            max_new_tokens=gen_cfg["max_new_tokens"],
            batch_prompts=gen_cfg["batch_prompts"],
            log_prefix=f"t={temp}",
        )
    greedy: list[dict] = []
    if gen_cfg.get("greedy_reference", True):
        print("greedy reference pass ...")
        greedy = _generate(
            tok,
            model,
            requests_bank,
            temperature=None,
            n=1,
            top_p=gen_cfg["top_p"],
            max_new_tokens=gen_cfg["max_new_tokens"],
            batch_prompts=gen_cfg["batch_prompts"],
            log_prefix="greedy",
        )

    # ---- validation (spec 3)
    for s in samples + greedy:
        v = classify_sample(s["raw_text"], s["name"], s["truncated"])
        s.update(v)
        s["json_tolerant_ok"] = parse_json_tolerant(s["raw_text"]) is not None

    # ---- leakage + wrapper asserts (spec 5)
    violations = sum(1 for s in samples + greedy if s["text"] and PROMPT_MARKER in s["text"])
    check(
        "D0-input hygiene: prompt marker absent from message texts",
        violations == 0,
        f"violations={violations}",
    )

    # ---- D0 scoring (spec 5)
    _score_with_d0(
        d0,
        d0_tok,
        [s for s in samples + greedy if s["status"] == "valid"],
        config["detector"]["max_length"],
    )
    lost = sum(1 for s in samples + greedy if s.get("d0_truncated_target"))
    check("target spans inside D0 max length", lost == 0, f"lost={lost}")

    _finalize_samples(samples + greedy, config["rewards"]["logit_clip"])
    for s in samples + greedy:
        s.pop("pred_spans", None)
        s.pop("d0_truncated_target", None)

    # determinism: rescore fresh copies of 50 valid texts and compare
    originals = [x for x in samples if x["status"] == "valid"][:50]
    if originals:
        copies = [
            {"text": s["text"], "target_span": s["target_span"], "status": "valid"}
            for s in originals
        ]
        _score_with_d0(d0, d0_tok, copies, config["detector"]["max_length"])
        same = all(
            math.isclose(float(a["p_target_mean"] or 0.0), float(b["p_target_mean"] or 0.0), abs_tol=1e-5)
            and a["outcome"] == b["outcome"]
            for a, b in zip(originals, copies, strict=True)
        )
        check("D0 scoring deterministic (50 texts rescored)", same)

    # ---- metrics (spec 4 + 5)
    report: dict = {
        "kind": "base-sampling-probe",
        "role": (
            "M07a: diversity + reward-signal probe of BASE Qwen3.5-2B before RL "
            "(M06 baseline candidate / RL starting policy / KL reference)"
        ),
        "started_utc": started.isoformat(),
        "environment": env,
        "config": config,
        "checks": checks,
    }
    greedy_metrics = diversity_metrics(greedy)
    per_temp: dict[str, dict] = {}
    for temp in gen_cfg["temperatures"]:
        subset = [s for s in samples if s["temperature"] == temp]
        dm = diversity_metrics(subset)
        rm = reward_metrics(subset, config["rewards"]["logit_clip"])
        rm["target_spans_lost_to_d0_truncation"] = lost
        dm["json_tolerant_rate"] = round(
            sum(1 for s in subset if s["json_tolerant_ok"]) / len(subset), 4
        )
        dm.update(rm)
        dm["thresholds"] = threshold_report(dm | rm, config["thresholds"])
        per_temp[str(temp)] = dm
    report["greedy_reference"] = {
        k: v for k, v in greedy_metrics.items() if k in ("contract_rate", "failure_breakdown")
    }
    report["per_temperature"] = per_temp

    go = {t: all(c["ok"] for c in m["thresholds"].values()) for t, m in per_temp.items()}
    report["go_no_go"] = go
    recommended_temp = next((t for t, ok in go.items() if ok), None)
    report["recommendation"] = (
        f"temperature {recommended_temp} passes all thresholds - candidate for M08"
        if recommended_temp
        else "no temperature passes all thresholds; report component failures before M08"
    )
    print("\nGO/NO-GO:", json.dumps(go))
    print("RECOMMENDATION:", report["recommendation"])

    # ---- artifacts (spec 8)
    run_id = datetime.datetime.now(datetime.UTC).strftime(
        f"{config['run_id_prefix']}-%Y%m%d-%H%M%S"
    )
    work_dir = os.path.join(config["work_dir"], run_id)
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, "samples.jsonl"), "w", encoding="utf-8") as f:
        for s in samples + greedy:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    report["run_id"] = run_id
    report["finished_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
    report["samples_total"] = len(samples) + len(greedy)
    report["experiments_entry"] = {
        "run_id": run_id,
        "kind": "base-sampling-probe",
        "milestone": "M07a",
        "executed": report["finished_utc"][:10],
        "where": f"colab-{gpu}",
        "repo_commit": config.get("repo_commit", "recorded-by-notebook"),
        "environment": env,
        "artifacts": f"{work_dir} (samples.jsonl, probe_report.json, plots/)",
        "checks": {
            "pass": sum(1 for c in checks.values() if c["ok"]),
            "fail": sum(1 for c in checks.values() if not c["ok"]),
        },
        "metrics": {
            t: {
                k: m[k]
                for k in (
                    "contract_rate",
                    "exact_dup_rate",
                    "unique_per_group_median",
                    "dead_group_share",
                    "live_group_share_r_prob",
                    "live_group_share_r_logit",
                )
            }
            for t, m in per_temp.items()
        },
        "outcome": report["recommendation"],
    }
    with open(os.path.join(work_dir, "probe_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _plot(work_dir, samples)
    _panels(samples, rng)
    print(f"\nrun_id: {run_id}\nreport: {os.path.join(work_dir, 'probe_report.json')}")
    if config.get("hf_upload"):
        print(
            "NOTE: HF artifact upload configured but requires HF_TOKEN; not attempted "
            "unless configured (artifacts are on Drive)."
        )
    return report
