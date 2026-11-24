"""Modal launcher for the M08 RL runs.

One-time setup (target detector weights go to the volume, everything else is fetched by `prepare`):
    modal volume put pii-redteam-rl <local>/d1-frozen-d1v2-20261110-075731 /d1-frozen-d1v2-20261110-075731
    modal run scripts/modal_rl.py::prepare

Timing benchmark, three arms in parallel on one L4 each (10 GRPO steps, bf16 LoRA):
    modal run scripts/modal_rl.py::bench                       # the vLLM arms
    modal run scripts/modal_rl.py::bench --arms hf,vllm_lm   # any subset
Arms: hf = Hugging Face generate (baseline); fla = plus flash-linear-attention; vllm_names =
TRL colocated vLLM with the attacker's weights renamed to the multimodal Qwen3.5 checkpoint names;
vllm_lm = the same plus vLLM loading only the language model (no vision tower).

Capped pilot of any repo config (Modal kills it after `PILOT_TIMEOUT_S`):
    modal run --detach scripts/modal_rl.py::pilot --config configs/m08_rl_person.yaml
The pilot function reads the Modal secret `mistral` (one-time setup, key never in the repo):
    modal secret create mistral MISTRAL_API_KEY=<key>

Augmentation texts from a trained attacker (same secret, killed after `GENERATE_TIMEOUT_S`):
    modal run --detach scripts/modal_rl.py::generate --config configs/m09_rl_generate.yaml

Cost is bounded by `BENCH_TIMEOUT_S` per arm: Modal kills the container at the timeout.
"""

import copy
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_DIR = "/repo"
VOLUME_DIR = "/vol"
TARGET_DETECTOR_DIR = f"{VOLUME_DIR}/d1-frozen-d1v2-20261110-075731"
CACHE_DIR = f"{VOLUME_DIR}/cache"
SPLITS_DIR = f"{VOLUME_DIR}/splits"
RUNS_DIR = f"{VOLUME_DIR}/runs"
HF_HOME = f"{VOLUME_DIR}/hf-cache"
BENCH_CONFIG = f"{REPO_DIR}/configs/m08_rl_bench.yaml"
LONG_CONFIG = f"{REPO_DIR}/configs/m08_rl_long.yaml"

GPU = "L4"
BENCH_TIMEOUT_S = 900
LONG_TIMEOUT_S = 9600
PILOT_TIMEOUT_S = 6600
LONG_PILOT_TIMEOUT_S = 10800  # 1000-step runs: about 140 min of steps on an L4 plus setup
GENERATE_TIMEOUT_S = 1800
JUDGE_SECRET = "mistral"  # holds MISTRAL_API_KEY for the person judge (reward.person_judge)
L4_CONTAINER_USD_PER_HOUR = 1.12  # approximate list prices: L4 + 4 CPU + 16 GiB
COMMIT_INTERVAL_S = 120

LOCKED_PACKAGES = [
    "torch==2.14.0",
    "transformers==5.18.0",
    "trl==1.17.1",
    "peft==0.21.2",
    "accelerate==1.15.0",
    "datasets==5.2.0",
    "huggingface-hub==1.33.0",
    "pyyaml",
]
FLA_PACKAGES = ["flash-linear-attention==0.6.0"]
VLLM_PACKAGES = ["vllm==0.33.0"]
VLLM_OVERRIDES = {
    "grpo.use_vllm": True,
    "grpo.vllm_mode": "colocate",
    "grpo.vllm_gpu_memory_utilization": 0.3,
    "grpo.vllm_max_model_length": 1024,
}
VLLM_NAMES_OVERRIDES = {**VLLM_OVERRIDES, "vllm_multimodal_names": True}
VLLM_LM_OVERRIDES = {**VLLM_NAMES_OVERRIDES, "vllm_language_model_only": True}
REPORTED_PACKAGES = ["torch", "transformers", "trl", "peft", "vllm", "flash-linear-attention"]

# Importing this launcher writes its own .pyc under scripts/, which would change mid-build.
COMPILED_PYTHON = modal.FilePatternMatcher("**/__pycache__", "**/*.pyc")


def build_image(packages: list[str]) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(*packages)
        .env(
            {
                "HF_HOME": HF_HOME,
                "PYTHONPATH": f"{REPO_DIR}/src",
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        .add_local_dir(REPO_ROOT / "src", f"{REPO_DIR}/src", ignore=COMPILED_PYTHON)
        .add_local_dir(REPO_ROOT / "configs", f"{REPO_DIR}/configs")
        .add_local_dir(REPO_ROOT / "banks", f"{REPO_DIR}/banks")
        .add_local_dir(REPO_ROOT / "scripts", f"{REPO_DIR}/scripts", ignore=COMPILED_PYTHON)
    )


IMAGE_HF = build_image(LOCKED_PACKAGES)
IMAGE_FLA = build_image(LOCKED_PACKAGES + FLA_PACKAGES)
IMAGE_VLLM = build_image(LOCKED_PACKAGES + VLLM_PACKAGES)

with IMAGE_HF.imports():
    import yaml
    from huggingface_hub import snapshot_download

app = modal.App("pii-redteam-rl")
volume = modal.Volume.from_name("pii-redteam-rl", create_if_missing=True)
GPU_FUNCTION: dict[str, Any] = {
    "gpu": GPU,
    "cpu": 4,
    "memory": 16384,
    "timeout": BENCH_TIMEOUT_S,
    "volumes": {VOLUME_DIR: volume},
}
LONG_GPU_FUNCTION: dict[str, Any] = {**GPU_FUNCTION, "timeout": LONG_TIMEOUT_S}
PILOT_GPU_FUNCTION: dict[str, Any] = {
    **GPU_FUNCTION,
    "timeout": PILOT_TIMEOUT_S,
    "secrets": [modal.Secret.from_name(JUDGE_SECRET)],
}


GENERATE_GPU_FUNCTION: dict[str, Any] = {**PILOT_GPU_FUNCTION, "timeout": GENERATE_TIMEOUT_S}
LONG_PILOT_GPU_FUNCTION: dict[str, Any] = {**PILOT_GPU_FUNCTION, "timeout": LONG_PILOT_TIMEOUT_S}


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def with_overrides(config: dict, overrides: dict) -> dict:
    """Copy of `config` with dotted keys such as "grpo.use_vllm" replaced."""
    merged = copy.deepcopy(config)
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split(".")
        node = merged
        for key in parents:
            node = node[key]
        node[leaf] = value
    return merged


def stream(command: list[str], log_path: str, env: dict) -> int:
    """Run `command`, echo its output to the Modal logs and keep a copy in `log_path`."""
    with (
        subprocess.Popen(
            command,
            cwd=REPO_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process,
        open(log_path, "w", encoding="utf-8") as log,
    ):
        for line in process.stdout or ():
            print(line, end="", flush=True)
            log.write(line)
    return process.returncode


def package_versions() -> dict:
    versions = {}
    for name in REPORTED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def latest_checkpoint(run_dir: str) -> str:
    """Highest-step checkpoint saved under a previous run directory."""
    checkpoints = glob.glob(os.path.join(run_dir, "*", "checkpoints", "checkpoint-*"))
    if not checkpoints:
        raise SystemExit(f"no checkpoint under {run_dir}")
    return max(checkpoints, key=lambda path: int(path.rsplit("-", 1)[1]))


def arm_summary(arm: str, exit_code: int, wall_seconds: float, run_dir: str) -> dict:
    summary = {
        "arm": arm,
        "run_dir": run_dir,
        "exit_code": exit_code,
        "wall_seconds": round(wall_seconds, 1),
        "estimated_cost_usd": round(wall_seconds / 3600 * L4_CONTAINER_USD_PER_HOUR, 3),
        "packages": package_versions(),
    }
    reports = glob.glob(os.path.join(run_dir, "*", "rl_*_report.json"))
    if not reports:
        return summary
    with open(reports[0], encoding="utf-8") as f:
        report = json.load(f)
    training = report.get("training", {})
    summary.update(
        {
            "gpu": report["environment"]["gpu"],
            "steps_done": training.get("steps_done"),
            "timing": training.get("timing"),
            "peak_memory_gb": report.get("peak_memory_gb"),
            "counts": report.get("counts"),
            "pre_valid_rate": report.get("pre", {}).get("valid_rate"),
            "post_valid_rate": report.get("post", {}).get("valid_rate"),
            "failed_checks": [
                name for name, check in report.get("checks", {}).items() if not check["ok"]
            ],
        }
    )
    return summary


def commit_volume_periodically(stop: threading.Event) -> None:
    """Persist checkpoints and logs written so far: uncommitted files are lost if Modal kills the run."""
    while not stop.wait(COMMIT_INTERVAL_S):
        volume.commit()


def run_arm(
    arm: str,
    overrides: dict,
    commit: str,
    base_config: str = BENCH_CONFIG,
    resume_from: str = "",
    command_name: str = "rl-smoke",
) -> dict:
    if resume_from:
        overrides = {**overrides, "resume_from_checkpoint": latest_checkpoint(resume_from)}
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = f"{RUNS_DIR}/{arm}-{stamp}"
    os.makedirs(run_dir)
    with open(base_config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    paths = {"work_dir": run_dir, "splits_dir": SPLITS_DIR}
    if not base["detector"]["path"].startswith(f"{VOLUME_DIR}/"):
        paths["detector.path"] = TARGET_DETECTOR_DIR  # a repo path means the target detector
    config = with_overrides(base, {**paths, **overrides})
    config_path = f"{run_dir}/config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    command = [sys.executable, "-u", "-m", "pii_redteam.cli", command_name, "--config", config_path]
    started = time.time()
    env = dict(os.environ, PROBE_REPO_COMMIT=commit)
    stop = threading.Event()
    committer = threading.Thread(target=commit_volume_periodically, args=(stop,), daemon=True)
    committer.start()
    exit_code = stream(command, f"{run_dir}/run.log", env)
    stop.set()
    committer.join()
    summary = arm_summary(arm, exit_code, time.time() - started, run_dir)
    with open(f"{run_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    volume.commit()
    return summary


@app.function(image=IMAGE_HF, volumes={VOLUME_DIR: volume}, cpu=2, memory=8192, timeout=1800)
def prepare() -> dict:
    """CPU only: check the target detector lock, cache the attacker model, fetch the data and carve the splits."""
    with open(BENCH_CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    weights = f"{TARGET_DETECTOR_DIR}/model.safetensors"
    if not os.path.exists(weights):
        raise SystemExit(
            f"target detector weights missing in the volume: upload the folder to {TARGET_DETECTOR_DIR}"
        )
    weights_sha = sha256_of(weights)
    if weights_sha != config["detector"]["sha256"]:
        raise SystemExit(f"target detector lock mismatch: {weights_sha}")
    snapshot_download(config["model_id"])
    carve = [
        sys.executable,
        f"{REPO_DIR}/scripts/carve_splits.py",
        "--cache-dir",
        CACHE_DIR,
        "--out",
        SPLITS_DIR,
    ]
    if stream(carve, f"{VOLUME_DIR}/carve_splits.log", dict(os.environ)) != 0:
        raise SystemExit("carve_splits failed, see the log above")
    volume.commit()
    return {
        "target_detector_sha256": weights_sha,
        "splits_dir": SPLITS_DIR,
        "model_id": config["model_id"],
    }


@app.function(image=IMAGE_HF, **GPU_FUNCTION)
def bench_hf(commit: str) -> dict:
    return run_arm("bench-hf", {}, commit)


@app.function(image=IMAGE_FLA, **GPU_FUNCTION)
def bench_fla(commit: str) -> dict:
    return run_arm("bench-fla", {}, commit)


@app.function(image=IMAGE_VLLM, **GPU_FUNCTION)
def bench_vllm_names(commit: str) -> dict:
    return run_arm("bench-vllm-names", VLLM_NAMES_OVERRIDES, commit)


@app.function(image=IMAGE_VLLM, **GPU_FUNCTION)
def bench_vllm_lm(commit: str) -> dict:
    return run_arm("bench-vllm-lm", VLLM_LM_OVERRIDES, commit)


@app.function(image=IMAGE_VLLM, **GPU_FUNCTION)
def resume_a(commit: str) -> dict:
    overrides = {**VLLM_NAMES_OVERRIDES, "grpo.max_steps": 20, "grpo.save_steps": 10}
    return run_arm("resume-a", overrides, commit)


@app.function(image=IMAGE_VLLM, **GPU_FUNCTION)
def resume_b(commit: str, resume_from: str) -> dict:
    overrides = {**VLLM_NAMES_OVERRIDES, "grpo.max_steps": 30, "grpo.save_steps": 10}
    return run_arm("resume-b", overrides, commit, resume_from=resume_from)


@app.function(image=IMAGE_VLLM, **LONG_GPU_FUNCTION)
def long_run(commit: str, resume_from: str = "") -> dict:
    return run_arm("long", {}, commit, base_config=LONG_CONFIG, resume_from=resume_from)


@app.function(image=IMAGE_VLLM, **PILOT_GPU_FUNCTION)
def pilot_run(commit: str, config: str, resume_from: str = "") -> dict:
    name = Path(config).stem.removeprefix("m08_rl_")
    return run_arm(
        f"pilot-{name}", {}, commit, base_config=f"{REPO_DIR}/{config}", resume_from=resume_from
    )


@app.function(image=IMAGE_HF, **GENERATE_GPU_FUNCTION)
def generate_run(commit: str, config: str) -> dict:
    return run_arm(
        "generate", {}, commit, base_config=f"{REPO_DIR}/{config}", command_name="rl-generate"
    )


BENCH_ARMS = {
    "hf": bench_hf,
    "fla": bench_fla,
    "vllm_names": bench_vllm_names,
    "vllm_lm": bench_vllm_lm,
}


def local_commit() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    return f"{head}+dirty" if dirty else head


def format_result(arm: str, result: dict | Exception) -> str:
    if isinstance(result, Exception):
        return f"{arm:<8} FAILED: {type(result).__name__}: {result}"
    timing = result.get("timing") or {}
    step = (timing.get("step") or {}).get("median")
    generation = (timing.get("generation") or {}).get("median")
    return (
        f"{arm:<8} exit={result['exit_code']} steps={result.get('steps_done')} "
        f"median step={step}s generation={generation}s "
        f"peak={result.get('peak_memory_gb')}GB wall={result['wall_seconds']}s "
        f"~${result['estimated_cost_usd']} failed_checks={result.get('failed_checks')}"
    )


@app.local_entrypoint()
def bench(arms: str = "vllm_names,vllm_lm"):
    commit = local_commit()
    print(f"repo state: {commit}; timeout {BENCH_TIMEOUT_S}s per arm on {GPU}")
    calls = {arm: BENCH_ARMS[arm].spawn(commit) for arm in arms.split(",")}
    results = {}
    for arm, call in calls.items():
        try:
            results[arm] = call.get()
        except (modal.exception.Error, TimeoutError) as error:
            results[arm] = error
    for arm, result in results.items():
        print(format_result(arm, result))


@app.local_entrypoint()
def resume_test():
    """20 steps with checkpoints every 10, then resume the latest checkpoint up to step 30."""
    commit = local_commit()
    first = resume_a.remote(commit)
    print(format_result("resume-a", first))
    second = resume_b.remote(commit, first["run_dir"])
    print(format_result("resume-b", second))


@app.local_entrypoint()
def long(resume_from: str = ""):
    """The long run; `--resume-from <run_dir>` continues from the latest checkpoint of a killed run."""
    commit = local_commit()
    print(f"repo state: {commit}; timeout {LONG_TIMEOUT_S}s on {GPU}")
    print(format_result("long", long_run.remote(commit, resume_from)))


@app.local_entrypoint()
def pilot(config: str, resume_from: str = ""):
    """A capped run of any repo config, e.g. `--config configs/m08_rl_opening.yaml`."""
    commit = local_commit()
    print(f"repo state: {commit}; config {config}; timeout {PILOT_TIMEOUT_S}s on {GPU}")
    print(format_result("pilot", pilot_run.remote(commit, config, resume_from)))


@app.function(image=IMAGE_VLLM, **LONG_PILOT_GPU_FUNCTION)
def long_pilot_run(commit: str, config: str, resume_from: str = "") -> dict:
    return pilot_run.local(commit, config, resume_from)


@app.local_entrypoint()
def long_pilot(config: str, resume_from: str = ""):
    """Like `pilot`, with a 3 h timeout for 1000-step runs.

    The run is spawned, not awaited: with `modal run --detach` it keeps going after this client
    exits. Awaiting it (`.remote`) tied the run to the local client, and twice on 2026-11-22 the
    client's "Jwt is expired" error (laptop asleep) stopped the app and killed the run. Follow it
    with `modal app logs <app id>` and read the results from the volume.
    """
    commit = local_commit()
    print(f"repo state: {commit}; config {config}; timeout {LONG_PILOT_TIMEOUT_S}s on {GPU}")
    call = long_pilot_run.spawn(commit, config, resume_from)
    print(f"spawned function call {call.object_id}; the run continues on Modal")


@app.local_entrypoint()
def generate(config: str):
    """Sample a trained RL attacker on fresh names and filter the texts."""
    commit = local_commit()
    print(f"repo state: {commit}; config {config}; timeout {GENERATE_TIMEOUT_S}s on {GPU}")
    result = generate_run.remote(commit, config)
    print(format_result("generate", result))
    print(json.dumps(result.get("counts"), indent=2))
