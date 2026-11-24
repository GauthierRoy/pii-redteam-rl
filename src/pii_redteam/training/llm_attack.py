"""Non-RL attack arm: a large API model writes texts and sees the detector's scores (owner idea 1).

Matched to the RL attack (configs/m10_attack_person_modal.yaml): same detector, same requests
(training bank and the 32 held-out requests), same validity rules, person judge, audit with
other-person rejection, opening-share guard and log-odds reward. One step = `requests_per_step`
requests x `texts_per_request` texts, as one GRPO step (4 x 8 = 32 detector queries).

What the writer knows that the RL attacker does not: the goal, the rules in words, and a board
of earlier texts with their detector probability and reward (hardest, easiest, rejected with
the reason). The RL attacker only receives one reward per text. Both see the same scores.

Every API call (writer, judge, audit) goes through one meter and one pacer: token counts per
model, an estimated price from `pricing_usd_per_million` (unverified list prices; the token
counts are the record), a minimum gap between calls on the shared key, a pause between steps
and a spending cap that stops the run. HTTP 429 answers are retried after Retry-After.

After the steps, the final board is frozen and the writer answers the 32 held-out requests
(8 texts each), the counterpart of the RL run's post-training samples.
"""

from __future__ import annotations

import datetime
import json
import os
import random
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from ..prompt_v2 import render_prompt_v2
from ..requests import CONTEXT_TYPE_LABELS
from ..run_io import write_jsonl
from .person_judge import (
    AUDIT_FIELDS,
    MISTRAL_CHAT_URL,
    PersonJudge,
    TextAudit,
    answer_text,
    backoff_seconds,
    http_transport,
)
from .rl_dataset import RL_PROMPT_VARIANT
from .rl_reward import TargetDetectorReward
from .rl_smoke import detector_reward, opening_share_guard

EOS = 0  # API texts are never cut by a token limit the validity rules can see; see write_texts
MISS_BELOW = 0.5

WRITER_BRIEF = """You are helping red-team a PII detector so that it can be retrained. The detector \
is a token-classification model that should find every person's name in a short message. For each \
text you write, it gives a score p: its probability that the requested name is a PERSON (averaged \
over the name's tokens). A text where p < 0.5 is a miss: the detector failed to find the name. \
Your goal is to write texts that the detector misses, or at least scores as low as possible.

Every text must still be a legitimate training example, or it earns nothing:
- It follows the request below (kind of message, length, the name exactly as given, exactly once).
- The requested name refers to a person. An independent judge rejects texts where the name is \
used as a place, brand, organisation, product or anything else.
- It names no other person (no other first names, surnames or full names).
- It is a natural message a real person or organisation could have written.
- Openings repeated across many different requests earn less: vary how texts start.

Reward of a valid text: 1 if p <= 0.5, else 1 + log((1 - p) / p) / 10 clipped to [0, 1] \
(p = 0.99 earns 0.54, p = 0.9998 earns 0.15). Rejected texts earn -1.
{board}
Request:
{request}

Write {n} different texts for this request (each one is a full answer to it; instead of one \
JSON object per text, put all {n} texts in one list). Reply with only a JSON object: \
{{"texts": ["first text", "second text", ...]}}"""


class UsageMeter:
    """Thread-safe token counts per model and the estimated price."""

    def __init__(self, prices: dict[str, list[float]]) -> None:
        self.prices = prices
        self.calls: Counter = Counter()
        self.prompt_tokens: Counter = Counter()
        self.completion_tokens: Counter = Counter()
        self.reasoning_tokens: Counter = Counter()  # part of completion_tokens, billed as output
        self.missing_usage: Counter = Counter()
        self._lock = threading.Lock()

    def record(self, model: str, usage: dict | None) -> None:
        with self._lock:
            self.calls[model] += 1
            if not usage:
                self.missing_usage[model] += 1
                return
            self.prompt_tokens[model] += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens[model] += int(usage.get("completion_tokens") or 0)
            details = usage.get("completion_tokens_details") or {}
            self.reasoning_tokens[model] += int(details.get("reasoning_tokens") or 0)

    def cost(self, model: str) -> float:
        price_in, price_out = self.prices[model]
        return (
            self.prompt_tokens[model] * price_in + self.completion_tokens[model] * price_out
        ) / 1e6

    def total_cost(self) -> float:
        return sum(self.cost(model) for model in self.calls)

    def summary(self) -> dict:
        return {
            model: {
                "calls": self.calls[model],
                "prompt_tokens": self.prompt_tokens[model],
                "completion_tokens": self.completion_tokens[model],
                "of_which_reasoning_tokens": self.reasoning_tokens[model],
                "calls_without_usage": self.missing_usage[model],
                "usd_per_million": self.prices[model],
                "estimated_usd": round(self.cost(model), 4),
            }
            for model in sorted(self.calls)
        }


class Pacer:
    """At most one call start per `min_gap_seconds` on the shared key, across threads."""

    def __init__(self, min_gap_seconds: float, sleep=time.sleep, clock=time.monotonic) -> None:
        self.min_gap_seconds = min_gap_seconds
        self.sleep = sleep
        self.clock = clock
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self.clock()
            start = max(now, self._next)
            self._next = start + self.min_gap_seconds
        if start > now:
            self.sleep(start - now)


def metered(post: Callable[[dict], dict], meter: UsageMeter, pacer: Pacer):
    def call(body: dict) -> dict:
        pacer.wait()
        payload = post(body)
        meter.record(body["model"], payload.get("usage"))
        return payload

    return call


def parse_texts(answer: str) -> list[str] | None:
    """The `texts` list of the writer's JSON answer, or None."""
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(answer[start : end + 1])
    except ValueError:
        return None
    texts = parsed.get("texts") if isinstance(parsed, dict) else None
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        return None
    return texts


class Board:
    """Every scored text so far; rendered as the writer's feedback."""

    def __init__(self, hardest: int, easiest: int, rejected: int, max_chars: int) -> None:
        self.hardest, self.easiest, self.rejected = hardest, easiest, rejected
        self.max_chars = max_chars
        self.rows: list[dict] = []

    def add(self, rows: list[dict]) -> None:
        self.rows.extend(rows)

    def _line(self, row: dict) -> str:
        text = (
            row["text"]
            if len(row["text"]) <= self.max_chars
            else row["text"][: self.max_chars] + "..."
        )
        return f"- [{row['context']}] name={row['name']!r}: {json.dumps(text, ensure_ascii=False)}"

    def render(self) -> str:
        if not self.rows:
            return "\nNo feedback yet: this is the first round.\n"
        valid = [r for r in self.rows if r["status"] == "valid"]
        by_p = sorted(valid, key=lambda r: r["p"])
        misses = sum(r["p"] < MISS_BELOW for r in valid)
        lines = [
            "",
            f"Feedback so far: {len(self.rows)} texts, {len(valid)} valid, {misses} missed by "
            "the detector.",
        ]
        if by_p:
            lines.append("Lowest scores (best for you), p and reward shown:")
            lines += [
                f"{self._line(r)} p={r['p']:.4f} reward={r['reward']:.2f}"
                for r in by_p[: self.hardest]
            ]
            lines.append("Highest scores (the detector found these easily):")
            lines += [f"{self._line(r)} p={r['p']:.4f}" for r in by_p[::-1][: self.easiest]]
        rejected = [r for r in self.rows if r["status"] != "valid" and r["text"]][-self.rejected :]
        if rejected:
            lines.append("Recently rejected (reward -1), with the reason:")
            lines += [f"{self._line(r)} reason={r['status']}" for r in rejected]
        return "\n".join(lines) + "\n"


WRITER_BRIEF_V2 = """You are helping red-team a PII detector so that it can be retrained. The \
detector is a token-classification model that should find every person's name in a short message. \
For each text you write, it gives a score p: its probability that the requested name is a PERSON \
(averaged over the name's tokens). A text where p < 0.5 is a miss: the detector failed to find the \
name. Your goal is to write texts that the detector misses, or at least scores as low as possible. \
Detector weaknesses are often not intuitive: look for patterns in the feedback (wording, position \
of the name, message format, what comes right before or after the name) and test them.

Every text must still be a legitimate training example, or it earns nothing:
- It follows its request (kind of message, length, the name exactly as given, exactly once).
- The requested name refers to a person. An independent judge rejects texts where the name is \
used as a place, brand, organisation, product or anything else.
- It names no other person (no other first names, surnames or full names).
- It is a natural message a real person or organisation could have written.
- Openings repeated across many different requests earn less: vary how texts start.

Reward of a valid text: 1 if p <= 0.5, else 1 + log((1 - p) / p) / 10 clipped to [0, 1] \
(p = 0.99 earns 0.54, p = 0.9998 earns 0.15). Rejected texts earn -1.

This is round {round} of {rounds}. Each round you answer {k} requests with {n} texts each; then \
every text is scored and the feedback below is updated.
{board}
Requests for this round:
{requests}

First write your current hypothesis: what makes the detector miss or doubt a name, based on the \
evidence, and what you will test this round (a few sentences; you will see it again next round). \
Then the texts. Reply with only a JSON object: {{"hypothesis": "...", "answers": [[{n} texts for \
request 1], [{n} texts for request 2], ...]}} with exactly {k} lists, in request order, each text a \
plain JSON string (not an object with a "text" field)."""


def as_text(item: object) -> str | None:
    """A text given as a plain string or as {"text": "..."} (the request's own format), also when
    that object arrives as a JSON string (GLM 5.3 did this for whole requests)."""
    if isinstance(item, str):
        if item.lstrip().startswith("{"):
            try:
                inner = json.loads(item)
            except ValueError:
                return item
            if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                return inner["text"]
        return item
    if isinstance(item, dict) and isinstance(item.get("text"), str):
        return item["text"]
    return None


def parse_answers(answer: str, k: int, n: int) -> tuple[str, list[list[str]]] | None:
    """(hypothesis, k lists of n texts) from the v2 JSON answer, or None."""
    start, end = answer.find("{"), answer.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(answer[start : end + 1])
    except ValueError:
        return None
    answers = parsed.get("answers") if isinstance(parsed, dict) else None
    if not isinstance(answers, list) or len(answers) != k:
        return None
    if not all(isinstance(a, list) for a in answers):
        return None
    texts = [[as_text(item) for item in a] for a in answers]
    if any(t is None for a in texts for t in a):
        return None
    hypothesis = parsed.get("hypothesis")
    return (
        hypothesis if isinstance(hypothesis, str) else "",
        [(a + [""] * n)[:n] for a in texts],
    )


def name_position(text: str, name: str) -> str:
    index = text.find(name)
    if index < 0:
        return "absent"
    if not text[:index].strip(" \"'*\n"):
        return "first word"
    share = index / max(1, len(text))
    return "first third" if share < 1 / 3 else "middle third" if share < 2 / 3 else "last third"


class BoardV2(Board):
    """Large feedback: statistics, every miss, near misses and easy finds (one per name),
    rejections by reason, the writer's previous round in full and its previous hypothesis."""

    def __init__(
        self,
        misses: int,
        closest: int,
        easiest: int,
        rejected: int,
        max_chars: int,
        last_rounds: int = 1,
    ):
        super().__init__(closest, easiest, rejected, max_chars)
        self.max_misses = misses
        self.last_rounds = last_rounds
        self.rounds: list[list[dict]] = []

    def add(self, rows: list[dict]) -> None:
        super().add(rows)
        self.rounds = (self.rounds + [rows])[-self.last_rounds :]

    @staticmethod
    def _table(title: str, valid: list[dict], key) -> list[str]:
        groups: dict[str, list[dict]] = {}
        for row in valid:
            groups.setdefault(key(row), []).append(row)
        lines = [f"{title} (valid texts, misses, mean reward):"]
        for label, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            misses = sum(r["p"] < MISS_BELOW for r in rows)
            mean = sum(r["reward"] for r in rows) / len(rows)
            lines.append(f"  {label}: {len(rows)}, {misses}, {mean:.2f}")
        return lines

    @staticmethod
    def _one_per_name(rows: list[dict], limit: int) -> list[dict]:
        seen, picked = set(), []
        for row in rows:
            if row["name"] not in seen:
                seen.add(row["name"])
                picked.append(row)
            if len(picked) == limit:
                break
        return picked

    def render(self, hypothesis: str = "") -> str:
        if not self.rows:
            return "\nNo feedback yet: this is the first round.\n"
        valid = [r for r in self.rows if r["status"] == "valid"]
        misses = [r for r in valid if r["p"] < MISS_BELOW]
        reasons = Counter(r["status"] for r in self.rows if r["status"] != "valid")
        lines = [
            "",
            f"FEEDBACK. So far: {len(self.rows)} texts, {len(valid)} valid, {len(misses)} missed.",
            "Rejections by reason: "
            + (", ".join(f"{k} {v}" for k, v in reasons.most_common()) or "none"),
        ]
        if valid:
            lines += self._table("By kind of message", valid, lambda r: r["context"])
            lines += self._table(
                "By position of the name", valid, lambda r: name_position(r["text"], r["name"])
            )
        if misses:
            lines.append(f"Every miss so far (most recent first, up to {self.max_misses}):")
            lines += [f"{self._line(r)} p={r['p']:.4f}" for r in misses[::-1][: self.max_misses]]
        near = [r for r in sorted(valid, key=lambda r: r["p"]) if r["p"] >= MISS_BELOW]
        if near:
            lines.append("Closest to a miss, one per name (p and reward):")
            lines += [
                f"{self._line(r)} p={r['p']:.4f} reward={r['reward']:.2f}"
                for r in self._one_per_name(near, self.hardest)
            ]
            lines.append("Found most easily, one per name:")
            lines += [
                f"{self._line(r)} p={r['p']:.4f}"
                for r in self._one_per_name(near[::-1], self.easiest)
            ]
        rejected: list[dict] = []
        per_reason: Counter = Counter()
        for row in self.rows[::-1]:
            if row["status"] != "valid" and row["text"] and per_reason[row["status"]] < 2:
                per_reason[row["status"]] += 1
                rejected.append(row)
        if rejected:
            lines.append("Rejected examples (reward -1), up to two per reason:")
            lines += [f"{self._line(r)} reason={r['status']}" for r in rejected[: self.rejected]]
        for back, rows in enumerate(self.rounds[::-1], start=1):
            label = "Your previous round" if back == 1 else f"Your round {back} rounds ago"
            lines.append(f"{label}, every text with its result:")
            for r in rows:
                result = f"p={r['p']:.4f}" if r["status"] == "valid" else f"rejected: {r['status']}"
                lines.append(f"{self._line(r)} {result}")
        if hypothesis:
            lines.append(f"Your hypothesis last round: {hypothesis}")
        return "\n".join(lines) + "\n"


class Writer:
    def __init__(self, settings: dict, post: Callable[[dict], dict]) -> None:
        self.model = settings["model"]
        self.temperature = settings.get("temperature", 1.0)
        self.max_tokens = settings["max_tokens"]
        self.attempts = settings.get("attempts", 4)
        self.extra = {k: settings[k] for k in ("reasoning_effort",) if settings.get(k)}
        self.post = post
        self.failed_calls = 0

    def ask(self, prompt: str, parse: Callable[[str], object | None]) -> object | None:
        """The parsed answer, or None after `attempts` failed calls or unparsable answers."""
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            **self.extra,
        }
        for attempt in range(self.attempts):
            try:
                payload = self.post(body)
                if payload["choices"][0].get("finish_reason") == "length":
                    raise ValueError("writer answer cut at max_tokens")
                parsed = parse(answer_text(payload))
                if parsed:
                    return parsed
                raise ValueError("writer answer is not the JSON object")
            except (OSError, KeyError, IndexError, ValueError) as error:
                print(f"  writer attempt {attempt + 1}: {error!r}"[:300], flush=True)
                time.sleep(backoff_seconds(attempt, error, 2.0, 60.0))
        self.failed_calls += 1
        return None

    def write(self, prompt: str, n: int) -> list[str]:
        """`n` texts (padded with "" when the writer returns fewer or fails every attempt)."""
        texts = self.ask(prompt, parse_texts)
        return (list(texts) + [""] * n)[:n] if texts else [""] * n


def request_prompt(request: dict) -> str:
    return render_prompt_v2(request, RL_PROMPT_VARIANT)


def score_batch(
    reward: TargetDetectorReward, requests: list[dict], texts: list[list[str]]
) -> list[dict]:
    """Score the writer's texts through the RL reward; one row per text."""
    flat = [(req, text) for req, batch in zip(requests, texts, strict=True) for text in batch]
    names = [req["task"]["entities"][0]["value"] for req, _ in flat]
    contexts = [CONTEXT_TYPE_LABELS[req["task"]["context_type"]] for req, _ in flat]
    extras: dict = {}
    rewards = reward(
        [json.dumps({"text": text}, ensure_ascii=False) for _, text in flat],
        names,
        completion_ids=[[EOS]] * len(flat),
        request_id=[req["request_id"] for req, _ in flat],
        context=contexts,
        log_extra=extras.__setitem__,
    )
    rows = []
    for i, ((req, text), r) in enumerate(zip(flat, rewards, strict=True)):
        rows.append(
            {
                "request_id": req["request_id"],
                "context_type": req["task"]["context_type"],
                "context": contexts[i],
                "name": names[i],
                "text": text,
                "status": extras["status"][i],
                "p": extras["p_target_mean"][i],
                "reward": r,
                "opening_share": extras["opening_share"][i],
                **{field: extras[field][i] for field in AUDIT_FIELDS},
            }
        )
    return rows


def batch_metrics(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["status"] == "valid"]
    return {
        "texts": len(rows),
        "valid": len(valid),
        "misses": sum(r["p"] < MISS_BELOW for r in valid),
        "mean_reward": round(sum(r["reward"] for r in rows) / len(rows), 4),
        "median_p_valid": (sorted(r["p"] for r in valid)[len(valid) // 2] if valid else None),
        "statuses": dict(Counter(r["status"] for r in rows)),
    }


def write_step(writer: Writer, board: Board, requests: list[dict], n: int) -> list[list[str]]:
    feedback = board.render()
    prompts = [
        WRITER_BRIEF.format(board=feedback, request=request_prompt(req), n=n) for req in requests
    ]
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        return list(pool.map(lambda p: writer.write(p, n), prompts))


def write_step_v2(
    writer: Writer,
    board: BoardV2,
    requests: list[dict],
    n: int,
    hypothesis: str,
    round_no: int,
    rounds: int,
) -> tuple[list[list[str]], str]:
    """One writer call for every request of the step; returns the texts and the new hypothesis."""
    listed = "\n\n".join(
        f"Request {i + 1}:\n{request_prompt(req)}" for i, req in enumerate(requests)
    )
    prompt = WRITER_BRIEF_V2.format(
        round=round_no,
        rounds=rounds,
        k=len(requests),
        n=n,
        board=board.render(hypothesis),
        requests=listed,
    )
    parsed = writer.ask(prompt, lambda answer: parse_answers(answer, len(requests), n))
    if not parsed:
        return [[""] * n for _ in requests], hypothesis
    new_hypothesis, texts = parsed
    return texts, new_hypothesis or hypothesis


def run_llm_attack(config: dict, repo_dir: str) -> dict:
    from . import rl_smoke

    started = datetime.datetime.now(datetime.UTC)
    run_id = started.strftime(f"{config['run_id_prefix']}-%Y%m%d-%H%M%S")
    work_dir = os.path.join(rl_smoke.resolve(config["work_dir"], repo_dir), run_id)
    os.makedirs(work_dir, exist_ok=True)

    train_requests, signal_requests, bank_provenance = rl_smoke.build_requests(config, repo_dir)
    order = list(range(len(train_requests)))
    random.Random(config["seed"]).shuffle(order)

    api = config["api"]
    api_key = os.environ.get(api["api_key_env"])
    if not api_key:
        raise SystemExit(f"llm-attack needs the {api['api_key_env']} environment variable")
    meter = UsageMeter(api["pricing_usd_per_million"])
    pacer = Pacer(api["min_seconds_between_calls"])

    def post(timeout: float):
        return metered(http_transport(api_key, MISTRAL_CHAT_URL, timeout), meter, pacer)

    judge_cfg, audit_cfg = config["reward"]["person_judge"], config["reward"]["audit"]
    judge = PersonJudge(judge_cfg["model"], post(judge_cfg["timeout_s"]), judge_cfg["max_workers"])
    audit = TextAudit(audit_cfg["model"], post(audit_cfg["timeout_s"]), audit_cfg["max_workers"])
    writer_cfg = config["writer"]
    writer = Writer(writer_cfg, post(writer_cfg["timeout_s"]))
    for model in (writer.model, judge.model, audit.model):
        if model not in meter.prices:
            raise SystemExit(f"no price for {model} in api.pricing_usd_per_million")

    scorer = rl_smoke.load_scorer(config, "cpu")
    reward = TargetDetectorReward(
        scorer,
        EOS,
        opening_share_guard(config),
        judge,
        audit,
        detector_reward=detector_reward(config),
        reject_other_person=bool(config["reward"]["audit"].get("reject_other_person")),
    )
    board_cfg = config["board"]
    version = board_cfg.get("version", 1)
    if version == 2:
        board = BoardV2(
            board_cfg["misses"],
            board_cfg["closest"],
            board_cfg["easiest"],
            board_cfg["rejected"],
            board_cfg["max_chars"],
            board_cfg.get("last_rounds", 1),
        )
    else:
        board = Board(
            board_cfg["hardest"],
            board_cfg["easiest"],
            board_cfg["rejected"],
            board_cfg["max_chars"],
        )
    hypothesis = ""
    hypotheses_path = os.path.join(work_dir, "hypotheses.jsonl")
    steps, per_step = config["steps"], config["requests_per_step"]
    n = config["texts_per_request"]
    pacing = config["pacing"]
    max_usd = api["max_usd"]
    history, stop_reason = [], "steps done"
    texts_path = os.path.join(work_dir, "train_texts.jsonl")
    open(texts_path, "w").close()

    for step in range(steps):
        t0 = time.time()
        batch = [train_requests[i] for i in order[step * per_step : (step + 1) * per_step]]
        if version == 2:
            texts, hypothesis = write_step_v2(writer, board, batch, n, hypothesis, step + 1, steps)
            with open(hypotheses_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step + 1, "hypothesis": hypothesis}) + "\n")
        else:
            texts = write_step(writer, board, batch, n)
        rows = score_batch(reward, batch, texts)
        board.add(rows)
        with open(texts_path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"step": step + 1, **row}, ensure_ascii=False) + "\n")
        entry = {
            "step": step + 1,
            "seconds": round(time.time() - t0, 1),
            **batch_metrics(rows),
            "estimated_usd_total": round(meter.total_cost(), 4),
        }
        history.append(entry)
        print(f"STEP {json.dumps(entry)}", flush=True)
        if meter.total_cost() >= max_usd:
            stop_reason = f"spending cap {max_usd} USD reached"
            break
        time.sleep(pacing["seconds_between_steps"])

    post_rows: list[dict] = []
    if config.get("post_heldout", True) and stop_reason == "steps done":
        print("post: held-out requests with the final board", flush=True)
        post_reward = TargetDetectorReward(
            scorer,
            EOS,
            opening_share_guard(config, window=len(signal_requests) * n),
            judge,
            audit,
            detector_reward=detector_reward(config),
            reject_other_person=bool(config["reward"]["audit"].get("reject_other_person")),
        )
        for i in range(0, len(signal_requests), per_step):
            batch = signal_requests[i : i + per_step]
            if version == 2:
                texts, _ = write_step_v2(writer, board, batch, n, hypothesis, steps, steps)
            else:
                texts = write_step(writer, board, batch, n)
            post_rows += score_batch(post_reward, batch, texts)
            time.sleep(pacing["seconds_between_steps"])
        write_jsonl(os.path.join(work_dir, "samples_post.jsonl"), post_rows)

    all_rows = board.rows
    report = {
        "kind": "llm-attack",
        "run_id": run_id,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "config": config,
        "bank": bank_provenance,
        "stop_reason": stop_reason,
        "steps_done": len(history),
        "writer_failed_calls": writer.failed_calls,
        "board_version": version,
        "final_hypothesis": hypothesis,
        "train": batch_metrics(all_rows) if all_rows else None,
        "train_by_10_steps": [
            {
                "steps": f"{i + 1}-{min(i + 10, len(history))}",
                "misses": sum(e["misses"] for e in history[i : i + 10]),
                "valid": sum(e["valid"] for e in history[i : i + 10]),
                "texts": sum(e["texts"] for e in history[i : i + 10]),
                "mean_reward": round(
                    sum(e["mean_reward"] for e in history[i : i + 10]) / len(history[i : i + 10]), 4
                ),
            }
            for i in range(0, len(history), 10)
        ],
        "post_heldout": batch_metrics(post_rows) if post_rows else None,
        "post_heldout_names_with_a_miss": len(
            {r["name"] for r in post_rows if r["status"] == "valid" and r["p"] < MISS_BELOW}
        ),
        "usage": meter.summary(),
        "estimated_usd_total": round(meter.total_cost(), 4),
        "pricing_note": "list prices from config, unverified; token counts are exact per API usage",
        "history": history,
    }
    rl_smoke.save_report(report, os.path.join(work_dir, "llm_attack_report.json"))
    print(
        "LLM ATTACK:",
        json.dumps(
            {
                k: report[k]
                for k in (
                    "stop_reason",
                    "steps_done",
                    "train",
                    "post_heldout",
                    "estimated_usd_total",
                )
            },
            indent=2,
        ),
    )
    return report
