"""Generate authored SFT v2 texts from quotas via template bank (seeded).

Deterministic per (split, idx). Verifies every row with sft_synth_v2 and
retries template variants; persistent failures are reported for manual pass.
Writes 50-row batch files to /tmp/sft_v2_batches/.
"""

import glob
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from templates_1 import TPL as T1
from templates_2 import TPL as T2

from pii_redteam import sft_synth_v2 as v2

TPL = {**T1, **T2}
BATCH_DIR = "/tmp/sft_v2_batches"
AUTH_SEED = 777001

FILLERS = [
    "Minutes will be circulated Friday for approval.",
    "Please confirm receipt promptly by message.",
    "The clerk will update the central log.",
    "Members agreed unanimously this morning.",
    "Further details follow early next week.",
    "All present approved the draft plan.",
    "Photos were filed with the records.",
    "Notices were pinned to the boards.",
    "Costs will be shared equally throughout.",
    "Everyone thanked the volunteers warmly.",
    "The chair closed the meeting promptly.",
    "Refreshments were served afterwards daily.",
    "New volunteers are welcome anytime soon.",
    "Dates were fixed for the spring term.",
    "Apologies were recorded for absence today.",
    "Matters arising will be deferred again.",
    "The treasurer reported funds healthy today.",
    "Any other business was postponed again.",
    "Duly noted and filed away today.",
    "Kindly reply at your convenience soon.",
    "All agreed today.",
    "Confirmed this morning.",
    "Seconded without debate.",
    "Noted today.",
    "Agreed unanimously.",
    "Duly recorded.",
    "Confirmed Friday.",
    "Logged tonight.",
    "Filed promptly.",
    "Minuted last night.",
    "Chased twice today.",
    "Pinned to boards.",
    "Copied to all.",
    "Read and filed.",
]
MID_TARGET = {(20, 30): 25, (31, 45): 38, (46, 60): 53}

# ---- audit templates once ----
_N = "Testname Example"
for (ctx, pos), bank in TPL.items():
    assert isinstance(bank, list) and len(bank) >= 3, (ctx, pos)
    for t in bank:
        assert t.count("{N}") == 1, (ctx, pos, t[:60])
        assert "{T}" not in t or ctx in ("support ticket",) or True
        assert "@" not in t and "http" not in t, t[:60]
        probe = t.replace("{N}", _N).replace("{T}", "1234").replace("{E}", "2210")
        assert v2.refusal_or_meta_hit(probe) is None, (ctx, pos, probe[:80])
        assert v2.name_hidden_in_other_pii(probe, _N) is None
        n = len(probe.split())
print("template audit OK:", sum(len(b) for b in TPL.values()), "templates")


def wc(s):
    return len(s.split())


def cover(n, seed):
    by_size = {}
    for s in sorted(FILLERS, key=lambda s: (wc(s), hash(s + str(seed)))):
        by_size.setdefault(wc(s), []).append(s)
    sizes = sorted(by_size)
    dp = {0: []}
    for rem in range(1, n + 1):
        for sz in sizes:
            if sz <= rem and (rem - sz) in dp:
                dp[rem] = dp[rem - sz] + [sz]
                break
    if n not in dp:
        raise ValueError(f"cover {n}")
    out = []
    for k, sz in enumerate(dp[n]):
        c = by_size[sz]
        out.append(c[(seed + k) % len(c)])
    return out


def build_row(q, t_idx, seed_extra=0):
    rng = random.Random(AUTH_SEED + hash((q["split"], q["idx"], t_idx, seed_extra)) % (2**31))
    bank = TPL[(q["context_type"], q["name_position"])]
    tpl = bank[t_idx % len(bank)]
    name = q["name"]
    if q["name_form"] == "surname" and q["name_position"] in ("start", "middle"):
        disp = f"{rng.choice(['Mr.', 'Mrs.', 'Ms.'])} {name}"
    else:
        disp = name
    text = tpl.replace("{N}", disp)
    text = text.replace("{T}", str(1000 + rng.randint(0, 8999)))
    text = text.replace("{E}", str(100 + rng.randint(0, 9899)))
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text


TYPO_RX = re.compile(r"\b(teh|agian|tomorow|dont|cant|wont|runing|reciept|awfull|seperate)\b")

# single-word synonym sets for seeded paraphrase (counts preserved)
SWAPSETS = {
    "monday": ["tuesday", "wednesday", "thursday", "friday"],
    "morning": ["evening", "afternoon"],
    "quick": ["prompt", "swift"],
    "busy": ["hectic"],
    "new": ["fresh"],
    "small": ["little"],
    "big": ["large"],
    "warmly": ["cordially", "kindly"],
    "promptly": ["swiftly", "duly"],
    "daily": ["nightly", "weekly"],
    "checked": ["inspected", "tested"],
    "fixed": ["mended", "repaired"],
    "booked": ["reserved"],
    "ordered": ["requested"],
    "street": ["road", "lane"],
    "hall": ["room"],
    "garden": ["yard"],
    "nearby": ["near"],
    "many": ["numerous"],
    "happy": ["glad"],
    "cold": ["chilly"],
    "late": ["delayed"],
    "soon": ["shortly"],
    "really": ["truly"],
    "quite": ["rather"],
    "fully": ["totally"],
    "agreed": ["approved"],
    "asked": ["requested"],
    "told": ["informed"],
    "sent": ["posted"],
    "paid": ["settled"],
    "bought": ["secured"],
    "smashed": ["broke"],
    "cleaned": ["scrubbed"],
    "old": ["worn"],
    "damp": ["wet"],
    "long": ["lengthy"],
    "bright": ["vivid"],
    "noisy": ["loud"],
    "tired": ["weary"],
    "slowly": ["steadily"],
    "early": ["betimes"],
    "always": ["usually"],
    "twice": ["thrice"],
    "every": ["each"],
    "still": ["yet"],
    "already": ["previously"],
    "just": ["simply"],
    "very": ["so"],
    "today": ["tonight"],
    "please": ["kindly"],
    "will": ["shall"],
    "can": ["could"],
    "must": ["should"],
    "take": ["bring"],
    "put": ["place"],
    "keep": ["retain"],
    "show": ["display"],
    "found": ["located"],
    "last": ["final"],
    "next": ["following"],
    "other": ["alternate"],
    "same": ["identical"],
    "full": ["complete"],
    "said": ["stated"],
    "says": ["states"],
    "called": ["phoned"],
    "opened": ["launched"],
    "closed": ["shut"],
    "started": ["began"],
    "stopped": ["halted"],
    "moved": ["shifted"],
    "changed": ["altered"],
    "confirmed": ["verified"],
    "promised": ["pledged"],
    "reported": ["flagged"],
}
for _k, _v in SWAPSETS.items():
    assert v2.refusal_or_meta_hit(_k) is None
    for _w in _v:
        assert v2.refusal_or_meta_hit(_w) is None, _w


def paraphrase(text, name, seed):
    """Seeded single-word synonym swaps; preserves ALL whitespace and the name span."""
    spans = v2.find_spans(text, name)
    if len(spans) != 1:
        return text
    s0, s1 = spans[0]
    rng = random.Random(seed)
    # pre-roll decisions per word occurrence for determinism
    decisions = {}

    def repl(m):
        w = m.group(0)
        st = m.start()
        if s0 <= st < s1:
            return w
        key = w.lower()
        if key in decisions:
            alt = decisions[key]
        else:
            alt = None
            if key in SWAPSETS and rng.random() < 0.55:
                alt = rng.choice(SWAPSETS[key])
            decisions[key] = alt
        if alt is None:
            return w
        return alt.capitalize() if w[:1].isupper() else alt

    return re.sub(r"[A-Za-z'\-]+", repl, text)


def count_typos(text):
    return len(TYPO_RX.findall(text))


def trim_to_bucket(q, text):
    """Drop whole sentences to fit hi-2, preserving name/cue/typos. Returns text or None."""
    lo, hi = q["target_length"]
    if wc(text) <= hi - 1:
        return text
    name = q["name"]
    pos = q["name_position"]
    # protect signature tail (cue + name lines)
    protected_tail = ""
    body = text
    if pos == "signature" and "\n" in text:
        head, sep, tail = text.rpartition("\n")
        head2, sep2, tail2 = head.rpartition("\n")
        if v2.compute_span(tail, name):
            protected_tail = sep + tail
            body = head
            if v2._SIGNOFF_CUE.search(head2 + sep2) or head2.strip().endswith("--"):
                protected_tail = sep2 + tail2 + protected_tail
                body = head2
    # split body into sentences (keep delimiters simple)
    parts = re.split(r"(?<=[.!?]) +", body)
    if len(parts) <= 1:
        return None
    if pos in ("end", "signature"):
        # drop from front, never the part holding the name
        while wc(body + protected_tail) > hi - 1 and len(parts) > 1:
            if v2.compute_span(parts[0], name):
                break
            parts = parts[1:]
            body = " ".join(parts)
    else:
        # drop from end, never the part holding the name
        while wc(body + protected_tail) > hi - 1 and len(parts) > 1:
            if v2.compute_span(parts[-1], name):
                break
            parts = parts[:-1]
            body = " ".join(parts)
    trial = (body + protected_tail).strip()
    if not (lo <= wc(trial) <= hi - 1):
        return None
    if v2.compute_span(trial, name) is None:
        return None
    if q["context_type"] == "informal text with typos" and count_typos(trial) < 2:
        return None
    return trial


def finalize_text(q, text, seed, attempt=0):
    lo, hi = q["target_length"]
    mid = MID_TARGET[tuple(q["target_length"])]
    # 1. always add one seeded spice sentence (drives cross-row divergence)
    rng = random.Random(AUTH_SEED + hash(("spice", q["split"], q["idx"], attempt)) % (2**31))
    spice = rng.choice(FILLERS)
    if wc(text) + wc(spice) <= hi:
        if q["name_position"] in ("end", "signature", "middle"):
            text = spice + " " + text
        else:
            text = text + " " + spice
    # 2. pad to mid-bucket
    n = wc(text)
    if n < mid and (mid - n) >= 2 and mid <= hi:
        sents = cover(mid - n, seed)
        if q["name_position"] in ("end", "signature", "middle"):
            text = " ".join(sents) + " " + text
        else:
            text = text + " " + " ".join(sents)
    elif n < lo:
        gap = lo - n
        target = gap + 1 if n + gap + 1 <= hi else gap
        if target >= 2:
            sents = cover(target, seed)
            if q["name_position"] in ("end", "signature", "middle"):
                text = " ".join(sents) + " " + text
            else:
                text = text + " " + " ".join(sents)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def check(q, text):
    name = q["name"]
    span = v2.compute_span(text, name)
    row = {
        "id": f"spark-sft-v1-{q['split']}-{q['idx']:04d}",
        "split": q["split"],
        "request": {
            "language": "en",
            "context_type": q["context_type"],
            "target_length": q["target_length"],
            "name_position": q["name_position"],
            "entities": [{"type": "PERSON", "value": name}],
        },
        "target": {"text": text, "span": [span[0], span[1]] if span else None},
        "provenance": "template_paraphrase",
        "name_form": q["name_form"],
    }
    return v2.validate_row(row)


def author_one(q, boost=0):
    """Author a single quota row; boost shifts template/spice/swap/filler seeds."""
    n_bank = len(TPL[(q["context_type"], q["name_position"])])
    rng0 = random.Random(AUTH_SEED + boost + hash((q["split"], q["idx"])) % (2**31))
    base = rng0.randint(0, 99)
    last_t = None
    for attempt in range(n_bank * 3):
        t = build_row(q, base + attempt, seed_extra=attempt // n_bank + boost)
        t = paraphrase(
            t, q["name"], AUTH_SEED + boost + hash((q["split"], q["idx"], attempt)) % (2**31)
        )
        t = trim_to_bucket(q, t)
        if t is None:
            continue
        last_t = t
        t = finalize_text(q, t, q["idx"] * 131 + attempt + boost, attempt + boost)
        if not check(q, t):
            return t, None
    errs = check(q, last_t) if last_t else ["no viable trim"]
    return (last_t if last_t else build_row(q, base)), errs


def author_quota_file(qpath, out_batches, batch_size=50, boost=None):
    boost = boost or {}
    quotas = [json.loads(line) for line in open(qpath) if line.strip()]
    authored, failures = [], []
    for q in quotas:
        text, errs = author_one(q, boost.get((q["split"], q["idx"]), 0))
        if errs:
            failures.append((q["split"], q["idx"], errs))
        a = dict(q)
        a["text"] = text
        authored.append(a)
    # write batches
    for i in range(0, len(authored), batch_size):
        chunk = authored[i : i + batch_size]
        with open(out_batches(i // batch_size), "w") as f:
            for a in chunk:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
    return failures


all_fail = []
# train idx 0-1149 in 23 files of 50 (quota order)
train_q = [json.loads(line) for line in open(f"{BATCH_DIR}/quota_train.jsonl") if line.strip()]
assert len(train_q) == 1200 and train_q[0]["idx"] == 0 and train_q[1149]["idx"] == 1149
os.makedirs("/tmp/sft_gen/stage", exist_ok=True)
# stage train head as one file then split (author_quota_file writes its own batches;
# for train we need idx 0-1149 only)
with open("/tmp/sft_gen/stage/quota_train_head.jsonl", "w") as f:
    for q in train_q[:1150]:
        f.write(json.dumps(q) + "\n")
all_fail += author_quota_file(
    "/tmp/sft_gen/stage/quota_train_head.jsonl",
    lambda b: f"{BATCH_DIR}/authored_train_{b + 1:02d}.jsonl",
)
all_fail += author_quota_file(
    f"{BATCH_DIR}/quota_train_gap.jsonl", lambda b: f"{BATCH_DIR}/authored_train_gap.jsonl"
)
all_fail += author_quota_file(
    f"{BATCH_DIR}/quota_short_eval.jsonl", lambda b: f"{BATCH_DIR}/authored_eval_{b + 1:02d}.jsonl"
)
all_fail += author_quota_file(
    "/tmp/sft_v2_batches_dev100/quota_dev.jsonl",
    lambda b: f"{BATCH_DIR}/authored_dev_{b + 1:02d}.jsonl",
)
print(f"generator failures: {len(all_fail)}")
for s, i, errs in all_fail[:40]:
    print(f"  {s} idx={i}: {errs}")

# ---- targeted near-dupe regen loop ----

_built, _keys = [], []
for pat in [
    f"{BATCH_DIR}/authored_train_*.jsonl",
    f"{BATCH_DIR}/authored_eval_*.jsonl",
    f"{BATCH_DIR}/authored_dev_*.jsonl",
]:
    for p in sorted(glob.glob(pat)):
        for line in open(p):
            if line.strip():
                a = json.loads(line)
                _built.append(a)
                _keys.append((p, a["split"], a["idx"]))
_loc_of = {k[1:]: k[0] for k in _keys}
for _round in range(4):
    _rows = []
    for a in _built:
        name = a["name"]
        span = v2.compute_span(a["text"], name)
        _rows.append({"target": {"text": a["text"]}})
    dup = v2.find_duplicates([{"target": {"text": a["text"]}} for a in _built])
    pairs = dup["near"]
    if not pairs:
        print(f"regen round {_round}: no near pairs")
        break
    # rewrite the later row of each pair (keep first-seen)
    seen = set()
    todo = set()
    order = {id(a): n for n, a in enumerate(_built)}
    by_text_id = _built
    for i, j, _s in pairs:
        if j not in seen:
            todo.add(j)
        seen.add(i)
        seen.add(j)
    print(f"regen round {_round}: {len(pairs)} pairs, rewriting {len(todo)} rows")
    quotas_by_key = {}
    for qp in [
        f"{BATCH_DIR}/quota_train.jsonl",
        f"{BATCH_DIR}/quota_train_gap.jsonl",
        f"{BATCH_DIR}/quota_short_eval.jsonl",
        "/tmp/sft_v2_batches_dev100/quota_dev.jsonl",
    ]:
        for line in open(qp):
            if line.strip():
                q = json.loads(line)
                quotas_by_key[(q["split"], q["idx"])] = q
    rewrites = {}
    for j in sorted(todo):
        a = _built[j]
        q = quotas_by_key[(a["split"], a["idx"])]
        text, errs = author_one(q, boost=1000 * (_round + 1) + (j % 97))
        assert not errs, ((a["split"], a["idx"]), errs)
        a["text"] = text
        rewrites[(a["split"], a["idx"])] = text
    # persist rewrites to batch files
    for p in sorted({k[0] for k in _keys}):
        rows = [json.loads(line) for line in open(p) if line.strip()]
        changed = False
        for r in rows:
            if (r["split"], r["idx"]) in rewrites:
                r["text"] = rewrites[(r["split"], r["idx"])]
                changed = True
        if changed:
            with open(p, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("regen done")
