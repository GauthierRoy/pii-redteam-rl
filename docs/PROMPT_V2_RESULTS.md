# PROMPT_V2_RESULTS — M07b: prompt variants vs template collapse (base Qwen3.5-2B)

**Status: EXECUTED 2026-10-12/13, run `m07b-20261012-224819` (local CPU).**
Everything up to §7 was final and committed BEFORE the generation run (pass
criteria included); §8-§9 are filled from `prompt_compare_report.json`
(committed at `results/m07b_prompt_v2/m07b-20261012-224819/`), and the
`EXPERIMENTS.yaml` entry is appended from the report's `experiments_entry`.

Question: can prompting alone make the BASE Qwen3.5-2B (no adapter,
non-thinking) produce varied, realistic short messages instead of
"Name said..." / "Hello Name, how are you?" — or is SFT v2 on Spark-written
short data required?

Branch: `feat/request-v2-prompting`. The M07a probe scorer is NOT touched
(separate fix in flight); all D0-dependent numbers here are flagged
**pending scorer verification**.

## 1. Stricter validity rules (v2)

Defined in `src/pii_redteam/validity_v2.py`, applied to every prompt variant
and used to re-label the M07a pilot samples. On top of the v1 contract (JSON
object with a string `text` field, requested name exactly once):

- at least 8 words AND at least one clause of content besides the name
  (pure greetings / small talk do not count as content)
- the name is not the whole text; the label form "Name:" is rejected
- refusal / meta-answer detector: "i cannot", "i can't", "as an ai",
  "i do not know what", "please provide", "i'm sorry", "unable to comply",
  "cannot comply/fulfill/generate/create", "ethical guidelines", etc.
- hard length cap: 80 words

v2 failure statuses: `invalid:truncated`, `invalid:json`, `invalid:name_missing`,
`invalid:name_dup`, `invalid:name_only`, `invalid:refusal`, `invalid:too_short`,
`invalid:no_content`, `invalid:too_long`.

The v1 classifier (`probe_metrics.classify_sample`) is unchanged; the M07a
scorer keeps its own definitions. Re-labeling any samples file:
`uv run python scripts/relabel_v2.py --samples <samples.jsonl>`.

## 2. M07a pilot samples re-labeled under v2

The M07a "pilot" is the local CPU smoke run `m07a-smoke-20261011-093350`
(3 requests x n=2, t=1.0, base 2B). Its `samples.jsonl` (`/tmp/probe_out/...`)
was deleted before this task; the five samples recoverable from the recorded
panel output are saved at `tests/fixtures/m07a_pilot_recovered.jsonl`
(raw completions reconstructed as strict-JSON wrappers; sample
`seed-20260920-0001 #0` was never shown in any panel and is lost).

Re-label result (`scripts/relabel_v2.py`): **5 v1-valid -> 3 v2-valid.**

| Sample | p_target | v1 outcome | v2 status | Why |
|---|---|---|---|---|
| 'Lucas, do you have any luck with the project today?' | 0.008 | complete_miss | **valid** | 10 words, real content |
| 'Ifigeneia wrote an email to ask about the new project timeline.' | 0.153 | complete_miss | **valid** | 12 words, real content |
| 'Paheeswaran mentioned that the weather is perfect.' | 0.682 | exact_hit | invalid:too_short | 7 words |
| 'Paheeswaran contacted the office today regarding the pending report.' | 0.742 | exact_hit | **valid** | 9 words, content |
| 'Hello, Ifigeneia, how are you?' | 0.801 | exact_hit | invalid:too_short | 6 words, greeting only |

**Hard-sample answer:** the genuinely hard pilot samples (the two
complete-miss outputs, lowest p_target) stay VALID under v2. What v2
invalidates is the easy template output — the exact_hit greetings/one-liners.
The v2 rules cut exactly the "Name said..." / "Hello Name..." collapse the
M07b experiment is trying to eliminate. Caveat: only 5 of 6 pilot samples are
recoverable; the full pilot `samples.jsonl` (on Drive, once exported) can be
re-labeled with the same script for the complete picture.

## 3. Request schema v2

`src/pii_redteam/requests.py`: records with `schema_version: 2` must carry

- `task.context_type` from the 10-type catalogue: support_ticket,
  email_with_signature, forwarded_email_thread, meeting_notes, chat_message,
  appointment_reminder, form_record_line, list_or_table_row,
  informal_text_with_typos, third_person_report
- `task.target_length` = `{min_words, max_words}` (e.g. 20-60)
- optional `task.name_position` hint: start / middle / end / signature

v1 records (no `schema_version`) validate exactly as before; the frozen v1
bank (sha `332a9eb0…dc58c`) is untouched. Both request sources are supported:
Spark-authored import (`load_bank`, provenance-gated) and the seeded sampler
(`seeded_sampler(..., task_extras=...)` emits v2 records). R06/R07 pool
enforcement applies to both.

## 4. Bank v2 (20 requests)

`banks/v2_prompting/spark_seed20260920_v2_n20.jsonl` (committed; the saved bank
is the artifact of record, rebuilt deterministically by
`scripts/build_request_bank_v2.py`):

- **sha256:** `1f0b9960ac797c3fbbc5e26dc2658a11e76837ca47a79da9b0443d37a7708fb7`
- 20 requests, names drawn with seed 20260920 from the 16,329 requestable
  train-side values only (verified: 20/20 in train_side_pool, 0 held-out,
  0 final-eval)
- all 10 context types, 2 rows each; target lengths 15-60 words (default
  20-60); name_position hints: middle x6, signature x2 (email rows),
  start x4, end x4, none x4
- manifest: `banks/v2_prompting/spark_seed20260920_v2_n20_manifest.json`

## 5. Prompt variants (same JSON output contract for all)

Rendered by `src/pii_redteam/prompt_v2.py:render_prompt_v2(record, variant)`;
`max_new_tokens=192` (fits the 80-word cap + JSON wrapper).

- **P0 (baseline)** — the current v1 prompt, v2 fields ignored:

```
Write one short realistic message.
Preserve each supplied name exactly as given, exactly once.
Introduce no additional person names.
Return only a JSON object with a 'text' field.
Entity PERSON: Brayn
```

- **P1 = P0 + context_type + target_length (+ name_position hint when the
  request carries one):**

```
Write one short realistic message.
Preserve each supplied name exactly as given, exactly once.
Introduce no additional person names.
Return only a JSON object with a 'text' field.
Entity PERSON: Brayn
Context type: a casual chat message
Target length: 20-50 words.
Place the name near the end.
```

- **P2 = P1 + 2 few-shot examples** (chat message + support ticket; fixed
  fictional names `Dorotea Lunvik` / `Keferis Onda`, verified in NO pool —
  not train-side, not held-out, not final-eval — name never at the start):

```
... (P1 lines) ...
Two examples of the style (different context types):
- a casual chat message: {"text": "Quick heads-up - rehearsal moved to Thursday 7pm at the studio, and Dorotea Lunvik asked everyone to bring the new scores."}
- a customer support ticket: {"text": "Order #48213 has not shipped after ten days and the tracking page errors out; customer requests a refund to the original card, filed by Keferis Onda on behalf of the account holder."}
```

## 6. Run plan and metrics

Runner: `python -m pii_redteam.cli compare-prompts --config configs/m07b_prompt_v2.yaml`
(3 variants x 2 temperatures {1.0, 0.7} x 20 requests x n=8 = 960 generations,
top_p 0.95, seed 20260920). ALL raw outputs are saved to
`<work_dir>/m07b-<ts>/samples_raw.jsonl` (valid + invalid, raw completion
text, token counts, truncation flag) so the run can be re-scored later without
regenerating. Colab launcher: `colab/M07b_prompt_compare.ipynb`.

Per variant x temperature:

- v2 validity rate + failure breakdown (per failure status)
- share of valid outputs starting with the name
- name-position distribution (start / middle / end, name midpoint fraction)
- top-10 opening-pattern share (normalized first 5 words) + the patterns
- distinct-1 / distinct-2 over valid texts
- median unique valid texts per 8-sample group (+ min)
- length distribution in words (p10/p50/p90, min/max)
- D0 p_target percentiles, outcome rates, live-group shares — **flagged
  `pending scorer verification`** (computed with a local reference-pipeline
  replica of the D0 eval path, NOT the M07a probe scorer under repair)

## 7. Pass criteria (pre-committed BEFORE the run)

| Criterion | Threshold |
|---|---|
| v2 validity rate (per variant x temperature) | >= 0.90 |
| name-at-start share (valid outputs) | <= 0.40 |
| top-10 opening-pattern share | <= 0.30 |
| median unique valid per group of 8 | >= 6 |

A variant passes only if its cell meets all four at some temperature. Best
passing cell = highest validity, then lowest name-at-start, then lowest
top-10 share. **If no variant passes: that is the agreed trigger for SFT v2 on
Spark-written short data — not more prompt hacking.**

## 8. Results — run `m07b-20261012-224819` (executed 2026-10-12/13)

**Run:** local laptop CPU (no GPU on this machine), 16 cores, fp32, torch
2.14.1+cpu, transformers 5.18.0, python 3.12.3, repo commit `7193374`.
Command:

    PYTHONPATH=src uv run --group ml python -m pii_redteam.cli \
        compare-prompts --config configs/m07b_prompt_v2.yaml

960 generations (3 variants x {1.0, 0.7} x 20 requests x n=8, top_p 0.95,
seed 20260920, max_new_tokens 192) in **58.8 min** (3526 s, ~3.7 s/generation
effective with batch=4 x n=8); frozen-D0 scored 668 valid samples. ALL raw
outputs saved: `results/m07b_prompt_v2/m07b-20261012-224819/samples_raw.jsonl`
(960 records incl. every invalid, re-scoreable) + `prompt_compare_report.json`.

### 8.1 Headline: P1 PASSES at both temperatures; P0 and P2 fail

**Recommendation (from the report): variant cell P1|t0.7 passes all
pre-committed criteria — candidate prompt for M06/M08.** Prompting alone IS
enough; the SFT-v2 trigger does NOT fire.

PASS BY CELL (criteria pre-committed in §7):

| Cell | validity >=0.90 | name-start <=0.40 | top10 <=0.30 | uniq >=6/8 | PASS |
|---|---|---|---|---|---|
| P0 t=1.0 | 0.3063 | 0.4898 | 0.2041 | 3 | FAIL |
| P0 t=0.7 | 0.2750 | 0.6364 | 0.2955 | 2 | FAIL |
| **P1 t=1.0** | **0.9000** | 0.2569 | 0.0833 | 7 | **PASS** |
| **P1 t=0.7** | **0.9375** | 0.3667 | 0.1733 | 8 | **PASS** |
| P2 t=1.0 | 0.8313 | 0.3910 | 0.0827 | 7 | FAIL (validity) |
| P2 t=0.7 | 0.9250 | 0.4189 | 0.1216 | 8 | FAIL (name-start) |

Best passing cell: **P1|t0.7** (highest validity 0.9375, median unique 8/8,
name-at-start 0.367).

### 8.2 Per-cell metrics

| Metric | P0 t1.0 | P0 t0.7 | P1 t1.0 | P1 t0.7 | P2 t1.0 | P2 t0.7 |
|---|---|---|---|---|---|---|
| v2-valid / 160 | 49 | 44 | 144 | 150 | 133 | 148 |
| failure breakdown | ts 83, no 17, nm 8, nc 3 | ts 90, no 16, nm 6, nc 4 | j 4, nm 6, nd 3, r 3 | nd 5, nm 4, r 1 | nm 23, nd 2, j 1, no 1 | nm 9, r 2, nd 1 |
| name-at-start share | 0.4898 | 0.6364 | 0.2569 | 0.3667 | 0.3910 | 0.4189 |
| name position start/mid/end | .84/.14/.02 | .93/.07/.00 | .72/.24/.05 | .83/.15/.01 | .80/.17/.04 | .82/.16/.01 |
| top-10 opening share | 0.2041 | 0.2955 | 0.0833 | 0.1733 | 0.0827 | 0.1216 |
| distinct-1 / distinct-2 | .495/.866 | .452/.727 | .328/.762 | .265/.623 | .306/.739 | .225/.585 |
| median (min) unique valid / group | 3 (1) | 2 (1) | 7 (5) | 8 (5) | 7 (5) | 8 (5) |
| length words p10/p50/p90 (max) | 8/10/15 (32) | 8/9/14 (29) | 12/20/30 (62) | 13/21/30 (58) | 15/22/29 (49) | 16/23/30 (45) |

ts = invalid:too_short, no = name_only, nm = name_missing, nc = no_content,
j = json, nd = name_dup, r = refusal. Groups = 20 per cell except P0 (19 — one
request produced no valid output at all).

Reading:
- **P0 (baseline) confirms the diagnosis**: ~70% of outputs are too short /
  name-only / greeting-only; name-at-start 49-64%; median unique 2-3/8. The
  v1 prompt cannot produce varied realistic short messages.
- **P1 (context_type + target_length + position hint) fixes it**: validity
  0.90-0.94, name-at-start 0.26-0.37, top-10 openings 0.08-0.17, median
  unique 7-8/8, lengths 12-30 words (inside the 15-60 targets). Remaining
  failures are contract-level (name_dup 5, name_missing 4-6, json 4, refusal 1-3).
- **P2 (few-shot) is NOT better than P1**: name fidelity degrades badly at
  t=1.0 (23 name_missing — the model substitutes other names, sometimes the
  few-shot names themselves, e.g. "Doris", "Elena Varga", "Dorotea Lunvik"),
  and name-at-start stays ~0.39-0.42. Few-shot examples trade contract
  compliance for style; P1 achieves the style without that cost.
- distinct-1 is LOWER for P1/P2 than P0 — valid P1/P2 outputs are longer and
  share support-ticket/email vocabulary ("please", "regarding", "team"), so
  token-level distinct-1 falls even as template collapse (openings) improves
  by 2-3x. distinct-1 is not among the pass criteria; the opening-pattern and
  unique-per-group metrics are the ones that measure the actual failure mode.

### 8.3 p_target / live-group stats — **pending scorer verification**

Computed with a local reference-pipeline replica of the frozen-D0 eval path
(sha-verified `0314ebe3…c43cb`), NOT the M07a probe scorer (under separate
fix). Do not use these numbers for go/no-go until re-scored:

| Metric | P0 t1.0 | P0 t0.7 | P1 t1.0 | P1 t0.7 | P2 t1.0 | P2 t0.7 |
|---|---|---|---|---|---|---|
| p_target mean p50 | 0.470 | 0.611 | 0.628 | 0.652 | 0.640 | 0.623 |
| complete_miss rate | 0.388 | 0.273 | 0.285 | 0.273 | 0.241 | 0.297 |
| exact_hit rate | 0.367 | 0.341 | 0.465 | 0.460 | 0.444 | 0.419 |
| live-group share (r_prob) | 0.474 | 0.368 | 0.900 | 0.850 | 0.950 | 0.950 |
| dead groups | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Directionally: valid P1/P2 outputs sit in a p_target band (~0.2-0.9) that
gives much stronger within-group reward variance than P0 (live-group 0.85-0.95
vs 0.37-0.47) — a healthy RL training signal, if it survives scorer
verification.

### 8.4 Manual-review samples (10 random valid per variant, t=1.0)

P0 (all short template one-liners):

- 'Rahima Haudebert Sinueno is going to the party.'
- 'Fithi Bettschen has been contacted today to discuss the event schedule.'
- 'Hi Val, hope you are having a good day.'
- 'Hi Jahan, this is my short realistic message.'
- 'Brayn has just arrived at the venue for the evening gala.'
- 'Gulzar has written an excellent review of the new film.'
- 'Jürg wrote a short message about the day.'
- 'Hi Düriye, did you hear the news about the market?'
- 'I received an unexpected text from Shelibare Soraya Posteraro asking to speak with me.'
- 'Lelouch-Ferdinand is back, ready to take the stage once again.'

P1 (varied, realistic, on-context):

- 'Hi Jürg, this email forwarded by Sarah regarding the project deadline is urgent...' (note: extra name "Sarah" — D0-flag territory, still v2-valid)
- 'Dear Mermolja, we have received your support request regarding the service issue you mentioned...'
- "Hey, Brayn, how's your day going? Need to grab a quick coffee sometime."
- 'Hello, this is Düriye regarding the recent issue with our billing process...'
- 'Hey, got your order already, Lelouch-Ferdinand. Just dropped it in the box...'
- 'Rahima Haudebert Sinueno completed her submission on the form line.'
- 'Today we briefly discussed how Gulzar and I should proceed with the meeting notes...'
- 'Dear Presset, just wanted to follow up on the meeting scheduled for next Tuesday...'
- 'The system logs a new Trémolat entry for processing in the main database.'

P2 (style-rich but leaky):

- "Reminder: Guled Grayson's meeting with the team is scheduled for next Tuesday at 3 PM in Conference Room B."
- "URGENT: Jürg's meeting details have changed. The session moves to Friday 2pm..."
- 'Hey team, just a quick note before meeting tonight... Lelouch-Ferdinand insisted we skip it.'
- 'Good morning! Did you see the new specs for the project? Al-Asadi is ready...'
- "Val and Dorotea Lunvik confirmed the schedule adjustment..." (few-shot name leaked in)

### 8.5 Known detector issues found by this run (quantified, as-run numbers kept)

- **Refusal-detector false positives** (6 hits across cells; at least 4 are
  in-message phrases, not refusals): "I can't wait to see how the feedback
  comes in", "I can't figure out what it is", "Please provide your reference
  number so we can look into it" inside otherwise realistic support emails.
  All 3 P1|t1.0 refusals look like false positives — with them relabeled
  valid, P1|t1.0 validity would be ~0.94 instead of 0.90 (still passes).
  Rule was pre-committed, so as-run numbers stand; tightening "i can't" /
  "please provide" to refusal-at-start-of-answer only is a follow-up owner
  decision.
- **Greeting-lexicon gap** ("having" was missing): would have flipped exactly
  one P0|t1.0 sample ("Hi Val, hope you are having a good day."). No
  pass/fallout impact; lexicon fixed in code after the run.
- P1 shows occasional extra names ("Sarah") inside otherwise valid outputs —
  that is the D0 extra-person-flag channel's job to police, not v2 validity.

## 9. What is (still) unverified

- Every p_target / outcome / live-group number in §8.3: **pending scorer
  verification** until the M07a scorer fix lands and is re-run over
  `samples_raw.jsonl` (raw outputs are saved for exactly this).
- The run is CPU fp32 on this laptop, not a GPU: sample-level outputs differ
  from what a T4/fp16 run would produce, but the variant ordering (P1 >> P2 >
  P0 on the pre-committed criteria) has wide margins and is unlikely to flip.
  If the owner wants GPU numbers, `colab/M07b_prompt_compare.ipynb` runs the
  same committed config on a T4 (~20-40 min).
- The 6th pilot sample (see §2); the complete pilot file remains on Drive.
- Colab notebook path (Drive D0 / config overrides) is untested until the
  owner opens it — it mirrors the M07a launcher pattern exactly.
- Refusal-detector scope (see §8.5) — owner decision pending.
