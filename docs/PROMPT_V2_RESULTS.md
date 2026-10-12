# PROMPT_V2_RESULTS — M07b: prompt variants vs template collapse (base Qwen3.5-2B)

**Status: prepared, run pending owner go.** Everything in this file up to §7 is
final and committed BEFORE any generation run (pass criteria included). §7-§9
are filled from `prompt_compare_report.json` after the run; the
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

## 8. Results — PENDING RUN

To be filled from `prompt_compare_report.json` (validity/failure breakdown,
name-at-start, openings, distinct-1/2, unique-per-group, length distribution,
p_target/live-group flagged pending scorer verification, PASS BY CELL table,
10 random samples per variant at t=1.0, recommendation).

## 9. What is (still) unverified

- All generation numbers in §8: the run has not executed yet (owner go
  pending; local CPU ~2-3 h projected vs Colab T4 ~20-40 min).
- Every p_target / live-group / outcome number once produced: pending scorer
  verification until the M07a scorer fix lands and is re-run over
  `samples_raw.jsonl`.
- The 6th pilot sample (see §2); the complete pilot file remains on Drive.
- Colab notebook path (Drive D0 / config overrides) is untested until the
  owner opens it — it mirrors the M07a launcher pattern exactly.
