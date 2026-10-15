# PROMPT_V2_RESULTS — M07b: prompt variants vs template collapse (base Qwen3.5-2B)

**Status: EXECUTED 2026-10-12/13, run `m07b-20261012-224819` (local CPU);
re-scored 2026-10-13 under the owner decisions D14/D15 (official probe scorer,
tightened refusal rule, position compliance, independent person checks).**
Everything up to §7 was final and committed BEFORE the generation run (pass
criteria included); §8-§10 are filled from `prompt_compare_report.json` and
`rescore_report.json` / `person_check_report.json` (committed under
`results/m07b_prompt_v2/m07b-20261012-224819/`), and the `EXPERIMENTS.yaml`
entries are appended from those reports' `experiments_entry` fields.

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

### 8.3 p_target / live-group stats — official probe scorer (flag dropped)

Re-scored 2026-10-13 with the OFFICIAL probe scorer
(`pii_redteam.probe._score_with_d0`, frozen D0 sha `0314ebe3…c43cb`), which the
M07a scorer-sanity work verified against the D0 training eval pipeline
(dev 124/124, pilot 32/32 — PR #5). Cross-check on this run: the as-run local
reference-replica and the official scorer agree to max |Δp_target| = 8.05e-07
(mean 1.38e-08) over 667 common samples, so the as-run numbers stand. Numbers
below are the official-scorer values over the D15-valid samples:

| Metric (official scorer, D15-valid) | P0 t1.0 | P0 t0.7 | P1 t1.0 | P1 t0.7 | P2 t1.0 | P2 t0.7 |
|---|---|---|---|---|---|---|
| p_target mean p50 | 0.519 | 0.611 | 0.621 | 0.652 | 0.640 | 0.623 |
| complete_miss rate | 0.375 | 0.273 | 0.286 | 0.272 | 0.241 | 0.293 |
| live-group share (r_prob) | 0.474 | 0.368 | 0.900 | 0.850 | 0.950 | 0.950 |
| dead groups | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Directionally: valid P1/P2 outputs sit in a p_target band (~0.2-0.9) that
gives much stronger within-group reward variance than P0 (live-group 0.85-0.95
vs 0.37-0.47). Note (D13): D0 still misses ~24-39% of these short-conversational
names — complete_miss is a REAL detector gap, not a scoring artifact, which is
exactly why D13 stages D1 before any RL and why the M07b.2 data gates
(§10) do not rely on D0.

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

### 8.4b D15 re-label of the saved 960 (both numbers)

Owner decision D15 (2026-10-13): refusal patterns flag only refusal/meta-text at
the answer start or refusal-only answers. Re-labeled via
`scripts/rescore_v2.py` over `samples_raw.jsonl`:

| Cell | validity as-run | validity D15 | flips |
|---|---|---|---|
| P0 t1.0 | 0.3063 | 0.3000 | 1 valid -> invalid:no_content (greeting-lexicon "having" fix) |
| P0 t0.7 | 0.2750 | 0.2750 | - |
| **P1 t1.0** | 0.9000 | **0.9187** | 3 invalid:refusal -> valid (in-message "I can't wait"/"please provide") |
| **P1 t0.7** | 0.9375 | **0.9437** | 1 invalid:refusal -> valid |
| P2 t1.0 | 0.8313 | 0.8313 | - |
| P2 t0.7 | 0.9250 | **0.9375** | 2 invalid:refusal -> valid |

All 6 as-run refusal labels that flipped were false positives (confirmed by
manual reading); P1 passes at both temperatures with wider margins under D15.
Pass/fallout table unchanged.

### 8.4c Per-hint name-position compliance (owner request, 2026-10-13)

Compliance of D15-valid outputs with the request's `name_position` hint
(`prompt_v2.hint_compliance`; P0 ignores hints, so it is the control):

| Hint | P0 t1.0 | P1 t1.0 | P2 t1.0 | P0 t0.7 | P1 t0.7 | P2 t0.7 |
|---|---|---|---|---|---|---|
| start | 0.83 | 0.94 | 0.92 | 1.00 | 1.00 | 0.80 |
| middle | 0.18 | 0.48 | 0.19 | 0.07 | 0.31 | 0.11 |
| end | 0.00 | 0.14 | 0.09 | 0.00 | 0.07 | 0.04 |
| signature | 0.00 | 0.13 | 0.07 | 0.00 | 0.00 | 0.07 |

Finding: the model follows `start` hints almost perfectly, but `middle`/`end`/
`signature` hints are weakly followed — the name lands early in most outputs
regardless (name-position start-share 0.72-0.93 across cells). Position hints
beyond `start` should not be relied on for D1 data construction.

### 8.5 Known detector issues found by this run (quantified; D15 resolved them)

- **Refusal-detector false positives** (6 hits across cells, all confirmed
  false by manual reading): in-message "I can't wait…", "please provide your
  reference number" inside realistic support emails. Resolved by D15
  (answer-start / refusal-only scope); see §8.4b for both numbers.
- **Greeting-lexicon gap** ("having" was missing): flipped exactly one P0|t1.0
  sample. Lexicon fixed in code; the D15 re-label applies it (§8.4b).
- P1 shows occasional extra names ("Sarah") inside otherwise valid outputs —
  policed by the independent D1 data gate (§10), not by v2 validity.

## 9. What is (still) unverified

- p_target / outcome / live-group numbers (§8.3) now come from the official
  probe scorer (verified by the M07a scorer-sanity regression, PR #5); the
  remaining caveat is D0 itself (D13: ~24-39% complete-miss on these texts) —
  by design, D1 exists to close that gap, and the D1 gates do not rely on D0.
- The person-sense judge never rejected (0/673): its discrimination is
  unvalidated — no counterexample appeared in this run or the audit.
- The LLM judge is Qwen3.5-0.8B (owner asked for a small judge; the 2B judge
  was tried first but was ~15x slower on CPU). A stronger judge could reduce
  the 87 unclear answers (12.9% overall, concentrated in P2 texts).
- The run is CPU fp32 on this laptop, not a GPU: sample-level outputs differ
  from what a T4/fp16 run would produce, but the variant ordering (P1 >> P2 >
  P0 on the pre-committed criteria) has wide margins and is unlikely to flip.
  If the owner wants GPU numbers, `colab/M07b_prompt_compare.ipynb` runs the
  same committed config on a T4 (~20-40 min).
- The 6th pilot sample (see §2); the complete pilot file remains on Drive.
- Colab notebook path (Drive D0 / config overrides) is untested until the
  owner opens it — it mirrors the M07a launcher pattern exactly.

## 10. M07b.2 D1 training-data gates (owner decisions, 2026-10-13)

Before ANY generation from this run becomes D1 training data (D13 Stage 1):

1. **Extra-person check — independent of D0** (R08). Checker =
   spaCy `en_core_web_sm` NER (filtered: digit-strings and role titles like
   "Team Lead" are not person names) OR a small LLM judge (Qwen3.5-0.8B,
   bf16 CPU, greedy, non-thinking, 2-token yes/no). Policy: **conservative —
   an unclear judge answer rejects the row** (the audit showed the leaked
   few-shot name sits exactly in that bucket). "Introduce no additional person
   names" was already in the P0/P1/P2 prompt contract (§5), so no prompt
   change was needed.
2. **Person-sense check** (name = a person, not a place/organization/object/
   work): LLM judge + manual audit of 50 stratified rows.

Executed over the 673 D15-valid samples of this run
(`scripts/person_check_v2.py`, judge pass 10.2 min CPU; outputs:
`samples_person_checked.jsonl`, `person_check_report.json`):

| Cell | n valid | spaCy extras | LLM yes (unclear) | sense "no" | accepted for D1 |
|---|---|---|---|---|---|
| P0 t1.0 | 48 | 0.000 | 0.021 (0) | 0.000 | 47/48 (0.979) |
| P0 t0.7 | 44 | 0.000 | 0.000 (0) | 0.000 | 44/44 (1.000) |
| **P1 t1.0** | 147 | 0.095 | 0.000 (14) | 0.000 | 132/147 (0.898) |
| **P1 t0.7** | 151 | 0.093 | 0.000 (17) | 0.000 | 133/151 (0.881) |
| P2 t1.0 | 133 | 0.053 | 0.015 (34) | 0.000 | 95/133 (0.714) |
| P2 t0.7 | 150 | 0.033 | 0.007 (22) | 0.000 | 127/150 (0.847) |
| **overall** | 673 | 0.062 | 0.006 (87) | 0.000 | **578/673 (0.859)** |

Under the lenient policy (unclear counts as clean): rejection 6.4%,
acceptance 93.6%. The conservative policy costs ~8 points of yield for zero
gate misses (see audit).

**Manual audit of 50 stratified rows (25 flagged / 25 clean; agent-reviewed,
verdicts in `audit_manual_verdicts.jsonl`):**
- final gate agreement **46/50 (92%)**; **0 false acceptances** (no extra
  person slipped through), 4 over-rejections (2 LLM judge false "yes" on
  texts that contain only the requested name; 2 role-title rows rejected only
  by the conservative unclear-reject policy)
- person-sense: the judge answered "yes" on all 673 samples; the audit
  confirms all 50 audited rows do use the name as a person — the check never
  fired on this run (0% rejection), so its discrimination is UNVALIDATED
  (no counterexamples existed to test it on; keep it as a guard)
- the audit caught 2 rows where the P2 few-shot name "Dorotea Lunvik" leaked
  into outputs and spaCy missed it — the conservative unclear-reject policy
  is what catches that class; also another argument for P1 (adopted, D14)
  over P2

For the adopted configuration (D14: P1 at t=0.7): **88.1% of D15-valid
outputs would be accepted as D1 data** after the person gates.
