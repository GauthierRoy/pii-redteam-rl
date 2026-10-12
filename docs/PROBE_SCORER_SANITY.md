# PROBE_SCORER_SANITY — verdict: **REAL SHIFT, not a scorer bug** (2026-10-11)

Investigation branch `fix/probe-scorer-sanity`. Question: the M07a probe reported
complete misses on trivial short texts ("Hey Lucas, how are you doing today?") —
is that a probe-scorer bug, or does frozen D0 genuinely fail on short messages?

**Verdict: the scorer is correct; D0 genuinely misses short conversational names.**
The probe scorer and the D0-training eval pipeline agree on every single sample.

## Evidence

### A. Probe scorer == training eval pipeline (AI4Privacy dev rows)

50 seeded (20260920) dev rows with gold PERSON spans, none in final-eval
(asserted; dev rows cannot be by construction). 124 gold spans scored by BOTH:

| Pipeline | exact_hit | partial | boundary | miss |
|---|---|---|---|---|
| probe scorer (`_score_with_d0`) | 96.77% | 0.81% | 0.0% | 2.42% |
| training eval pipeline | 96.77% | 0.81% | 0.0% | 2.42% |

**Outcome agreement: 124/124 spans (100%).** The probe's dev hit rate matches D0's
recorded dev span recall (0.968). p_target_mean on exact hits: median 0.9978
(min 0.58) — a sane difficulty signal.

### B. Pilot-like short texts: same agreement, real misses

32 regenerated pilot texts (same 8 bank requests, t=1.0, n=4, seed 20260920+10;
local CPU regeneration — not the exact Colab 32, same protocol):

| Pipeline | exact_hit | partial | miss |
|---|---|---|---|
| probe scorer | 28.1% | 18.8% | **53.1%** |
| training eval pipeline | 28.1% | 18.8% | **53.1%** |

Row-by-row: **32/32 agree.** The probe's reported misses are genuine model misses.

### C. Controlled pairs: the model's own probabilities are low

Full token/offset/label dumps for the controlled sentences (all in the run log):

- "Lucas called me today." → tokens `Luc`(0,3) `as`(3,5): p(B-PERSON) = 0.060 / 0.013.
- "Yesterday, Lucas called me." → `ĠLucas`(10,16): p(B-PERSON) = 0.079.
- "Hey Lucas, how are you?" → `Lucas`: p(B-PERSON) = 0.023.
- "I spoke with Lucas Martin yesterday." → `Lucas` p(B) = 0.020, `Martin` p(I) = 0.016.

Argmax is O on every token; even a single clean `ĠLucas` token gets < 0.08 for
B-PERSON. No decode, alignment, or padding artifact can produce that — the model
itself assigns these probabilities. **Domain shift is real:** D0 was trained on
long AI4Privacy form-letters ("Dear …, Social Number: …"), and short
conversational messages are out of that distribution.

### D. Explicit checks (all clean)

- **Special tokens:** CLS/SEP carry (0,0) offsets and are labeled O — identical to
  D0 training (training labels used the same (0,0) → -100/O convention).
- **id2label:** `{0: 'O', 1: 'B-PERSON', 2: 'I-PERSON'}` — matches training exactly.
- **p_target semantics:** reads the per-token GOLD-label probability (B-PERSON for
  the first target token, I-PERSON after) — not the summed PERSON mass. Summing
  would not change any outcome (outcomes are argmax-decoded) and the recorded
  B+I probabilities on missed names are both tiny (e.g. 0.060+0.007).
- **Char→token alignment:** name-at-text-start works ("Lucas…" → gold labels
  B-PERSON, I-PERSON on tokens (0,3),(3,5)); the BPE leading-space quirk only
  affects mid-text words and `trim_span_whitespace` handles recovery — verified
  in the dev-row agreement above.
- **Single-token names:** 20 sampled single-token gold names → 0 partial_overlap.

## Consequences

1. **p_target and the probe outcomes are trustworthy.** No fix applied to the
   scorer; nothing to re-score.
2. **The M07a reward signal is strong by construction:** ~53% of valid pilot
   samples are complete misses with low p_target — high within-group reward
   variance is expected, and the go/no-go "miss rate 1–30%" OR "live-group"
   framing needs revisiting upward (the pilot miss rate ALONE exceeds the 30%
   band). This is good news for RL: there is real headroom to optimize.
3. **D0's short-message blind spot is the experiment's target**: the augmentation
   arms exist precisely to teach the detector short-conversational names.

## Regression guard

`tests/integration/test_probe_scorer_regression.py` (+ fixture
`tests/fixtures/probe_scorer_regression.json`): 5 fixed AI4Privacy dev rows that
must score exact_hit with p_target > 0.9, and one guard asserting the
"Hey Lucas…" sentence stays a complete_miss (if that ever flips, D0's behavior
changed and the reward design must be revisited). Requires `PII_D0_PATH` env var
pointing at the frozen d0-best; skipped otherwise.

## Commands run / environment

- `uv run --group ml python /tmp/sanity/probe_scorer_sanity.py` (laptop CPU,
  frozen D0 sha-verified `0314ebe3…`, transformers 5.18.0, torch 2.14.1+cpu).
- Outputs: `/tmp/sanity_results.json` (rates + pilot texts), full dumps in the log.

## Still unverified

- The exact 32 Colab pilot texts were not re-scored (regenerated locally instead —
  same requests/seed/protocol; agreement logic is text-agnostic, and dev-row
  agreement is the decisive check).
- GPU (fp16) scoring path: verified on CPU fp32; the probe's determinism check on
  Colab covers this at run time.
