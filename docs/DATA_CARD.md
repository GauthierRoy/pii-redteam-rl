# DATA_CARD — ai4privacy/pii-masking-300k (English pilot)

- **Source:** `ai4privacy/pii-masking-300k` (public, ungated)
- **Revision pinned:** `c8c77895a005822682b66ab547fc0422579bc1d3` (lastModified 2026-06-03)
- **License file:** `LICENSE.md` (`license: other`). Owner reports personal approval from the
  AI4Privacy team (D03, recorded once); the file itself is not re-interpreted here.
- **Pilot files (streamed 2026-09-20, not stored):**

| File | Split field | Rows | sha256 (prefix) | Masks | Offset mismatches |
|---|---|---|---|---|---|
| `data/train/1english_openpii_30k.jsonl` | `train` | 29,908 | `5b39bb2f…` | 193,086 | 909 (0.47%) |
| `data/validation/1english_openpii_8k.jsonl` | `validation` | 7,946 | `3112f549…` | 51,184 | 113 (0.22%) |

- **Note:** file names advertise 30k/8k; actual rows are 29,908 / 7,946. Do not trust advertised counts.
- **Dataset is itself synthetic** (template artifacts like `[GIVENNAME_B(…` leak into some rows).
  This study is targeted synthetic augmentation of an existing synthetic dataset.

## Observed schema (all 9 keys, every sampled row)

| Field | Type | Meaning |
|---|---|---|
| `source_text` | str | Original text with PII present (may itself be serialized JSON) |
| `target_text` | str | Masked text with `[LABEL]` placeholders |
| `privacy_mask` | list[{value, start, end, label}] | Char offsets into `source_text` (authoritative; verified) |
| `span_labels` | str (serialized JSON) | Same spans as `[[start, end, label], …]`, reverse order — parse with `json.loads`, never `eval` |
| `mbert_text_tokens` / `mbert_bio_labels` | parallel lists | mBERT tokenization — **do not reuse with another tokenizer** (plan §4.2) |
| `id` | str | Record id (`example_id` = `ai4privacy:<id>`) |
| `language` | str | `"English"` in both pilot files (100%) |
| `set` | str | `"train"` / `"validation"`, matches the file |

## Label census (full streaming count, train file)

GIVENNAME1 7895, GIVENNAME2 2102, LASTNAME1 9066, LASTNAME2 2275, LASTNAME3 734,
TITLE 7560, USERNAME 11115, EMAIL 9759, TIME 14676, DATE 6780, BOD 8461, CITY 7657,
STATE 7590, COUNTRY 5875, STREET 7359, BUILDING 7383, POSTCODE 7458, SECADDRESS 3117,
TEL 7312, IP 8145, IDCARD 9609, PASSPORT 9020, DRIVERLICENSE 8698, PASS 5637,
SOCIALNUMBER 9599, SEX 7496. Plus rare CARDISSUER (5) and GEOCOORD (703).

## PERSON mapping (policy in `docs/ANNOTATION_POLICY.md`)

`GIVENNAME1/2` + `LASTNAME1/2/3` → PERSON (adjacent given+family merged into one span).
`TITLE` (e.g. `Sel`), `USERNAME`, and all other labels are NOT PERSON.

## Split roles (locked 2026-10-04, M02.3; seed 20260920)

Carved by `scripts/carve_splits.py` at the pinned revision; manifests in `artifacts/splits/`
(gitignored, regenerable from seed + revision; final-eval sha256 recorded here).

| Role | Rows | Source |
|---|---|---|
| `final_eval` (locked) | 1,000 | seeded carve of train file, exact-text-unique, no text in validation file |
| `train_side` (detector training, generator SFT, attack development) | 28,713 | train file minus final_eval |
| `dev_calibration` | 7,923 | validation file, minus quarantined |

- **Final-eval lock:** `artifacts/splits/final_eval.jsonl` sha256
  `513335badf5752a340698d3c7efa7029a8780c51fde2675c80ef4208ce27d39c`, row ids+hashes in
  `final_eval_manifest.json`. No generator training, prompt authoring, sample selection, or
  reward tuning from these rows.
- **Row-level quarantine:** 195 train / 23 validation rows have ≥1 offset mismatch (the earlier
  census counted 909/113 mismatching masks; same criterion, different unit).
- **Duplicates:** 0 exact-text duplicates within or across the two files; every one of the
  28,713 train-side rows also has a distinct PII-masked template skeleton, so no
  near-duplicate/template leakage was found at pilot scale.
- **Composition:** 290/1,000 final-eval rows contain PERSON spans (28.5–29% person-row share
  holds across roles); the rest serve as negatives for the precision guardrail.

### Name pools (M02.4, `artifacts/splits/name_pools.json`)

- `train_side_pool`: 16,403 distinct PERSON values from train-side rows, with counts.
  This is the only pool Spark request banks and the sampler may draw names from.
- `heldout_names`: 200 seeded names occurring only in final-eval rows (593 such names exist;
  seeded pick), reserved for M04 paired-name tests. Exact-match-disjoint from the train pool;
  19/200 occur as substrings of train-side full names — recorded limitation, not a blocker.
- Name overlap between final-eval and train-side texts is not banned: detector-training texts
  keep their names; the pools govern generation requests, not the training texts.

## Quarantine rule

Any record with a `privacy_mask` offset mismatch (`text[start:end] != value`, observed
0.2–0.5% with template-fragment values) is excluded from SFT/detector pools. Counts reported.
