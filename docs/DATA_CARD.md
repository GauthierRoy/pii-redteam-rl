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

## Split roles (provisional, M04 locks final evaluation)

- Train file → detector training/dev, generator SFT pairs, attack development.
- Validation file → development/calibration only. It is NOT the final test set.
- Final-evaluation role unassigned: carve a locked held-out partition before M10; no
  generator training or prompt authoring from it.

## Quarantine rule

Any record with a `privacy_mask` offset mismatch (`text[start:end] != value`, observed
0.2–0.5% with template-fragment values) is excluded from SFT/detector pools. Counts reported.
