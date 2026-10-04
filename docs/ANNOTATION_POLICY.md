# ANNOTATION_POLICY — PERSON (AI4Privacy pilot)

## Source labels → PERSON

- **PERSON:** `GIVENNAME1`, `GIVENNAME2`, `LASTNAME1`, `LASTNAME2`, `LASTNAME3`.
- **Not PERSON:** `TITLE` (honorifics such as `Sel` — kept as context, never a span),
  `USERNAME` (handles such as `luka.burg` are identifiers, not person references),
  and every other label. Do not reinterpret companies, usernames, or IDs as names.

## Span formation

- Offsets are Python str character indices, end-exclusive; every span must satisfy
  `text[start:end] == value` (verified in code; mismatches quarantine the record).
- Adjacent given + family names separated only by whitespace merge into ONE PERSON span
  (e.g. `Ana Ruiz` → `[15, 23)`), since the mention refers to one person.
- Non-adjacent or repeated name parts stay separate spans; records with more than one
  PERSON span are valid supervision but excluded from the single-name-once SFT pilot.
- Possessives and punctuation stay outside spans (`Ana Ruiz's` → span covers `Ana Ruiz`).
- Matching is case-sensitive exact; Unicode preserved (accented names in tests).

## Limits of this policy

Labels are synthetic-authoritative, not human-adjudicated: a labeled string is treated
as a person mention under this policy without claiming anything about a real person.
Ambiguous common words keep their label; semantic doubt is handled by audit sampling
(random + high-reward), not by dictionary filtering unfamiliar names.
