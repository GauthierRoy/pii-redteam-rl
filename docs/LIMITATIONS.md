# Limitations

Known weaknesses of this project's design and results. Each entry says what was wrong, what
it affected, what was done about it, and what remains. The final report (M13) must carry these.

## L1. B/I labels on a single-entity task: a design error

**What was wrong.** Every detector in the project (base, target, retrained, and the M06
baseline detectors) was trained with BIO tags: `O`, `B-PERSON` (first token of a name) and
`I-PERSON` (any later token). BIO exists to separate two entities of the same type that touch
("Mary John Smith" as two people). This project has one entity type, and its use is redaction,
where a token is hidden whether it starts a name or continues one. The B/I split gives the
project nothing:

| AI4Privacy split (final eval excluded) | PERSON spans | pairs of touching PERSON spans | share |
|---|---|---|---|
| train side | 18,253 | 35 (in 20 texts) | 0.19% |
| dev | 5,031 | 15 (in 3 texts) | 0.30% |

Most of those few pairs are one person's first and last name split across a line break, or
one name per line in a list. In the generated messages there is one requested name per text.
A plain `PERSON` / `O` scheme was the right design from the start. BIO came in with the
standard token-classification setup in M03 and was never questioned, by the owner or by the
assistant that built it.

**What it broke.** The RL reward was scored on the probability of the exact BIO label of each
name token. In `m10-attack-logodds-20261117-212849` (D23) the attacker learned to put an
invented first name or title before the requested surname ("Dr. Sarah Hentzy" for "Hentzy",
234 of 245 held-out texts). The detector tagged the whole longer name, so the surname became
`I-PERSON` where the gold label is `B-PERSON`: the score fell to about 0.67 while the name
would still be redacted. One 400-step run ($1.23) was spent on this artefact. The earlier
runs used 1 - p, where the B/I gap paid almost nothing; their findings (the "Dear" letters)
are real misses.

**What was done (D25, 2026-11-18).** The detectors are not retrained. Every use of a detector
now reads it as PERSON / not-PERSON: the PERSON probability of a token is
`P(B-PERSON) + P(I-PERSON) = 1 - P(O)`, a token counts as PERSON when that is at least 0.5,
and consecutive PERSON tokens form one span whatever their B/I split
(`pii_redteam.detector.person_spans`). This applies to the RL reward, the probe scoring, and
the detector evaluation in `train_detector.py`. Training still uses BIO targets.

**What remains.**
- The detectors were *trained* with a three-way objective that the task does not need. Whether
  a `PERSON` / `O` detector would learn better is untested; the counts above suggest the
  difference is small. Planned check: one PERSON / O detector trained at the end (M12), compared
  with its BIO twin.
- Results recorded before 2026-11-18 used the BIO argmax decoding and the gold-label
  probability. Comparisons quoted after D25 are re-scored with the new reading; older numbers in
  `EXPERIMENTS.yaml` and `STATE.md` stay as recorded and are marked by date.
- Historical scripts (`scripts/audit_negative_fps.py`, `scripts/flag_extra_names_base_detector.py`,
  `scripts/modernbert_cpu_gate.py`) keep the BIO decoding so their recorded outputs reproduce.
