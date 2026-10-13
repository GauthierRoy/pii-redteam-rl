# Decisions

Plan v1.3 (2026-09-20) settles several items by owner report. Owner-reported statements
are recorded once as project decisions below; they are not independent legal or
service verification by the agent.

| ID | Decision | Position (2026-09-20) | Status |
|---|---|---|---|
| D01 | Hardware and runtime | Colab first; Modal/Vast later subject to verified credits/hardware | Open (measure at first GPU job) |
| D02 | Language and domain | English pilot default; coherent populated AI4Privacy subset; no invented domain metadata | Settled default, owner-confirmed scope |
| D03 | Data policy and sources | Start WITH ai4privacy/pii-masking-300k (rev pinned in `docs/DATA_CARD.md`); owner personally knows the AI4Privacy team and reports approval — recorded once, not reopened; no Gretel switch; no work/customer records ever | Settled |
| D04 | Budget | Free-first; ~€50 envelope is not authorization — each paid run needs measured estimate + cap + shutdown plan | Open |
| D05 | Model choices | Qwen3.5 ladder 0.8B (real-model smoke) → 2B (first SFT/RL) → 4B escalation candidate; non-thinking; ModernBERT detector; pin tested checkpoints before training | Settled ladder, checkpoints TBD |
| D06 | Name-group metadata | Curated sources with documented uncertainty; `unknown` over forced labels; defer elaborate studies to post-pilot | Open |
| D07 | Evaluation criteria | Span recall + precision guardrail; fix floor, effect threshold, seeds before final tests | Open |
| D08 | Human review | Small structured audits with short validity rubric; no universal name verifier | Open |
| D09 | Generator inference precision | FP8 serving baseline + BF16/FP16 reference; AWQ optional; must not block first pilot | Open |
| D10 | Speculative decoding | Supported baseline (native MTP candidate, not DFlash) before DFlash | Open |
| D11 | DFlash draft training | Frozen target, separate cap, break-even analysis; defer if it risks M09–M13 | Open |
| D12 | Generation-request authoring | BOTH options stay open: (A) Spark authors JSONL request banks from owner briefs (owner reports free/unlimited Spark tokens in current workflow — access arrangement, not a public guarantee; replay saved banks, never assume regeneration); (B) seeded programmatic sampler to the same schema. Same frozen bank/matched distribution across arms; Spark is not the reward judge; token counts recorded when known | Settled (both open) |
| D14 | Generation prompt + temperatures (owner, 2026-10-13, after M07b run `m07b-20261012-224819`) | Adopt **P1** (v1 prompt + `context_type` + `target_length` + optional `name_position` hint, request schema v2) as the generation prompt. M06 / D1 data generation runs at **t=0.7**; the RL temperature is decided in M08 (consider t=1.0). Evidence: P1 passes the pre-committed M07b criteria at BOTH temperatures (t=0.7: validity 0.9437 D15, name-at-start 0.364, top-10 openings 0.173, median unique 8/8; t=1.0: 0.9187 / 0.252 / 0.083 / 7); P0 (baseline) fails on template collapse, P2 (few-shot) trades name fidelity for style | Settled |
| D15 | Validity-v2 refusal scope (owner, 2026-10-13) | Refusal patterns flag only refusal/meta-text at the ANSWER START or answers with no content besides the boilerplate (`validity_v2.is_refusal`); mid-text phrases inside realistic messages ("I can't wait…", "please provide your reference number" in a support email) are content, not refusals. Re-label of run `m07b-20261012-224819`: 6 as-run refusal labels flip to valid (all false positives), 1 greeting flips to invalid; P1 passes at both temperatures with wider margins | Settled |

## Settled implementation choices (M01, reversible)

- R01: stdlib + pyyaml runtime at M01; heavy ML deps deferred to their pilot milestones.
- R02: `grpo_compat` reused from `../rl` with lazy `trl` import (see `docs/REUSE_AUDIT.md`).
- R03: Offsets are Python str indices, end-exclusive; recorded in `validation.py`.
- R04: Invalid reward total (-1.0) sits below the valid [0, 1] range; components logged separately.
- R05: Split roles locked 2026-10-04 (M02.3): seeded final-eval carve (1,000 train-file rows,
  sha256 in `docs/DATA_CARD.md`); validation file stays development-only; exact-text
  duplicates banned, template reuse measured (found none at skeleton level).
- R06: Name pools (M02.4): generation requests draw only from `train_side_pool`
  (16,403 values); 200 held-out names are final-eval-unique and reserved for M04 paired tests;
  exact-match disjoint from the pool, substring overlap (19/200) recorded as a limitation.
- R07: Requestable-name filter (owner-approved 2026-10-04): pool values that are
  dataset artifacts (digits, `@`, pipe/slash fragments, single initials; 74/16,403 =
  0.45%) are excluded from generation request pools. Enforced in
  `load_person_name_pool` (`is_requestable_name`), so all request paths reject them;
  detector training texts are NOT filtered.
- R08: D1 training-data gates (owner, 2026-10-13, M07b.2 precondition): the
  extra-person check must NOT rely on D0 (D13: ~50% miss rate on short
  conversational names). Independent checker = spaCy `en_core_web_sm` NER OR a
  small LLM judge (Qwen3.5-2B, greedy yes/no), rejecting any row with an extra
  person mention; person-sense check (name = a person, not a place/object/entry)
  via the same LLM judge, validated by a manual audit of 50 rows. Rejection
  rates reported with the M07b results (`docs/PROMPT_V2_RESULTS.md`).
