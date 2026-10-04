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
