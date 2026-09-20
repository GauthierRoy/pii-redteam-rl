# Decisions (provisional — owner approval required for each)

All D01–D11 from `PROJECT_PLAN.md` section 3 remain **open**. Nothing below is approval
to spend, download restricted data, or contact external services. Recommendations are
the agent's; the owner decides.

| ID | Decision | Recommendation at M01 | Status |
|---|---|---|---|
| D01 | Hardware and runtime | Colab T4 for tiny real-model pilots; verify GPU/VRAM/persistence (incl. S00 FP8 matrix) before any training | Open |
| D02 | Language and domain | English short support-style messages (plan default) pending owner confirmation | Open |
| D03 | Data policy and sources | Synthetic + explicitly permitted research sets only; no work/customer records ever | Open |
| D04 | Budget | Free-first; ~€50 envelope is not authorization — each paid run needs a measured estimate + cap + shutdown plan | Open |
| D05 | Model choices | Small instruction generator + ModernBERT-compatible detector; pin after S00/hardware check | Open |
| D06 | Name-group metadata | Curated source with documented uncertainty; `unknown` over forced labels | Open |
| D07 | Evaluation criteria | Span recall + precision guardrail; fix floor, effect threshold, seeds before final tests | Open |
| D08 | Human review | Small structured audits; needs reviewer + rubric agreement | Open |
| D09 | Generator inference precision | FP8 serving baseline + BF16/FP16 reference (S01); AWQ optional | Open |
| D10 | Speculative decoding | Supported lightweight baseline first (S03), DFlash after (S04) | Open |
| D11 | DFlash draft training | Frozen target, separate cap, break-even analysis; defer if it risks M09–M13 budget | Open |

## Settled implementation choices (M01, reversible)

- R01: stdlib + pyyaml runtime at M01; heavy ML deps deferred to their pilot milestones.
- R02: `grpo_compat` reused from `../rl` with lazy `trl` import (see `docs/REUSE_AUDIT.md`).
- R03: Offsets are Python str indices, end-exclusive; recorded in `validation.py`.
- R04: Invalid reward total (-1.0) sits below the valid [0, 1] range; components logged separately.
