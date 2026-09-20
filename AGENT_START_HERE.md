# Coding-Agent Kickoff — Muse Spark 1.3 xhigh

You are the implementation agent for **RL-Guided Synthetic Data for Robust PII Detection**. Treat `PROJECT_PLAN.md` as the main long-term project plan. The project uses a small LLM with SFT followed by RL to generate hard, valid synthetic person-name examples, then measures whether those examples improve a ModernBERT-based detector.

You are implementing the project; you are not automatically the research generator, teacher model, or an authorized external data processor.

## Start here

Read `PROJECT_PLAN.md` in full before implementing research components. Inspect the existing repository, if any, and reconcile this plan with existing work without overwriting it. Then read `STATE.md` and `docs/DECISIONS.md` if they exist.

Begin with milestones **M00 and M01**. Establish scope, hardware, data permissions, and budget. While decisions remain open, build only low-cost scaffolding and synthetic CPU fixtures. Do not start training, download large models, spend paid compute, or send data to external services without the relevant approval.

The first deliverable is a small, testable repository foundation, not an impressive-looking but unvalidated RL run.

## Maintain continuity

At the end of every work session, update `STATE.md` with the current milestone, completed task IDs, next three actions, blockers, open decisions, tests actually executed, artifact locations, and compute usage. Add consequential choices to `docs/DECISIONS.md` with their rationale.

For each increment, report what changed, what evidence validates it, and what is still unverified. Do not mark a milestone complete merely because code exists. Preserve failures and distinguish mocks, smoke runs, pilots, and final experiments.

## Preserve the research question

The key comparison is **RL versus strong non-RL synthetic-data generation**, especially **SFT generate-and-filter**. Keep the original-data continued-training control. Compare matched accepted augmentation sizes, and separately account for full compute cost. SFT, rejected rollouts, filtering, and detector retraining are not free.

Keep the detector frozen during the MVP RL phase. Keep final evaluation data out of training, rewards, sample selection, and development. Start with one language, person names, exact supplied-name preservation, short contexts, and one detector-retraining round.

A missed detection does not mean privacy. Higher generator reward does not demonstrate a better detector. The downstream held-out comparison is the main result.

## Build the generator-acceleration showcase

Follow Section 6A of the main plan. The sequence is **S00 compatibility checks → S01 FP8 inference baseline → S02 optional AWQ → S03 supported speculative decoding → S04 DFlash draft-model training → S05 integrated benchmark**.

Begin S00 during M00–M01. Verify primary DFlash sources, licenses, exact implementation versions, target/tokenizer support, and the actual GPU/backend feature matrix. Do not invent current compatibility or train a generic draft model and label it DFlash. Keep a higher-precision reference for quality validation; FP8 is the preferred serving baseline, not a requirement for FP8 SFT or RL training.

Run initial serving tests on G_sft after M05 and repeat validation for the final G_rl export. Train the DFlash draft only against a stable, frozen target after speculative inference works. AWQ is optional. DFlash training has a separate budget and uses training-side traces, never final evaluation data.

Measure **valid and retained hard examples per second**, quality, memory, and full costs alongside token throughput. Speculation may not help short outputs or highly batched workloads. Report measured negative results honestly. Separate quantization-induced changes from speculative verification relative to the configured target.

Keep the initial acceleration work in offline generation. Do not put quantized or speculative serving into on-policy RL rollouts until behavior-policy probabilities and the trainer's sampling/correction requirements have been tested. Protect the budget for the downstream detector comparison.

## Compute constraints: free first, paid only after evidence

Start on Colab. Use Modal Starter only after verifying current credits, GPU availability, runtime, and persistence constraints. Consider Vast.ai for later longer runs. The owner may consider approximately €50 of paid compute depending on need; this is not approval to spend.

Keep one Python training/generation pipeline and thin provider launchers. Start with CPU fixtures and tiny real-model runs, test checkpoint/resume across sessions, and keep authoritative artifacts in approved durable storage. Do not rely on notebook session disks.

Before any paid run, provide a measured feasibility benchmark, runtime/cost estimate using current provider rates, hard cap, and shutdown procedure, then request approval. Prioritize the complete core comparison over larger models or DFlash training. One exploratory result with honest limitations is better than several expensive unfinished components.

## Handle uncertainty honestly

Recommend choices with trade-offs when hardware, datasets, licenses, or statistical requirements are unresolved. Do not invent environment capabilities, training results, benchmark scores, data permissions, or demographic identity labels.

Use name-group metadata only with documented provenance and limitations. Names may have demographic or linguistic associations; they do not establish a person's ethnicity or gender.

A negative RL result is acceptable. Do not weaken baselines or change final-test criteria to obtain a win.

## First handoff

Provide a concise progress summary linked to milestone IDs, the repository skeleton and tests, commands actually run with their outcomes, the open decision list with recommendations, and a clear next step. Continue in milestone order once prerequisites are satisfied, scheduling the S00–S05 track according to its explicit dependencies.
