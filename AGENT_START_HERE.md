# Agent Handoff — Continue After M01

**Plan version:** 1.3 — 2026-09-20  
**Owner:** Gauthier Roy  
**Implementation agent:** Muse Spark 1.3 xhigh  
**Main document:** PROJECT_PLAN.md

## Current state: M01 is already completed

The owner reports that M01 is done. Continue from the existing repository and implementation. This handoff does not independently verify code or claim that later experiments ran.

Read PROJECT_PLAN.md, STATE.md, docs/DECISIONS.md, existing configs, tests, and run manifests. Reconcile the changes below without resetting completed work. Resume M02 unless the actual state shows it has already advanced. Resolve only genuinely missing prerequisites from M00.

Do not re-scaffold the repository, create another project, rename packages, replace the dependency manager, or repeat M01 to match a suggested architecture. Fix a concrete gap only when needed for the next task. Preserve prior experiments, decisions, and evidence.

## Updated project decisions

**Dataset:** Start WITH ai4privacy/pii-masking-300k. Use it for baseline detector training and construct generator SFT pairs from its training portion. Generated data augments this foundation rather than replacing it. The owner personally knows the AI4Privacy team and reports their approval. Record that once and proceed; do not reopen licensing as a blocker or switch to Gretel.

**Scope:** One language, PERSON names, short texts, one frozen attack detector, one retraining round. Keep an already recorded language/domain choice; otherwise English is the pilot default. Audit actual dataset rows and label mapping rather than trusting advertised counts. Additional languages and PII types can wait.

**Models:** Working ladder: Qwen/Qwen3.5-0.8B for real-model smoke tests, Qwen/Qwen3.5-2B for the first substantive SFT/RL run, and Qwen/Qwen3.5-4B only if quality/resources justify escalation. Use non-thinking output and verify a real training update in the chosen hybrid-model stack. Do not run every model size by default or silently replace an already working configuration.

**SFT task:** Entity types plus exact values → text containing those entities. Derive the input entities from existing AI4Privacy annotations; use the corresponding original text/snippet as the assistant target. Initially choose one clear person name occurring once. Do not create contradictory instruction/target pairs. Compute output spans in code; do not ask the LLM to calculate character offsets.

## Keep BOTH ways of producing generation inputs

### Option A: Spark directly authors requests

The owner may ask Spark directly to produce a bank of generation inputs from a natural-language brief. Spark may create the entity values, context/style choices, structured requests, and complete prompts. It is NOT limited to writing sampler code or proposing a context catalogue.

The owner reports free/unlimited Spark tokens in the current workflow. Use this as an available authoring option without adding an unnecessary paid teacher-model dependency. Do not infer that a separate API or autonomous service integration exists. Spark can simply export JSONL requests and the pipeline can import them.

Validate and save the exact request bank, original brief, provenance, model/settings when known, and hash. Reproducibility comes from replaying the saved bank, not assuming a second Spark call will produce the same inputs. Avoid final-test names/examples/contexts according to the experiment's split rules.

### Option B: seeded programmatic sampler

Keep a lightweight seeded sampler as the second option. It combines training-side name pools, optional contexts, and constraints into the SAME request schema. Record the seed and pool versions. Do not force a fully developed sampler before Spark-authored inputs can be imported.

### Shared rules

Both sources feed one request loader, validator, prompt/output contract, generator, and reward pipeline. Keep request_source in provenance. Save the actual rendered prompt. If a Spark-written prompt contradicts its structured entities/constraints, reject or repair the record explicitly.

Select the request source per run later; the owner has not chosen a permanent winner. Use the same frozen request bank or a matched distribution across ordinary generation, generate-and-filter, and RL. Spark-authored better prompts must not become an unacknowledged advantage for only one method.

The runtime division is: request bank → Qwen-generated text → code validation and span recovery → frozen detector score → RL update where applicable. Spark is not automatically a per-example reward judge. User-reported free Spark tokens do not make GPU training/rollouts free.

## Preserve the experiment and compute priorities

Keep generate-and-filter and original-data-only continued training as controls. Distinguish matched accepted augmentation size from matched total compute. Keep final evaluation data out of training, prompt authoring, sample selection, and reward tuning.

Use Colab first, Modal when verified credits/hardware fit, and Vast for later paid jobs. The owner may consider about €50 depending on need; this is not spending authorization. Before a paid run, give a measured feasibility check, current-rate cost estimate, hard cap, checkpoint path, and shutdown plan.

Retain the acceleration roadmap: S00 compatibility, S01 FP8 with a higher-precision reference, S02 optional AWQ, S03 working speculative baseline, S04 genuine DFlash draft training, and S05 measured showcase. Native Qwen MTP may be a useful first speculative option if supported; it is not DFlash. Do not let unsupported FP8 or custom draft training block the first scientific pilot.

## What to do next

Continue M02 by inspecting a small AI4Privacy sample and mapping its real fields/labels into the existing canonical schema. Validate spans, Unicode handling, source-to-PERSON mapping, and split boundaries. If fields contain serialized JSON, parse safely. Do not reuse mBERT token labels with a different tokenizer.

Produce a small set of dataset-derived SFT input/target pairs and show that the target really satisfies the prompt. Add or adapt the shared generation-request schema and Spark-bank importer. Keep the sampler as a compatible second path rather than making it a prerequisite for the import path.

The next handoff should contain observed dataset fields/labels, converted examples, exact commands/tests run, artifact locations, genuine blockers, and the next three actions. No need to start large training, design a universal name validator, or build a new multi-provider platform.

## Evidence and continuity

For each completed task, state what was implemented, the exact command actually executed, the resulting artifact/test output, and what remains unverified. Distinguish mocked, smoke-tested, trained, and evaluated work. Do not claim an experiment from code that has not run.

Update STATE.md with milestone/task IDs, progress, next actions, decisions, blockers, artifacts, and resource usage. Preserve failed runs and negative results. The aim is a small trustworthy end-to-end experiment, not an unsupported success story.
