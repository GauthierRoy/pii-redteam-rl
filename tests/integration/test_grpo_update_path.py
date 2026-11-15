"""M08.2: TRL GRPOTrainer update path on a tiny random model (CPU, seconds per test).

Opt-in: needs torch, transformers, peft, trl and the cached tiny models; skipped otherwise.
Run: uv run --no-project --with torch --with transformers --with trl --with peft
  --with accelerate --with datasets --with pyyaml --default-index
  https://download.pytorch.org/whl/cpu --index https://pypi.org/simple
  python -m unittest tests.integration.test_grpo_update_path
Checks the properties the plan lists for M08.2: completion masks, prompt/group layout,
reference vs behavior log-probs, trainable vs frozen parameters (attacker base, reference,
detector), zero-variance and all-invalid groups, and a controlled reward-learning run.
The tiny random Llama never emits EOS on its own, so its output layer gets an EOS bias
(completions then end at varied lengths); the learning test also scales the output layer
so the attacker actually responds to small hidden-state changes.
"""

import importlib.util
import os
import tempfile
import unittest
from unittest import mock

HAS_STACK = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "peft", "trl"))
TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_DETECTOR = "hf-internal-testing/tiny-random-ModernBertForTokenClassification"
GROUP = 4
NAMES = ["Lucas", "Amara"]
EOS_BIAS = 9.0
LEARNING_NAMES = [f"Name{i}" for i in range(8)]
LEARNING_GROUP = 8
LEARNING_HEAD_SCALE = 20.0
LEARNING_EOS_BIAS = 4.0
BIO_LABELS = {0: "O", 1: "B-PERSON", 2: "I-PERSON"}

if HAS_STACK:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForTokenClassification,
        AutoTokenizer,
        set_seed,
    )
    from trl import GRPOConfig, GRPOTrainer

    from pii_redteam.training.detector_scorer import DetectorScorer
    from pii_redteam.training.rl_reward import INVALID_REWARD, TargetDetectorReward
    from pii_redteam.training.rl_smoke import (
        TimedGRPOTrainer,
        WallClockStop,
        align_vllm_names_with_multimodal_checkpoint,
        opening_share_guard,
        person_judge,
        planned_steps,
        restrict_vllm_to_language_model,
        text_audit,
        timing_summary,
        to_multimodal_qwen_name,
        upcast_trainable_params_to_fp32,
    )

    class RecordingTrainer(GRPOTrainer):
        """Keeps every generation batch the trainer builds, before it is split for training."""

        recorded: list[dict] = []

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            self.recorded.append({k: v.clone() for k, v in output.items() if torch.is_tensor(v)})
            return output

    def eos_prone_model(eos_id, eos_bias, head_scale):
        model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
        head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=True)
        head.weight.data.copy_(model.lm_head.weight.data * head_scale)
        head.bias.data.zero_()
        head.bias.data[eos_id] = eos_bias
        model.lm_head = head
        return model

    def build_trainer(
        reward_funcs,
        tmp_dir,
        tokenizer,
        names=NAMES,
        group=GROUP,
        eos_bias=EOS_BIAS,
        head_scale=1.0,
        trainer_cls=RecordingTrainer,
        **overrides,
    ):
        set_seed(0)
        args = {
            "output_dir": tmp_dir,
            "per_device_train_batch_size": group * len(names),
            "num_generations": group,
            "max_completion_length": 8,
            "max_steps": 1,
            "learning_rate": 1e-2,
            "beta": 0.0,
            "report_to": "none",
            "use_cpu": True,
            "save_strategy": "no",
            "logging_steps": 1,
            "temperature": 1.0,
            "seed": 0,
        }
        args.update(overrides)
        rows = [{"prompt": f"Write a message for {n}.", "name": n} for n in names]
        trainer = trainer_cls(
            model=eos_prone_model(tokenizer.eos_token_id, eos_bias, head_scale),
            reward_funcs=reward_funcs,
            args=GRPOConfig(**args),
            train_dataset=Dataset.from_list(rows),
            processing_class=tokenizer,
            peft_config=LoraConfig(r=8, target_modules="all-linear", task_type="CAUSAL_LM"),
        )
        trainer.recorded = []
        return trainer

    def snapshot(model, lora):
        return {
            n: p.detach().clone() for n, p in model.named_parameters() if ("lora_" in n) == lora
        }

    def max_change(before, model):
        current = dict(model.named_parameters())
        return max(float((current[n].detach() - v).abs().max()) for n, v in before.items())

    class RaiseIfValidSampleScored:
        def score(self, samples):
            if any(sample["status"] == "valid" for sample in samples):
                raise AssertionError("gibberish completions must never classify as valid")

    def constant_reward(completions, **kwargs):
        return [1.0] * len(completions)

    def length_reward(completion_ids, **kwargs):
        return [-float(len(ids)) for ids in completion_ids]

    class NameRecorder:
        """Length reward that records the `name` column order the trainer passes."""

        __name__ = "name_recorder"

        def __init__(self):
            self.seen_names = []

        def __call__(self, completion_ids, name, **kwargs):
            self.seen_names.append(list(name))
            return length_reward(completion_ids)

    class ScoringReward:
        """Length reward that also runs a fixed valid sample through the frozen detector."""

        __name__ = "scoring_reward"

        def __init__(self, scorer, sample):
            self.scorer = scorer
            self.sample = sample

        def __call__(self, completion_ids, **kwargs):
            self.scorer.score([dict(self.sample)])
            return length_reward(completion_ids)


@unittest.skipUnless(HAS_STACK, "needs torch, transformers, peft and trl")
class TestGrpoUpdatePath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
        except OSError as error:
            raise unittest.SkipTest(f"tiny model not available: {error}") from error
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def trainer(self, reward_funcs, **overrides):
        tmp_dir = self.enterContext(tempfile.TemporaryDirectory())
        return build_trainer(reward_funcs, tmp_dir, self.tokenizer, **overrides)

    def test_prompt_groups_are_contiguous_and_advantages_sum_to_zero_per_group(self):
        recorder = NameRecorder()
        trainer = self.trainer(recorder)
        trainer.train()
        self.assertEqual(recorder.seen_names[0], [n for n in NAMES for _ in range(GROUP)])
        batch = trainer.recorded[0]
        prompts = batch["prompt_ids"].view(len(NAMES), GROUP, -1)
        for group in prompts:
            self.assertTrue(bool((group == group[0]).all()))
        self.assertFalse(bool((prompts[0, 0] == prompts[1, 0]).all()))
        group_sums = batch["advantages"].view(len(NAMES), GROUP).sum(dim=1)
        self.assertTrue(torch.allclose(group_sums, torch.zeros(len(NAMES)), atol=1e-5))

    def test_completion_mask_stops_at_the_first_eos_and_pads_after_it(self):
        trainer = self.trainer(length_reward)
        trainer.train()
        batch = trainer.recorded[0]
        eos, pad = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        lengths = batch["completion_mask"].sum(dim=1)
        self.assertGreater(len(set(lengths.tolist())), 1)
        for ids, mask in zip(batch["completion_ids"], batch["completion_mask"], strict=True):
            kept = int(mask.sum())
            self.assertTrue(bool((mask[:kept] == 1).all()))
            self.assertTrue(bool((mask[kept:] == 0).all()))
            if kept < len(ids):
                self.assertEqual(int(ids[kept - 1]), eos)
                self.assertTrue(bool((ids[kept:] == pad).all()))
            self.assertNotIn(eos, ids[: kept - 1].tolist())

    def test_only_lora_parameters_move(self):
        trainer = self.trainer(length_reward)
        lora_before = snapshot(trainer.model, lora=True)
        base_before = snapshot(trainer.model, lora=False)
        trainer.train()
        self.assertGreater(max_change(lora_before, trainer.model), 0.0)
        self.assertEqual(max_change(base_before, trainer.model), 0.0)

    def test_constant_reward_group_is_a_noop_update(self):
        trainer = self.trainer(constant_reward)
        lora_before = snapshot(trainer.model, lora=True)
        trainer.train()
        log = trainer.state.log_history[0]
        self.assertEqual(log["frac_reward_zero_std"], 1.0)
        self.assertEqual(log["grad_norm"], 0.0)
        self.assertEqual(max_change(lora_before, trainer.model), 0.0)
        self.assertTrue(bool((trainer.recorded[0]["advantages"] == 0).all()))

    def test_all_invalid_completions_give_zero_variance_and_no_update(self):
        reward = TargetDetectorReward(RaiseIfValidSampleScored(), self.tokenizer.eos_token_id)
        trainer = self.trainer(reward)
        lora_before = snapshot(trainer.model, lora=True)
        trainer.train()
        log = trainer.state.log_history[0]
        self.assertEqual(log["reward"], INVALID_REWARD)
        self.assertEqual(log["frac_reward_zero_std"], 1.0)
        self.assertEqual(max_change(lora_before, trainer.model), 0.0)

    def test_reference_stays_the_initial_policy_while_the_policy_moves(self):
        trainer = self.trainer(length_reward, beta=0.5, max_steps=4)
        trainer.train()
        kl = [entry["kl"] for entry in trainer.state.log_history if "kl" in entry]
        self.assertEqual(len(kl), 4)
        self.assertLess(abs(kl[0]), 1e-6)
        self.assertGreater(kl[-1], 1e-6)
        self.assertIn("ref_per_token_logps", trainer.recorded[0])

    def test_no_reference_logprobs_are_computed_when_beta_is_zero(self):
        trainer = self.trainer(length_reward, beta=0.0)
        trainer.train()
        self.assertNotIn("ref_per_token_logps", trainer.recorded[0])

    def test_frozen_detector_is_untouched_by_scoring_during_training(self):
        detector = AutoModelForTokenClassification.from_pretrained(
            TINY_DETECTOR,
            num_labels=len(BIO_LABELS),
            id2label=BIO_LABELS,
            label2id={label: i for i, label in BIO_LABELS.items()},
            ignore_mismatched_sizes=True,
        )
        scorer = DetectorScorer(
            detector, AutoTokenizer.from_pretrained(TINY_DETECTOR), max_length=64
        )
        self.assertFalse(detector.training)
        self.assertFalse(any(p.requires_grad for p in detector.parameters()))
        before = {n: p.detach().clone() for n, p in detector.named_parameters()}
        sample = {
            "status": "valid",
            "text": "Hi Lucas, please call me back.",
            "target_span": (3, 8),
        }
        self.trainer(ScoringReward(scorer, sample), max_steps=2).train()
        self.assertEqual(max_change(before, detector), 0.0)

    def test_generation_is_timed_once_per_update_and_within_the_step_time(self):
        trainer = self.trainer(length_reward, max_steps=3, trainer_cls=TimedGRPOTrainer)
        trainer.train()
        step_times = [e["step_time"] for e in trainer.state.log_history if "step_time" in e]
        self.assertEqual(len(step_times), 3)
        self.assertEqual(len(trainer.generation_seconds), 3)
        for step, generation in zip(step_times, trainer.generation_seconds, strict=True):
            self.assertGreater(generation, 0.0)
            self.assertGreaterEqual(step, generation)

    def test_controlled_reward_learning_shortens_completions(self):
        trainer = self.trainer(
            length_reward,
            names=LEARNING_NAMES,
            group=LEARNING_GROUP,
            head_scale=LEARNING_HEAD_SCALE,
            eos_bias=LEARNING_EOS_BIAS,
            max_steps=30,
        )
        trainer.train()
        rewards = [e["reward"] for e in trainer.state.log_history if "reward" in e]
        first, last = sum(rewards[:5]) / 5, sum(rewards[-5:]) / 5
        self.assertGreater(last, first + 2.0)


@unittest.skipUnless(HAS_STACK, "needs torch, transformers, peft and trl")
class TestRestrictVllmToLanguageModel(unittest.TestCase):
    def test_trl_vllm_engine_is_built_with_language_model_only(self):
        from trl.generation import vllm_generation

        engine = mock.Mock()
        with mock.patch.object(vllm_generation, "LLM", engine, create=True):
            restrict_vllm_to_language_model()
            vllm_generation.LLM(model="m", seed=1)
        engine.assert_called_once_with(model="m", seed=1, language_model_only=True)


@unittest.skipUnless(HAS_STACK, "needs torch, transformers, peft and trl")
class TestMultimodalQwenNames(unittest.TestCase):
    def test_text_only_names_get_the_language_model_prefix(self):
        self.assertEqual(
            to_multimodal_qwen_name("model.layers.0.mlp.down_proj.weight"),
            "model.language_model.layers.0.mlp.down_proj.weight",
        )
        self.assertEqual(
            to_multimodal_qwen_name("model.embed_tokens.weight"),
            "model.language_model.embed_tokens.weight",
        )

    def test_head_and_already_multimodal_names_are_left_alone(self):
        for name in (
            "lm_head.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.visual.blocks.0.attn.qkv.weight",
        ):
            self.assertEqual(to_multimodal_qwen_name(name), name)

    def test_trl_name_fixer_applies_both_the_trl_cleanup_and_the_rename(self):
        from trl.generation import vllm_generation

        with mock.patch.object(vllm_generation.VLLMGeneration, "_fix_param_name_to_vllm"):
            align_vllm_names_with_multimodal_checkpoint()
            fixed = vllm_generation.VLLMGeneration._fix_param_name_to_vllm(
                object(), "model._checkpoint_wrapped_module.layers.0.norm.weight"
            )
        self.assertEqual(fixed, "model.language_model.layers.0.norm.weight")


@unittest.skipUnless(HAS_STACK, "needs torch, transformers, peft and trl")
class TestTimingSummary(unittest.TestCase):
    def test_median_and_rounded_per_step_values(self):
        summary = timing_summary([1.0, 3.0, 2.0], [0.5, 0.25, 0.123])
        self.assertEqual(summary["step"], {"median": 2.0, "per_step": [1.0, 3.0, 2.0]})
        self.assertEqual(summary["generation"], {"median": 0.25, "per_step": [0.5, 0.25, 0.12]})

    def test_no_steps_gives_no_median(self):
        self.assertIsNone(timing_summary([], [])["step"]["median"])


@unittest.skipUnless(HAS_STACK, "needs torch, transformers, peft and trl")
class TestUpcastTrainableParams(unittest.TestCase):
    def test_trainable_adapters_become_fp32_and_frozen_weights_keep_their_dtype(self):
        try:
            base = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float16)
        except OSError as error:
            raise unittest.SkipTest(f"tiny model not available: {error}") from error
        model = get_peft_model(base, LoraConfig(r=4, target_modules="all-linear"))
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.bfloat16)
        changed = upcast_trainable_params_to_fp32(model)
        self.assertGreater(changed, 0)
        for name, param in model.named_parameters():
            expected = torch.float32 if param.requires_grad else torch.float16
            self.assertEqual(param.dtype, expected, name)


@unittest.skipUnless(HAS_STACK, "needs torch, transformers, peft and trl")
class TestStopAndGuardConfig(unittest.TestCase):
    def control_after(self, stopper, step):
        from transformers import TrainerControl, TrainerState

        return stopper.on_step_end(None, TrainerState(global_step=step), TrainerControl())

    def test_stop_at_step_ends_training_without_a_wall_clock_stop(self):
        stopper = WallClockStop(max_minutes=60, stop_at_step=3)
        self.assertFalse(self.control_after(stopper, 2).should_training_stop)
        control = self.control_after(stopper, 3)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(control.should_save)
        self.assertFalse(stopper.triggered)

    def test_wall_clock_stop_is_reported_as_triggered(self):
        stopper = WallClockStop(max_minutes=-1)
        self.assertTrue(self.control_after(stopper, 1).should_training_stop)
        self.assertTrue(stopper.triggered)

    def test_planned_steps_prefers_stop_at_step(self):
        self.assertEqual(planned_steps({"grpo": {"max_steps": 1000}}), 1000)
        self.assertEqual(planned_steps({"stop_at_step": 200, "grpo": {"max_steps": 1000}}), 200)

    def test_guard_is_off_without_reward_settings(self):
        self.assertIsNone(opening_share_guard({}))
        self.assertIsNone(opening_share_guard({"reward": None}))

    def test_person_judge_is_off_without_settings_and_needs_its_key(self):
        self.assertIsNone(person_judge({}))
        settings = {
            "model": "zai-glm-5-3-flash",
            "api_key_env": "PII_REDTEAM_TEST_MISSING_KEY",
            "timeout_s": 60,
            "max_workers": 32,
        }
        with self.assertRaises(SystemExit):
            person_judge({"reward": {"person_judge": settings}})
        with mock.patch.dict(os.environ, {"PII_REDTEAM_TEST_MISSING_KEY": "k"}):
            judge = person_judge({"reward": {"person_judge": settings}})
        self.assertEqual((judge.model, judge.max_workers), ("zai-glm-5-3-flash", 32))

    def test_text_audit_is_off_without_settings_and_needs_its_key(self):
        self.assertIsNone(text_audit({"reward": {"person_judge": {}}}))
        settings = {
            "model": "mistral-medium-2604",
            "api_key_env": "PII_REDTEAM_TEST_MISSING_KEY",
            "timeout_s": 30,
            "max_workers": 16,
        }
        with self.assertRaises(SystemExit):
            text_audit({"reward": {"audit": settings}})
        with mock.patch.dict(os.environ, {"PII_REDTEAM_TEST_MISSING_KEY": "k"}):
            audit = text_audit({"reward": {"audit": settings}})
        self.assertEqual((audit.model, audit.max_workers), ("mistral-medium-2604", 16))

    def test_guard_reads_settings_and_window_override(self):
        settings = {
            "full_reward_until": 0.25,
            "zero_reward_from": 0.5,
            "window": 256,
            "min_other_texts": 128,
        }
        config = {"reward": {"opening_share": settings}}
        guard = opening_share_guard(config)
        self.assertEqual((guard.full_reward_until, guard.zero_reward_from), (0.25, 0.5))
        self.assertEqual((guard.recent.maxlen, guard.min_other_texts), (256, 128))
        self.assertEqual(opening_share_guard(config, window=64).recent.maxlen, 64)


if __name__ == "__main__":
    unittest.main()
