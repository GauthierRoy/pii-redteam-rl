"""Run outputs reach disk during generation, so a crash cannot lose a whole run."""

import importlib
import importlib.util
import json
import os
import tempfile
import unittest

from pii_redteam.run_io import append_jsonl, write_jsonl


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class TestRunIO(unittest.TestCase):
    def test_append_accumulates_and_keeps_unicode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.jsonl")
            append_jsonl(path, [{"name": "Élodie"}])
            append_jsonl(path, [{"name": "Ana"}])
            self.assertEqual(read_jsonl(path), [{"name": "Élodie"}, {"name": "Ana"}])

    def test_failed_rewrite_keeps_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.jsonl")
            write_jsonl(path, [{"v": 1}])

            def failing_rows():
                yield {"v": 2}
                raise RuntimeError("crash mid-write")

            with self.assertRaises(RuntimeError):
                write_jsonl(path, failing_rows())
            self.assertEqual(read_jsonl(path), [{"v": 1}])


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch")
class TestGenerationSurvivesCrash(unittest.TestCase):
    """A model failure in the second batch must leave the first batch on disk."""

    def _fakes(self):
        import torch

        class Encoding(dict):
            def to(self, _device):
                return self

        class Tokenizer:
            eos_token_id = 2

            def __call__(self, texts, **_kwargs):
                return Encoding(input_ids=torch.ones((len(texts), 3), dtype=torch.long))

            def apply_chat_template(self, messages, **_kwargs):
                return messages[0]["content"]

            def decode(self, _seq, **_kwargs):
                return '{"text": "ok"}'

        class Model:
            calls = 0

            def generate(self, input_ids, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("simulated GPU failure")
                n = kwargs.get("num_return_sequences", 1)
                out = torch.full((input_ids.shape[0] * n, 5), 2, dtype=torch.long)
                return out

        return Tokenizer(), Model()

    def _bank(self):
        return [
            {
                "request_id": f"r{i}",
                "task": {
                    "entities": [{"type": "PERSON", "value": "Ana"}],
                    "context_type": "chat_message",
                    "target_length": {"min_words": 20, "max_words": 30},
                },
            }
            for i in range(4)
        ]

    def _assert_first_batch_saved(self, generate, **variant_kwargs):
        tok, model = self._fakes()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "partial.jsonl")
            with self.assertRaises(RuntimeError):
                generate(
                    tok,
                    model,
                    self._bank(),
                    temperature=1.0,
                    n=2,
                    top_p=0.95,
                    max_new_tokens=5,
                    batch_prompts=2,
                    log_prefix="test",
                    device="cpu",
                    sink_path=path,
                    **variant_kwargs,
                )
            saved = read_jsonl(path)
        self.assertEqual(len(saved), 4)  # 2 requests x n=2
        self.assertEqual({s["request_id"] for s in saved}, {"r0", "r1"})
        self.assertEqual(saved[0]["raw_text"], '{"text": "ok"}')

    def test_prompt_compare(self):
        module = importlib.import_module("pii_redteam.prompt_compare")
        self._assert_first_batch_saved(module._generate, variant="P1")

    def test_probe(self):
        module = importlib.import_module("pii_redteam.probe")
        self._assert_first_batch_saved(module._generate)


if __name__ == "__main__":
    unittest.main()
