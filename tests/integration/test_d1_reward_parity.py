"""RL reward parity (M08.2): `D1Reward` + `D1Scorer` reproduce the recorded probe scores.

The five recovered M07a pilot generations (real base-2B outputs) were scored by frozen D0
in the probe. For those that pass validity v2 the RL reward must equal `1 - p_target_mean`,
so it matches what the go/no-go table measured; the rest must get the invalid reward. Needs torch/transformers and the frozen
D0 checkpoint: set PII_D0_PATH to the d0-best directory (skipped otherwise).
"""

import hashlib
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "m07a_pilot_recovered.jsonl")
FROZEN_D0_SHA = "0314ebe36d7f4dc6d784f0e139ff5b9d32633aef1a6f88e764a610a8d8ec43cb"
RECORDED_P_PRECISION = 6e-4
EOS = 1

try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from pii_redteam.training.d1_scorer import D1Scorer
    from pii_redteam.training.rl_reward import D1Reward
    from pii_redteam.validity_v2 import classify_sample_v2

    HAS_ML = True
except ImportError:
    HAS_ML = False

D0_PATH = os.environ.get("PII_D0_PATH", "")
HAS_D0 = bool(D0_PATH) and os.path.exists(os.path.join(D0_PATH, "model.safetensors"))


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@unittest.skipUnless(HAS_ML and HAS_D0, "needs torch/transformers and PII_D0_PATH")
class TestD1RewardParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sha = sha256_of(os.path.join(D0_PATH, "model.safetensors"))
        assert sha == FROZEN_D0_SHA, f"frozen-D0 sha mismatch: {sha}"
        scorer = D1Scorer(
            AutoModelForTokenClassification.from_pretrained(D0_PATH),
            AutoTokenizer.from_pretrained(D0_PATH),
        )
        cls.reward = D1Reward(scorer, EOS)
        with open(FIXTURE, encoding="utf-8") as f:
            cls.rows = [json.loads(line) for line in f]

    def test_reward_is_one_minus_the_recorded_probe_probability_for_v2_valid_rows(self):
        rewards = self.reward(
            [row["raw_text"] for row in self.rows],
            [row["name"] for row in self.rows],
            completion_ids=[[EOS] for _ in self.rows],
        )
        for row, reward in zip(self.rows, rewards, strict=True):
            status = classify_sample_v2(row["raw_text"], row["name"], False)["status"]
            if status == "valid":
                self.assertAlmostEqual(
                    reward,
                    1.0 - row["p_target_mean"],
                    delta=RECORDED_P_PRECISION,
                    msg=row["raw_text"],
                )
            else:
                self.assertEqual(reward, -1.0, msg=f"{status}: {row['raw_text']}")

    def test_truncated_completion_of_a_scorable_text_is_still_invalid(self):
        rewards = self.reward(
            [self.rows[0]["raw_text"]], [self.rows[0]["name"]], completion_ids=[[5, 6]]
        )
        self.assertEqual(rewards, [-1.0])


if __name__ == "__main__":
    unittest.main()
