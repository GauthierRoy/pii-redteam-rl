"""GRPO reward adapter (M08.2): frozen-detector name probability as the RL reward.

Same scorer path as the M07a probe and the validity-v2 rules the M07b prompt comparison
passed under (strict JSON contract, one word-boundary mention of the requested name, minimum
length and content, no refusal), so a degenerate text such as "Hi Lucas" cannot earn the
reward by being short. Reward = 1 - mean detector probability over the name's tokens. Invalid completions
get INVALID_REWARD, strictly below the valid range [0, 1].
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from ..probe_metrics import reward_variants
from ..validity_v2 import classify_sample_v2

INVALID_REWARD = -1.0
SCORER_TRUNCATED = "invalid:scorer_truncated"


class Scorer(Protocol):
    """Fills `p_target_mean` and `outcome` on every sample whose status is valid."""

    def score(self, samples: list[dict]) -> None: ...


def completion_text(completion: str | list[dict]) -> str:
    """TRL passes plain strings or [{"role": ..., "content": ...}] message lists."""
    if isinstance(completion, str):
        return completion
    return completion[-1]["content"]


class TargetDetectorReward:
    """TRL reward function: `reward = TargetDetectorReward(scorer, eos_token_id)`, pass to GRPOTrainer."""

    __name__ = "target_detector_reward"

    def __init__(self, scorer: Scorer, eos_token_id: int) -> None:
        self.scorer = scorer
        self.eos_token_id = eos_token_id

    def __call__(
        self,
        completions: list,
        name: list[str],
        completion_ids: list[list[int]] | None = None,
        log_extra=None,
        log_metric=None,
        **kwargs,
    ) -> list[float]:
        ids_per_completion = completion_ids or [[self.eos_token_id]] * len(completions)
        truncated = [self.eos_token_id not in ids for ids in ids_per_completion]
        samples = [
            classify_sample_v2(completion_text(c), n, t)
            for c, n, t in zip(completions, name, truncated, strict=True)
        ]
        self.scorer.score(samples)
        rewards = [self._reward(sample) for sample in samples]
        if log_extra:
            log_extra("status", [s["status"] for s in samples])
            log_extra("p_target_mean", [s.get("p_target_mean") for s in samples])
        if log_metric:
            for key, value in self.metrics(samples).items():
                log_metric(key, value)
        return rewards

    @staticmethod
    def _reward(sample: dict) -> float:
        if sample["status"] != "valid":
            return INVALID_REWARD
        if sample.get("p_target_mean") is None:
            sample["status"] = SCORER_TRUNCATED
            return INVALID_REWARD
        return reward_variants(sample["p_target_mean"])["r_prob"]

    @staticmethod
    def metrics(samples: list[dict]) -> dict[str, float]:
        """Per-batch rates over ALL completions (invalid ones count in the denominator)."""
        total = len(samples)
        statuses = Counter(s["status"] for s in samples)
        outcomes = Counter(s.get("outcome") for s in samples if s["status"] == "valid")
        return {
            "valid_rate": statuses["valid"] / total,
            "truncated_rate": statuses["invalid:truncated"] / total,
            "true_miss_rate": outcomes["complete_miss"] / total,
        }
