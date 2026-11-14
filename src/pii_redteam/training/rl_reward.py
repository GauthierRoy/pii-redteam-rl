"""GRPO reward adapter (M08.2): frozen-detector name probability as the RL reward.

Same scorer path as the M07a probe and the validity-v2 rules the M07b prompt comparison
passed under (strict JSON contract, one word-boundary mention of the requested name, minimum
length and content, no refusal), so a degenerate text such as "Hi Lucas" cannot earn the
reward by being short. Reward = 1 - mean detector probability over the name's tokens. Invalid completions
get INVALID_REWARD, strictly below the valid range [0, 1].

Optional collapse guard (D20): a valid text whose opening word is shared by too many texts
written for other requests earns REPEATED_OPENING_REWARD instead of its detector reward.
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol

from ..probe_metrics import reward_variants
from ..validity_v2 import classify_sample_v2
from .opening_share import OpeningShare, opening_word

INVALID_REWARD = -1.0
REPEATED_OPENING_REWARD = 0.0
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

    def __init__(
        self, scorer: Scorer, eos_token_id: int, opening_share: OpeningShare | None = None
    ) -> None:
        self.scorer = scorer
        self.eos_token_id = eos_token_id
        self.opening_share = opening_share

    def __call__(
        self,
        completions: list,
        name: list[str],
        completion_ids: list[list[int]] | None = None,
        request_id: list[str] | None = None,
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
        self._flag_repeated_openings(samples, name, request_id or name)
        rewards = [self._reward(sample) for sample in samples]
        if log_extra:
            log_extra("status", [s["status"] for s in samples])
            log_extra("p_target_mean", [s.get("p_target_mean") for s in samples])
            log_extra("opening_share", [s.get("opening_share") for s in samples])
        if log_metric:
            for key, value in self.metrics(samples).items():
                log_metric(key, value)
        return rewards

    def _flag_repeated_openings(
        self, samples: list[dict], names: list[str], requests: list[str]
    ) -> None:
        """Set `opening_share` and `repeated_opening` on valid samples when the guard is on."""
        if self.opening_share is None:
            return
        valid = [i for i, sample in enumerate(samples) if sample["status"] == "valid"]
        shares = self.opening_share.shares(
            [requests[i] for i in valid],
            [opening_word(samples[i]["text"], names[i]) for i in valid],
        )
        for i, share in zip(valid, shares, strict=True):
            samples[i]["opening_share"] = share
            samples[i]["repeated_opening"] = self.opening_share.is_repeated(share)

    @staticmethod
    def _reward(sample: dict) -> float:
        if sample["status"] != "valid":
            return INVALID_REWARD
        if sample.get("p_target_mean") is None:
            sample["status"] = SCORER_TRUNCATED
            return INVALID_REWARD
        if sample.get("repeated_opening"):
            return REPEATED_OPENING_REWARD
        return reward_variants(sample["p_target_mean"])["r_prob"]

    @staticmethod
    def metrics(samples: list[dict]) -> dict[str, float]:
        """Per-batch rates over ALL completions (invalid ones count in the denominator)."""
        total = len(samples)
        statuses = Counter(s["status"] for s in samples)
        outcomes = Counter(s.get("outcome") for s in samples if s["status"] == "valid")
        metrics = {
            "valid_rate": statuses["valid"] / total,
            "truncated_rate": statuses["invalid:truncated"] / total,
            "true_miss_rate": outcomes["complete_miss"] / total,
        }
        shares = [s["opening_share"] for s in samples if s.get("opening_share") is not None]
        if shares:
            metrics["opening_share_mean"] = sum(shares) / len(shares)
            metrics["repeated_opening_rate"] = (
                sum(1 for s in samples if s.get("repeated_opening")) / total
            )
        return metrics
