"""GRPO reward adapter (M08.2): frozen-detector name probability as the RL reward.

Same scorer path as the M07a probe and the validity-v2 rules the M07b prompt comparison
passed under (strict JSON contract, one word-boundary mention of the requested name, minimum
length and content, no refusal), so a degenerate text such as "Hi Lucas" cannot earn the
reward by being short. Reward = 1 - mean detector probability that the name's tokens are PERSON (1 - P(O), B and I
summed; D25, docs/LIMITATIONS.md L1). Invalid completions
get INVALID_REWARD, strictly below the valid range [0, 1].

Optional log-odds scale (D23): reward = clip(1 + log((1 - p) / p) / span, 0, 1), so a text at
p = 0.9998 earns 0.15 and one at p = 0.99 earns 0.54, where 1 - p gives 0.0002 and 0.01. The
plain reward cannot tell these apart, which left GRPO with nothing to climb against the
retrained detector (m10-attack-retrained-20261115-203246: median p 0.9998 for 400 steps).
p <= 0.5 (a miss) earns the full reward of 1.

Optional collapse guard (D20): the detector reward of a valid text is scaled down, to zero
at worst, when its opening word is shared by too many texts written for other requests.

Optional person judge (D21): a text that passes the validity rules but uses the requested
name as something other than a person is invalid (NOT_PERSON_STATUS), so a detector that
rightly ignores a place or brand name is not counted as fooled.

Optional `record`: called once per batch with one row per completion (text, status, reward,
scores), so a training run can keep the texts it produced.

Optional audit (D21): three more judge answers per text (another person's name, natural,
fits the requested format), fetched while the person judge runs. Logged only.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from ..probe_metrics import reward_variants
from ..validity_v2 import classify_sample_v2
from .opening_share import OpeningShare, opening_word
from .person_judge import AUDIT_FIELDS, PERSON, UNPARSED, PersonJudge, TextAudit

INVALID_REWARD = -1.0
SCORER_TRUNCATED = "invalid:scorer_truncated"
NOT_PERSON_STATUS = "invalid:name_not_person"
JUDGE_UNPARSED_STATUS = "invalid:judge_unparsed"
OTHER_PERSON_STATUS = "invalid:other_person"


def probability_reward(p: float) -> float:
    """Default detector reward: 1 - p."""
    return reward_variants(p)["r_prob"]


class LogOddsReward:
    """clip(1 + log((1 - p) / p) / span, 0, 1): 0 at p >= 1/(1 + e^-span), 1 at p <= 0.5."""

    def __init__(self, span: float = 10.0) -> None:
        if span <= 0:
            raise ValueError("span must be positive")
        self.span = span

    def __call__(self, p: float) -> float:
        p = min(max(p, 1e-9), 1 - 1e-9)
        return min(1.0, max(0.0, 1.0 + math.log((1 - p) / p) / self.span))


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
        self,
        scorer: Scorer,
        eos_token_id: int,
        opening_share: OpeningShare | None = None,
        person_judge: PersonJudge | None = None,
        audit: TextAudit | None = None,
        record: Callable[[list[dict]], None] | None = None,
        detector_reward: Callable[[float], float] = probability_reward,
        reject_other_person: bool = False,
    ) -> None:
        self.scorer = scorer
        self.eos_token_id = eos_token_id
        self.opening_share = opening_share
        self.person_judge = person_judge
        self.audit = audit
        self.record = record
        self.detector_reward = detector_reward
        self.reject_other_person = reject_other_person

    def __call__(
        self,
        completions: list,
        name: list[str],
        completion_ids: list[list[int]] | None = None,
        request_id: list[str] | None = None,
        context: list[str] | None = None,
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
        self._judge_texts(samples, name, context)
        if self.reject_other_person:
            # The augmentation sets reject such texts, so the attacker must not earn on them
            # (94% of the D24 exploit texts named another person). A failed audit (None) passes.
            for sample in samples:
                if sample["status"] == "valid" and (sample.get("audit") or {}).get("other_person"):
                    sample["status"] = OTHER_PERSON_STATUS
        self.scorer.score(samples)
        self._flag_repeated_openings(samples, name, request_id or name)
        rewards = [self._reward(sample) for sample in samples]
        if log_extra:
            log_extra("status", [s["status"] for s in samples])
            log_extra("p_target_mean", [s.get("p_target_mean") for s in samples])
            log_extra("opening_share", [s.get("opening_share") for s in samples])
            for field in AUDIT_FIELDS:
                log_extra(field, [(s.get("audit") or {}).get(field) for s in samples])
        if log_metric:
            for key, value in self.metrics(samples).items():
                log_metric(key, value)
        if self.record:
            self.record(
                [
                    {
                        "request_id": request,
                        "name": requested_name,
                        "raw_text": completion_text(completion),
                        "status": sample["status"],
                        "reward": reward,
                        "p_target_mean": sample.get("p_target_mean"),
                        "opening_share": sample.get("opening_share"),
                        **(sample.get("audit") or {}),
                    }
                    for request, requested_name, completion, sample, reward in zip(
                        request_id or name, name, completions, samples, rewards, strict=True
                    )
                ]
            )
        return rewards

    def _judge_texts(
        self, samples: list[dict], names: list[str], contexts: list[str] | None
    ) -> None:
        """Person judge (gate) and audit (log only) on the valid samples, at the same time.

        A sample whose name is not used as a person becomes invalid before scoring.
        """
        valid = [i for i, sample in enumerate(samples) if sample["status"] == "valid"]
        valid_names = [names[i] for i in valid]
        texts = [samples[i]["text"] for i in valid]
        with ThreadPoolExecutor(max_workers=1) as background:
            audits = None
            if self.audit is not None and contexts is not None:
                audits = background.submit(
                    self.audit.audits, valid_names, texts, [contexts[i] for i in valid]
                )
            if self.person_judge is not None:
                for i, verdict in zip(
                    valid, self.person_judge.verdicts(valid_names, texts), strict=True
                ):
                    if verdict != PERSON:
                        samples[i]["status"] = (
                            JUDGE_UNPARSED_STATUS if verdict == UNPARSED else NOT_PERSON_STATUS
                        )
            if audits is not None:
                for i, audit in zip(valid, audits.result(), strict=True):
                    samples[i]["audit"] = audit

    def _flag_repeated_openings(
        self, samples: list[dict], names: list[str], requests: list[str]
    ) -> None:
        """Set `opening_share` and `opening_factor` on valid samples when the guard is on."""
        if self.opening_share is None:
            return
        valid = [i for i, sample in enumerate(samples) if sample["status"] == "valid"]
        shares = self.opening_share.shares(
            [requests[i] for i in valid],
            [opening_word(samples[i]["text"], names[i]) for i in valid],
        )
        for i, share in zip(valid, shares, strict=True):
            samples[i]["opening_share"] = share
            samples[i]["opening_factor"] = self.opening_share.reward_factor(share)

    def _reward(self, sample: dict) -> float:
        if sample["status"] != "valid":
            return INVALID_REWARD
        if sample.get("p_target_mean") is None:
            sample["status"] = SCORER_TRUNCATED
            return INVALID_REWARD
        return self.detector_reward(sample["p_target_mean"]) * sample.get("opening_factor", 1.0)

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
            "not_person_rate": statuses[NOT_PERSON_STATUS] / total,
            "other_person_rejected_rate": statuses[OTHER_PERSON_STATUS] / total,
        }
        audited = [s["audit"] for s in samples if s["status"] == "valid" and s.get("audit")]
        if audited:
            metrics["audit_other_person_rate"] = sum(a["other_person"] for a in audited) / len(
                audited
            )
            metrics["audit_unnatural_rate"] = sum(not a["natural"] for a in audited) / len(audited)
            metrics["audit_off_format_rate"] = sum(not a["fits_format"] for a in audited) / len(
                audited
            )
        shares = [s["opening_share"] for s in samples if s.get("opening_share") is not None]
        if shares:
            metrics["opening_share_mean"] = sum(shares) / len(shares)
            metrics["repeated_opening_rate"] = (
                sum(1 for s in samples if s.get("opening_factor", 1.0) < 1.0) / total
            )
        return metrics
