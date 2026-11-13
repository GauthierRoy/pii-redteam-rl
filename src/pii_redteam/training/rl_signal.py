"""Group-level view of the RL reward signal on sampled, scored generations (M08.3)."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from ..probe_metrics import parse_json_strict

MISS_REWARD = 0.5


def signal_summary(samples: list[dict], metrics: dict) -> dict:
    """Is there anything for GRPO to learn from? Needs `request_id`, `raw_text`, `reward`, `status`."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[sample["request_id"]].append(sample)
    group_std = [statistics.pstdev([s["reward"] for s in g]) for g in groups.values()]
    group_has_miss = [any(s["reward"] > MISS_REWARD for s in g) for g in groups.values()]
    group_unique = [len({s["raw_text"] for s in g}) / len(g) for g in groups.values()]
    valid = [s for s in samples if s["status"] == "valid"]
    words = [len(parse_json_strict(s["raw_text"]).split()) for s in valid]
    return {
        "samples": len(samples),
        "groups": len(groups),
        **metrics,
        "status_counts": dict(Counter(s["status"] for s in samples)),
        "groups_with_detector_miss": sum(group_has_miss) / len(groups),
        "mean_group_reward_std": statistics.mean(group_std),
        "mean_valid_reward": statistics.mean(s["reward"] for s in valid) if valid else None,
        "mean_unique_text_share_per_group": statistics.mean(group_unique),
        "mean_valid_words": statistics.mean(words) if words else None,
    }
