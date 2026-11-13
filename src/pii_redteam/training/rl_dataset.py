"""Request bank -> GRPO dataset rows (M08.3 prep).

One row per request, with exactly the prompt the M07b comparison validated (`render_prompt_v2`)
as a single user chat message. The row carries `name` (the requested PERSON value), which TRL
passes to the reward function, and `request_id` for provenance. Chat-template options such as
`enable_thinking=False` belong to the trainer config, not to the rows.
"""

from __future__ import annotations

from ..prompt_v2 import PROMPT_VARIANTS, render_prompt_v2

RL_PROMPT_VARIANT = "P1"


def rl_rows(records: list[dict], variant: str = RL_PROMPT_VARIANT) -> list[dict]:
    """Validated v2 request records -> `{"prompt", "name", "request_id"}` rows."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}")
    rows = []
    seen_ids: set[str] = set()
    for record in records:
        request_id = record["request_id"]
        if request_id in seen_ids:
            raise ValueError(f"duplicate request_id {request_id!r}")
        seen_ids.add(request_id)
        if record.get("schema_version") != 2:
            raise ValueError(f"{request_id}: RL rows need schema_version 2 requests")
        entities = record["task"]["entities"]
        if len(entities) != 1:
            raise ValueError(f"{request_id}: expected one entity, got {len(entities)}")
        rows.append(
            {
                "prompt": [{"role": "user", "content": render_prompt_v2(record, variant)}],
                "name": entities[0]["value"],
                "request_id": request_id,
            }
        )
    return rows
