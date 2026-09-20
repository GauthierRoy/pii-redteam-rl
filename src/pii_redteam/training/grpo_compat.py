"""Version-tolerant GRPOConfig construction.

Reused from ../rl (rl-mini) src/grpo_compat.py — see docs/REUSE_AUDIT.md.
M01 adaptation: `trl` is imported lazily so CPU-only environments can import this
module without the training stack; construction still requires `trl` installed.
Logic otherwise verbatim (pass everything, keep what the installed TRL accepts).
"""

from __future__ import annotations

import inspect


def build_grpo_config(**kwargs):
    """Build GRPOConfig from loose kwargs, warning-and-dropping unknown keys."""
    from trl import GRPOConfig

    accepted = set(inspect.signature(GRPOConfig).parameters)
    dropped = [k for k in kwargs if k not in accepted]
    for k in dropped:
        print(f"warn: installed GRPOConfig has no {k!r}, dropping it")
    return GRPOConfig(**{k: v for k, v in kwargs.items() if k in accepted})
