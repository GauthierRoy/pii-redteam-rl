"""Generator interface (plan section 7) with a deterministic CPU fake for M01 smoke.

The real SFT/RL generator (M05/M08) plugs into `generate`. The fake emits one
fixed template containing the supplied name exactly once.
"""

from __future__ import annotations


class FakeGenerator:
    """Deterministic mock generator: no sampling, no model, no network."""

    def __init__(self, model_id: str = "fake-generator-001") -> None:
        self.model_id = model_id

    def generate(self, *, supplied_name: str, context: str = "billing") -> str:
        """Return a short message containing the exact supplied name once."""
        return f"Hello, I am {supplied_name} and I need help with my {context} account."
