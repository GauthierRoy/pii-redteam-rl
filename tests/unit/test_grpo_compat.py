"""grpo_compat tests: unknown TRL keys dropped, known keys kept (no `trl` needed).

Reused component from ../rl — the one item this suite covers that the source repo
never tested (see docs/REUSE_AUDIT.md section 3).
"""

import sys
import types
import unittest

from pii_redteam.training import grpo_compat


class FakeGRPOConfig:
    """Stand-in with a fixed accepted signature."""

    def __init__(self, output_dir=None, max_steps=None):
        self.output_dir = output_dir
        self.max_steps = max_steps


class TestGrpoCompat(unittest.TestCase):
    def test_drops_unknown_keys(self):
        fake_trl = types.ModuleType("trl")
        fake_trl.GRPOConfig = FakeGRPOConfig
        sys.modules["trl"] = fake_trl
        try:
            cfg = grpo_compat.build_grpo_config(output_dir="o", max_steps=5, does_not_exist=1)
        finally:
            del sys.modules["trl"]
        self.assertEqual((cfg.output_dir, cfg.max_steps), ("o", 5))
        self.assertFalse(hasattr(cfg, "does_not_exist"))

    def test_module_imports_without_trl(self):
        self.assertNotIn("trl", sys.modules)
        import importlib

        importlib.reload(grpo_compat)
        self.assertNotIn("trl", sys.modules)


if __name__ == "__main__":
    unittest.main()
