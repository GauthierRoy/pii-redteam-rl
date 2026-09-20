"""Config loader tests: defaults, validation, stable hashing."""

import os
import unittest

from pii_redteam import config
from pii_redteam.backends import UnsupportedBackendError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig(unittest.TestCase):
    def test_smoke_config_resolves(self):
        cfg = config.resolve_config(
            config.load_config(os.path.join(ROOT, "configs", "smoke", "smoke.yaml"))
        )
        self.assertEqual(cfg["model_id"], "fake-generator-001")
        self.assertEqual(cfg["backend"], {"precision": "fp32", "decoder": "reference"})

    def test_missing_model_id_rejected(self):
        with self.assertRaises(ValueError):
            config.resolve_config({"seed": 0})

    def test_non_mapping_rejected(self):
        with self.assertRaises(ValueError):
            config.resolve_config(["not", "a", "mapping"])

    def test_hash_stable_and_short(self):
        cfg = config.resolve_config({"model_id": "m", "seed": 1})
        self.assertEqual(config.config_hash(cfg), config.config_hash(dict(cfg)))

    def test_unsupported_backend_rejected(self):
        with self.assertRaises(UnsupportedBackendError):
            config.resolve_config(
                {"model_id": "m", "backend": {"precision": "fp8", "decoder": "reference"}}
            )


if __name__ == "__main__":
    unittest.main()
