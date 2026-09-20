"""Backend contract tests: reference runs, everything else fails usefully."""

import unittest

from pii_redteam import backends
from pii_redteam.backends import UnsupportedBackendError


class TestBackends(unittest.TestCase):
    def test_reference_validates(self):
        spec = backends.validate_backend({"precision": "fp32", "decoder": "reference"})
        self.assertEqual((spec.precision, spec.decoder), ("fp32", "reference"))

    def test_fp8_rejected_with_s01_pointer(self):
        with self.assertRaises(UnsupportedBackendError) as ctx:
            backends.validate_backend({"precision": "fp8", "decoder": "reference"})
        self.assertIn("S01", str(ctx.exception))

    def test_speculative_rejected_with_s03_pointer(self):
        with self.assertRaises(UnsupportedBackendError) as ctx:
            backends.validate_backend({"precision": "fp32", "decoder": "speculative-ngram"})
        self.assertIn("S03", str(ctx.exception))

    def test_dflash_rejected_with_s04_pointer(self):
        with self.assertRaises(UnsupportedBackendError) as ctx:
            backends.validate_backend({"precision": "fp32", "decoder": "dflash"})
        self.assertIn("S04", str(ctx.exception))

    def test_unknown_precision_names_known_options(self):
        with self.assertRaises(UnsupportedBackendError) as ctx:
            backends.validate_backend({"precision": "int4", "decoder": "reference"})
        self.assertIn("fp32", str(ctx.exception))

    def test_capabilities_advertise_reference_only(self):
        caps = backends.capabilities()
        self.assertEqual(caps["precisions"], ["fp32"])
        self.assertFalse(caps["speculative"])


if __name__ == "__main__":
    unittest.main()
