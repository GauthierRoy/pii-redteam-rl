"""M07a probe-scorer regression: frozen D0 must exact-hit 5 fixed AI4Privacy dev rows.

Guards against regressions in `_score_with_d0` / decode / alignment after the
2026-10-11 sanity investigation (docs/PROBE_SCORER_SANITY.md: verdict REAL SHIFT,
scorer verified correct). Requires torch+transformers and the frozen D0 checkpoint:
set PII_D0_PATH to the d0-best directory (skipped otherwise - CPU-only CI does not
carry the checkpoint).
"""

import hashlib
import json
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "probe_scorer_regression.json")

try:
    import torch  # noqa: F401
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    HAS_ML = True
except ImportError:
    HAS_ML = False

D0_PATH = os.environ.get("PII_D0_PATH", "")
HAS_D0 = bool(D0_PATH) and os.path.exists(os.path.join(D0_PATH, "model.safetensors"))

# The REAL-SHIFT tripwire is pinned to frozen D0 specifically (sanity verdict,
# docs/PROBE_SCORER_SANITY.md). Against a different checkpoint (e.g. a future D1,
# where short-message misses are EXPECTED to shrink) it must not fire.
FROZEN_D0_SHA_PREFIX = "0314ebe36d7f4dc6"


@unittest.skipUnless(HAS_ML and HAS_D0, "needs torch/transformers and PII_D0_PATH")
class TestProbeScorerRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sha = hashlib.sha256(
            open(os.path.join(D0_PATH, "model.safetensors"), "rb").read()
        ).hexdigest()
        fixture = json.load(open(FIXTURE, encoding="utf-8"))
        assert sha.startswith(fixture["d0_sha256_prefix"]), f"frozen-D0 sha mismatch: {sha}"
        cls.tok = AutoTokenizer.from_pretrained(D0_PATH)
        cls.d0 = AutoModelForTokenClassification.from_pretrained(D0_PATH)
        cls.d0.eval()
        cls.rows = fixture["rows"]

    def test_five_dev_rows_exact_hit(self):
        samples = [
            {"text": r["text"], "target_span": tuple(r["target_span"]), "status": "valid"}
            for r in self.rows
        ]
        from pii_redteam.probe import _score_with_d0

        _score_with_d0(self.d0, self.tok, samples, 512)
        for r, s in zip(self.rows, samples, strict=True):
            self.assertEqual(s["outcome"], "exact_hit", f"{r['example_id']}: {s['outcome']}")
            self.assertGreater(
                s["p_target_mean"], 0.9, f"{r['example_id']}: p={s['p_target_mean']}"
            )

    def test_short_conversational_text_is_a_genuine_miss(self):
        """REAL SHIFT tripwire, PINNED TO FROZEN D0 (sha 0314ebe3…).

        Frozen D0 truly misses short conversational names (sanity verdict). If this
        flips to exact_hit on frozen D0, D0's behavior changed - revisit
        docs/PROBE_SCORER_SANITY.md and the probe reward design before trusting it.
        Skipped for any other checkpoint (a future D1 SHOULD hit this text).
        """
        sha = hashlib.sha256(
            open(os.path.join(D0_PATH, "model.safetensors"), "rb").read()
        ).hexdigest()
        if not sha.startswith(FROZEN_D0_SHA_PREFIX):
            self.skipTest(
                f"tripwire pinned to frozen D0 ({FROZEN_D0_SHA_PREFIX}…); "
                f"loaded checkpoint is not D0 ({sha[:16]}…)"
            )
        s = {
            "text": "Hey Lucas, how are you doing today?",
            "target_span": (4, 9),
            "status": "valid",
        }
        from pii_redteam.probe import _score_with_d0

        _score_with_d0(self.d0, self.tok, [s], 512)
        self.assertEqual(s["outcome"], "complete_miss")
        self.assertLess(s["p_target_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
