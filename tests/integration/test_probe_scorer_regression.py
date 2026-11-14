"""M07a probe-scorer regression: frozen base detector must exact-hit 5 fixed AI4Privacy dev rows.

Guards against regressions in `_score_with_detector` / decode / alignment after the
2026-10-11 sanity investigation (docs/PROBE_SCORER_SANITY.md: verdict REAL SHIFT,
scorer verified correct). Requires torch+transformers and the frozen base detector checkpoint:
set PII_BASE_DETECTOR_PATH to the d0-best directory (skipped otherwise - CPU-only CI does not
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

BASE_DETECTOR_PATH = os.environ.get("PII_BASE_DETECTOR_PATH", "")
HAS_BASE_DETECTOR = bool(BASE_DETECTOR_PATH) and os.path.exists(
    os.path.join(BASE_DETECTOR_PATH, "model.safetensors")
)

# The REAL-SHIFT tripwire is pinned to the frozen base detector specifically (sanity verdict,
# docs/PROBE_SCORER_SANITY.md). Against a different checkpoint (e.g. a future target detector,
# where short-message misses are EXPECTED to shrink) it must not fire.
FROZEN_BASE_DETECTOR_SHA_PREFIX = "0314ebe36d7f4dc6"


@unittest.skipUnless(
    HAS_ML and HAS_BASE_DETECTOR, "needs torch/transformers and PII_BASE_DETECTOR_PATH"
)
class TestProbeScorerRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sha = hashlib.sha256(
            open(os.path.join(BASE_DETECTOR_PATH, "model.safetensors"), "rb").read()
        ).hexdigest()
        fixture = json.load(open(FIXTURE, encoding="utf-8"))
        assert sha.startswith(fixture["base_detector_sha256_prefix"]), (
            f"frozen base detector sha mismatch: {sha}"
        )
        cls.tok = AutoTokenizer.from_pretrained(BASE_DETECTOR_PATH)
        cls.base_detector = AutoModelForTokenClassification.from_pretrained(BASE_DETECTOR_PATH)
        cls.base_detector.eval()
        cls.rows = fixture["rows"]

    def test_five_dev_rows_exact_hit(self):
        samples = [
            {"text": r["text"], "target_span": tuple(r["target_span"]), "status": "valid"}
            for r in self.rows
        ]
        from pii_redteam.probe import _score_with_detector

        _score_with_detector(self.base_detector, self.tok, samples, 512)
        for r, s in zip(self.rows, samples, strict=True):
            self.assertEqual(s["outcome"], "exact_hit", f"{r['example_id']}: {s['outcome']}")
            self.assertGreater(
                s["p_target_mean"], 0.9, f"{r['example_id']}: p={s['p_target_mean']}"
            )

    def test_short_conversational_text_is_a_genuine_miss(self):
        """REAL SHIFT tripwire, PINNED TO FROZEN base detector (sha 0314ebe3…).

        Frozen base detector truly misses short conversational names (sanity verdict). If this
        flips to exact_hit on the frozen base detector, base detector's behavior changed - revisit
        docs/PROBE_SCORER_SANITY.md and the probe reward design before trusting it.
        Skipped for any other checkpoint (a future target detector SHOULD hit this text).
        """
        sha = hashlib.sha256(
            open(os.path.join(BASE_DETECTOR_PATH, "model.safetensors"), "rb").read()
        ).hexdigest()
        if not sha.startswith(FROZEN_BASE_DETECTOR_SHA_PREFIX):
            self.skipTest(
                f"tripwire pinned to the frozen base detector ({FROZEN_BASE_DETECTOR_SHA_PREFIX}…); "
                f"loaded checkpoint is not base detector ({sha[:16]}…)"
            )
        s = {
            "text": "Hey Lucas, how are you doing today?",
            "target_span": (4, 9),
            "status": "valid",
        }
        from pii_redteam.probe import _score_with_detector

        _score_with_detector(self.base_detector, self.tok, [s], 512)
        self.assertEqual(s["outcome"], "complete_miss")
        self.assertLess(s["p_target_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
