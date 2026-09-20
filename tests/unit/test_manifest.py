"""Manifest/resume-guard tests (M01.3 checkpoint-resume gate, stdlib only)."""

import unittest

from pii_redteam import manifest


class TestManifest(unittest.TestCase):
    def test_fresh_dir_has_no_manifest(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(manifest.guard_resume(d, model_id="m", config_hash="h", resume=False))

    def test_existing_dir_without_resume_fails(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest.save_manifest(manifest.new_manifest(model_id="m", config_hash="h", seed=0), d)
            with self.assertRaises(manifest.ResumeError):
                manifest.guard_resume(d, model_id="m", config_hash="h", resume=False)

    def test_same_model_and_config_may_resume(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest.save_manifest(manifest.new_manifest(model_id="m", config_hash="h", seed=0), d)
            existing = manifest.guard_resume(d, model_id="m", config_hash="h", resume=True)
            self.assertEqual(existing["model_id"], "m")

    def test_unrelated_model_resume_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest.save_manifest(
                manifest.new_manifest(model_id="model-a", config_hash="h", seed=0), d
            )
            with self.assertRaises(manifest.ResumeError) as ctx:
                manifest.guard_resume(d, model_id="model-b", config_hash="h", resume=True)
            self.assertIn("unrelated model", str(ctx.exception))

    def test_changed_config_resume_refused(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            manifest.save_manifest(manifest.new_manifest(model_id="m", config_hash="h1", seed=0), d)
            with self.assertRaises(manifest.ResumeError):
                manifest.guard_resume(d, model_id="m", config_hash="h2", resume=True)


if __name__ == "__main__":
    unittest.main()
