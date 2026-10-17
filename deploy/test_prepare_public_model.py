import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from deploy.prepare_public_model import PROFILES, prepare

class PreparationTests(unittest.TestCase):
    def test_public_reconstruction_list(self):
        self.assertEqual(len(PROFILES),3)
        self.assertIn("fable-reconstruction-9b",PROFILES)
        self.assertIn("fable-reconstruction-27b",PROFILES)
        self.assertIn("fable-opus-merge-9b",PROFILES)
    def test_offline_default_no_download(self):
        with tempfile.TemporaryDirectory() as t:
            with patch("deploy.prepare_public_model.download") as download:
                result=prepare(PROFILES[0],Path(t),fetch=False)
                download.assert_not_called()
                self.assertEqual(result["status"],"DRY_RUN_NOT_DOWNLOADED")
                self.assertFalse(result["original_anthropic_fable_or_opus"])
    def test_exact_weight_pinning(self):
        with tempfile.TemporaryDirectory() as t:
            for name in PROFILES:
                result=prepare(name,Path(t))
                self.assertEqual(len(result["expected_sha256"]),64)
                self.assertGreater(result["expected_bytes"],5_000_000_000)
                self.assertIn(result["model_revision"],result["source"])
    def test_wrong_file_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            result=prepare(PROFILES[0],Path(t))
            (Path(t)/result["target_filename"]).write_bytes(b"fake model")
            with self.assertRaises(ValueError):
                prepare(PROFILES[0],Path(t))
    def test_disallow_arbitrary_models(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                prepare("claude-fable-5-1",Path(t))
    def test_no_provider_usage(self):
        with tempfile.TemporaryDirectory() as t:
            for p in PROFILES:
                result=prepare(p,Path(t))
                self.assertEqual(result["proprietary_llm_api_calls"],0)
                self.assertEqual(result["paid_cloud_jobs_created"],0)
if __name__=="__main__":
    unittest.main()
