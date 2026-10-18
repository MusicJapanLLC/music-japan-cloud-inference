import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import experimental_qwen122b as q

class Qwen122Tests(unittest.TestCase):
    def test_actual_122b_moe_identifier(self):
        self.assertEqual(q.MODEL_ID,"Qwen3.5-122B-A10B-IQ4_XS")
        self.assertEqual(q.REPO,"AesSedai/Qwen3.5-122B-A10B-GGUF")
        self.assertEqual(len(q.REV),40)
        self.assertEqual(len(q.PARTS),3)
    def test_pinned_parts_have_unique_hashes_and_names(self):
        self.assertEqual(len({p["name"] for p in q.PARTS}),3)
        self.assertEqual(len({p["sha256"] for p in q.PARTS}),3)
        for part in q.PARTS:
            self.assertEqual(len(part["sha256"]),64)
            self.assertGreater(part["bytes"],0)
            self.assertIn(q.REV,q.url(part))
            self.assertTrue(q.url(part).startswith("https://huggingface.co/"))
    def test_all_weight_bytes_fit_ephemeral_runner(self):
        self.assertEqual(q.MODEL_BYTES,63_117_245_760)
        self.assertGreater(q.MIN_FREE_GIB,58)
        self.assertLess(q.MIN_FREE_GIB,85)
    def test_fake_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake=Path(tmp)/"fake.gguf"
            fake.write_bytes(b"not a true GGUF")
            with self.assertRaisesRegex(ValueError,"SHA256"):
                q.verify_part(q.PARTS[0],fake)
    def test_prohibits_random_repo_and_path(self):
        with self.assertRaises(ValueError):q.url({"name":"../../untrusted"})
        with self.assertRaises(ValueError):q.url({"name":q.PARTS[0]["name"],"bytes":42})
    def test_hardware_risk_evaluated(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(q,"linux_memory_gib",return_value=16),\
                 patch.object(q.shutil,"disk_usage",return_value=type("Free",(),{"free":85*2**30})()):
                obs=q.validate_resources(Path(d))
                self.assertGreater(obs["disk_required_gib"],60)
            with patch.object(q,"linux_memory_gib",return_value=8),\
                 patch.object(q.shutil,"disk_usage",return_value=type("Free",(),{"free":85*2**30})()):
                with self.assertRaisesRegex(RuntimeError,"RAM"):
                    q.validate_resources(Path(d))
    def test_narrow_allowlisted_server_paths(self):
        with self.assertRaises(ValueError):
            q.http("secret","https://external-customer.example",timeout=1)
    def test_offline_default_never_downloads(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"result.json"
            proc=subprocess.run([sys.executable,q.__file__,"--binary","does-not-exist",
                "--weights-dir",d,"--out",str(out)],
                capture_output=True,text=True,timeout=5)
            self.assertEqual(proc.returncode,0)
            self.assertIn("DRY_RUN",proc.stdout)
            self.assertFalse(out.exists())
    def test_second_and_third_shards_not_contiguous_aliases(self):
        self.assertTrue(q.PARTS[0]["name"].endswith("-00001-of-00003.gguf"))
        self.assertTrue(q.PARTS[1]["name"].endswith("-00002-of-00003.gguf"))
        self.assertTrue(q.PARTS[2]["name"].endswith("-00003-of-00003.gguf"))
if __name__=="__main__":
    unittest.main()
