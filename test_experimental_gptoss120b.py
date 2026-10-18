import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import experimental_gptoss120b as m

class OSS120BTests(unittest.TestCase):
    def test_original_official_model_derivative(self):
        self.assertEqual(m.REPO,"ggml-org/gpt-oss-120b-GGUF")
        self.assertEqual(m.SIZE_BYTES,63_387_346_208)
        self.assertEqual(len(m.SHA256),64)
        self.assertTrue(m.SOURCE.startswith("https://huggingface.co/"))
        self.assertIn(m.REV,m.SOURCE)
    def test_fake_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"not-real.gguf"
            p.write_bytes(b"fake")
            with self.assertRaisesRegex(ValueError,"INTEGRITY"):
                m.verify_weights(p)
    def test_local_resource_guard_fails_if_disk_small(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(m,"linux_memory_gib",return_value=16), patch.object(m.shutil,"disk_usage",return_value=type("D",(),{"free":10*2**30,"total":144*2**30})()):
                with self.assertRaisesRegex(RuntimeError,"DISK"):
                    m.local_resources(Path(t))
    def test_local_resource_guard_fails_if_ram_small(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(m,"linux_memory_gib",return_value=8), patch.object(m.shutil,"disk_usage",return_value=type("D",(),{"free":100*2**30,"total":144*2**30})()):
                with self.assertRaisesRegex(RuntimeError,"RAM"):
                    m.local_resources(Path(t))
    def test_offline_default_never_downloads_model(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/"report.json"
            res=subprocess.run([sys.executable,m.__file__,
                "--binary","missing","--weights",t,"--out",str(out)],
                capture_output=True,text=True,timeout=8)
            self.assertEqual(res.returncode,0)
            self.assertIn("DRY_RUN",res.stdout)
            self.assertFalse(out.exists())
    def test_url_allowlist_on_http(self):
        with self.assertRaises(ValueError):
            m.client("a-key","http://unsafe.tld/api",timeout=1)
    def test_exact_params_not_imitating_gpt6(self):
        self.assertEqual(m.ID,"gpt-oss-120b-MXFP4")
        self.assertNotIn("claude",m.ID.lower())
    def test_cgroup_oom_counters_only(self):
        with tempfile.TemporaryDirectory() as t:
            fp=Path(t)/"events"
            fp.write_text("low 1\\nhigh 2\\nmax 3\\noom 4\\noom_kill 5\\n",encoding="utf8")
            self.assertEqual(m.cgroup_memory_events(fp),{"high":2,"max":3,"oom":4,"oom_kill":5})

    def test_missing_cgroup_is_empty(self):
        self.assertEqual(m.cgroup_memory_events(Path("/nonexistent/mj-cgroup-events")),{})

    def test_log_diagnostic_only_categories(self):
        with tempfile.TemporaryDirectory() as t:
            fp=Path(t)/"log"
            fp.write_text("failed to allocate buffer; OOM killer; mmap=true; account-key-not-for-logs",encoding="utf8")
            result=m.diagnose_runtime(fp)
            self.assertTrue(result["oom"])
            self.assertTrue(result["allocation"])
            self.assertTrue(result["mmap"])
            self.assertNotIn("account-key-not-for-logs",str(result))

    def test_limited_loading_grace(self):
        self.assertLessEqual(m.LOAD_LIMIT_SECONDS,600)
        self.assertGreaterEqual(m.LOAD_LIMIT_SECONDS,300)

    def test_compute_claim_not_baked_in(self):
        self.assertGreater(m.MIN_DISK_FREE_GIB,60)
        self.assertEqual(m.MIN_RAM_GIB,12)

if __name__=="__main__":
    unittest.main()
