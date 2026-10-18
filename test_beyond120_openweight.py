"""Offline-only tests for truly public 235B and 397B Qwen checkpoints."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import beyond120_openweight as f

class ExtremeOpenWeights(unittest.TestCase):
    def test_two_models_genuinely_above_120b(self):
        self.assertEqual(set(f.MODELS),{"thinking-235b-iq2m","frontier-397b-iq1s"})
        self.assertEqual(f.MODELS["thinking-235b-iq2m"]["params_b"],235)
        self.assertEqual(f.MODELS["frontier-397b-iq1s"]["params_b"],397)
        self.assertTrue(all(f.check_manifest(k) for k in f.MODELS))

    def test_original_source_and_exact_files(self):
        for key,cfg in f.MODELS.items():
            self.assertEqual(len(cfg["revision"]),40)
            self.assertTrue(cfg["repo"].startswith("bartowski/"))
            self.assertGreater(f.model_size_gib(key),70)
            for part in cfg["parts"]:
                self.assertTrue(f.pinned_url(key,part).startswith("https://huggingface.co/"))
                self.assertIn(cfg["revision"],f.pinned_url(key,part))
                self.assertEqual(len(part[2]),64)

    def test_altered_model_bytes_blocked(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"broken.gguf"
            p.write_bytes(b"not original")
            for cfg in f.MODELS.values():
                with self.assertRaises(ValueError):
                    f.verify_part(p,cfg["parts"][0])

    def test_prevent_untrusted_publisher(self):
        with self.assertRaises(ValueError):
            f.check_manifest("unregistered/huge-private-model")
        with self.assertRaises(ValueError):
            f.pinned_url("thinking-235b-iq2m",("fake.gguf",1234,"a"*64))

    def test_hardware_gate_disallows_unknown_ram(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(f,"linux_memory_gib",return_value=None):
                with self.assertRaises(RuntimeError):
                    f.resource_gate("thinking-235b-iq2m",Path(t))

    def test_397_disk_minimum(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(f,"linux_memory_gib",return_value=16),patch.object(f.shutil,"disk_usage",return_value=type("Disk",(),{"free":80*2**30,"total":144*2**30})()):
                with self.assertRaisesRegex(RuntimeError,"DISK"):
                    f.resource_gate("frontier-397b-iq1s",Path(t))
            with patch.object(f,"linux_memory_gib",return_value=16),patch.object(f.shutil,"disk_usage",return_value=type("Disk",(),{"free":85*2**30,"total":144*2**30})()):
                r=f.resource_gate("frontier-397b-iq1s",Path(t))
                self.assertGreater(r["required_free_gib"],82)

    def test_offline_mode_no_network_no_jobs(self):
        with tempfile.TemporaryDirectory() as t:
            p=subprocess.run([sys.executable,str(Path(f.__file__).resolve()),
                "--profile","thinking-235b-iq2m","--binary","unused",
                "--data",t,"--out",str(Path(t)/"proof.json")],
                capture_output=True,text=True,timeout=7)
            self.assertEqual(p.returncode,0)
            self.assertIn("DRY_RUN",p.stdout)
            self.assertFalse((Path(t)/"proof.json").exists())

    def test_cgroup_only_numeric_oom_counts(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"mem"
            p.write_text("oom 7\\noom_kill 3\\nhigh 0\\nsecret password\\n",encoding="utf8")
            v=f.read_cgroup(p)
            self.assertTrue(set(v).issubset({"oom","oom_kill","max","high"}))

    def test_no_third_party_inference_request(self):
        with self.assertRaises(ValueError):
            f.local_http("key","https://huggingface.co/private/token")

    def test_fixed_public_synthetic_problems(self):
        self.assertEqual(len(f.PROMPTS),2)
        self.assertIn("72 / 8",f.PROMPTS[0])
        self.assertNotIn("SECRET",f.PROMPTS[0])
        self.assertLess(f.MODELS["frontier-397b-iq1s"]["max_output_tokens"],200)

    def test_intentional_low_bit_quality_is_not_assumed(self):
        self.assertEqual(f.MODELS["frontier-397b-iq1s"]["quant"],"IQ1_S")
        self.assertEqual(f.MODELS["thinking-235b-iq2m"]["quant"],"IQ2_M")
        self.assertGreater(f.model_size_gib("frontier-397b-iq1s"),80)

if __name__=="__main__":
    unittest.main()
