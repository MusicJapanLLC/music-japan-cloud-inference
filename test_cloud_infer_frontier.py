import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cloud_infer_frontier as f

class FrontierTests(unittest.TestCase):
    def test_exact_weight_provenance(self):
        self.assertEqual(set(f.MODEL_MAP), {"baseline-9b","dense-27b","moe-35b"})
        for name,model in f.MODEL_MAP.items():
            self.assertEqual(len(model["sha256"]),64)
            self.assertGreater(model["bytes"],5_000_000_000)
            self.assertTrue(f.weight_url(model).startswith("https://huggingface.co/"))
            self.assertIn(model["revision"],f.weight_url(model))
    def test_no_false_proprietary_label(self):
        self.assertTrue(all(x["repo"] and "Qwen" in x["model"] for x in f.MODEL_MAP.values()))
        self.assertEqual(f.SCHEMA,"mj.cloud-frontier-benchmark/1.0.0")
    def test_reject_changed_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"bad.gguf"
            p.write_bytes(b"not a model")
            with self.assertRaises(ValueError):
                f.verify_weight(p,f.MODEL_MAP["dense-27b"])
    def test_no_model_inference_without_execute(self):
        import sys,subprocess
        with tempfile.TemporaryDirectory() as t:
            proc=subprocess.run([sys.executable,f.__file__,
                "--profile","dense-27b","--binary","unused","--weights",t,
                "--receipt",str(Path(t)/"proof.json")],
                capture_output=True,text=True,timeout=10)
            self.assertEqual(proc.returncode,0)
            self.assertIn("DRY_RUN",proc.stdout)
            self.assertFalse((Path(t)/"proof.json").exists())
    def test_12_diverse_problems(self):
        self.assertEqual(len(f.CHALLENGES),12)
        self.assertEqual(len({x["id"] for x in f.CHALLENGES}),12)
        self.assertEqual(sum(x["grade"]=="json" for x in f.CHALLENGES),1)
    def test_json_scoring_uses_schema_and_types(self):
        c=next(x for x in f.CHALLENGES if x["id"]=="json_schema")
        self.assertTrue(f.grade('{"count": 3, "ok": true}',c))
        self.assertFalse(f.grade('{"count": "3", "ok": true}',c))
        self.assertFalse(f.grade('{"count": 3, "ok": 1}',c))
    def test_casefold_and_exact_strict(self):
        casefold=next(x for x in f.CHALLENGES if x["id"]=="logical_rule")
        word=next(x for x in f.CHALLENGES if x["id"]=="format")
        self.assertTrue(f.grade("no",casefold))
        self.assertFalse(f.grade("Of course NO!",casefold))
        self.assertTrue(f.grade("ONLINE",word))
        self.assertFalse(f.grade("ONLINE.",word))
    def test_memory_threshold_fails_closed(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(f,"linux_memory_gib",return_value=14):
                with self.assertRaisesRegex(RuntimeError,"MEMORY"):
                    f.available(f.MODEL_MAP["dense-27b"],Path(t))
    def test_reserved_disk_threshold(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(f,"linux_memory_gib",return_value=32):
                with self.assertRaisesRegex(RuntimeError,"DISK"):
                    f.available(f.MODEL_MAP["moe-35b"],Path(t))
    def test_no_untrusted_source(self):
        f.validate = getattr(f,"verify_weight")
        self.assertTrue(f.MODEL_MAP["dense-27b"]["repo"].startswith("bartowski/"))
    def test_unsigned_history_not_present(self):
        self.assertIsNone(f.MODEL_MAP["dense-27b"].get("token"))
if __name__=="__main__":unittest.main()
