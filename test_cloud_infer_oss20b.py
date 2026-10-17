import tempfile,subprocess,sys,unittest
from pathlib import Path
import cloud_infer_oss20b as mod
class OSS20BTests(unittest.TestCase):
 def test_actual_original_openai_model_conversion(self):
    self.assertIn("gpt-oss-20b",mod.HF_REPO)
    self.assertIn("20B",mod.MODEL)
    self.assertEqual(len(mod.SHA256),64)
    self.assertGreater(mod.SIZE,12_000_000_000)
    self.assertTrue(mod.URL.startswith("https://huggingface.co/"))
 def test_dryrun_no_cost(self):
    with tempfile.TemporaryDirectory() as d:
        p=subprocess.run([sys.executable,mod.__file__,"--binary","none","--data",d,"--out",d+"/proof.json"],
        capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0)
        self.assertIn("DRY_RUN",p.stdout)
        self.assertFalse((Path(d)/"proof.json").exists())
 def test_fake_weights_rejected(self):
    with tempfile.TemporaryDirectory() as d:
        fake=Path(d)/"fake.gguf";fake.write_bytes(b"FAKE")
        with self.assertRaises(ValueError):mod.verify(fake)
 def test_three_quality_tests(self):
    self.assertEqual(len(mod.PROMPTS),3)
if __name__=="__main__":unittest.main()
