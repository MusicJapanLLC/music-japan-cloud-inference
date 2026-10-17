import tempfile,subprocess,sys,unittest
from pathlib import Path
import cloud_infer_9b as mod
class ModelTests(unittest.TestCase):
 def test_pinned_real_9b(self):
    self.assertIn("9B",mod.MODEL)
    self.assertEqual(len(mod.SHA256),64)
    self.assertTrue(mod.URL.startswith("https://huggingface.co/"))
    self.assertGreater(mod.SIZE,5_000_000_000)
 def test_wrong_file_rejected(self):
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"bad.gguf";p.write_bytes(b"bad")
        with self.assertRaises(ValueError):mod.verify(p)
 def test_default_offline(self):
    with tempfile.TemporaryDirectory() as d:
        p=subprocess.run([sys.executable,mod.__file__,"--binary","none","--data",d,"--out",d+"/out.json"],
            capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0)
        self.assertIn("DRY_RUN",p.stdout)
        self.assertFalse((Path(d)/"out.json").exists())
 def test_quality_tasks(self):
    self.assertEqual(len(mod.PROMPTS),3)
    self.assertTrue(all("needle" in x for x in mod.PROMPTS))
if __name__=="__main__":unittest.main()
