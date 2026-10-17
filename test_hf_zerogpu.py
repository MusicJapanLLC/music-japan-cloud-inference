"""Offline lint only; no HF Space or GPU is created by tests."""
import ast
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent/"hf_zerogpu"
class SpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=(ROOT/"app.py").read_text("utf8")
        cls.node=ast.parse(cls.source)
    def test_python_parses(self):
        self.assertIsInstance(self.node,ast.Module)
    def test_model_identity(self):
        self.assertIn('Qwen/Qwen3.5-9B',self.source)
        self.assertNotIn("claude-fable-5-1",self.source)
    def test_gpu_decorator(self):
        self.assertIn('@spaces.GPU(size="large",duration=60)',self.source)
    def test_text_only(self):
        self.assertIn('{"type":"text","text":message}',self.source)
    def test_gradio_and_supported_python(self):
        readme=(ROOT/"README.md").read_text("utf8")
        self.assertIn("sdk: gradio",readme)
        self.assertIn("python_version: 3.12.12",readme)
    def test_no_model_api_credential(self):
        for item in ("ANTHROPIC_API_KEY","OPENAI_API_KEY","HF_TOKEN","os.environ"):
            self.assertNotIn(item,self.source)
if __name__=="__main__":unittest.main()
