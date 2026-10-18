import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cloud_api_acceptance as a

class AcceptanceTests(unittest.TestCase):
    def test_three_provider_shapes(self):
        self.assertEqual([x[0] for x in a.CASES],
                         ["openai_chat","anthropic_messages","openai_responses"])
        self.assertEqual(len({x[1] for x in a.CASES}),3)
    def test_parse_openai_chat(self):
        self.assertEqual(a.body_text("openai_chat",{
            "choices":[{"message":{"content":"MUSICJAPAN"}}]}),"MUSICJAPAN")
    def test_parse_anthropic(self):
        self.assertEqual(a.body_text("anthropic_messages",{
            "content":[{"type":"text","text":"MUSIC"},{"type":"text","text":"JAPAN"}]}),
            "MUSICJAPAN")
    def test_parse_openai_responses(self):
        self.assertEqual(a.body_text("openai_responses",{"output":[{
            "type":"message","content":[{"type":"output_text","text":"MUSICJAPAN"}]}]}),
            "MUSICJAPAN")
        self.assertEqual(a.body_text("openai_responses",{"output_text":"MUSICJAPAN"}),"MUSICJAPAN")
    def test_partial_endpoints_are_not_successes(self):
        result=a.evaluate_one("anthropic_messages",200,{"content":[]},2.0)
        self.assertFalse(result["model_generated_visible_text"])
        self.assertFalse(result["exact_instruction_pass"])
    def test_wrong_status_is_not_completion(self):
        self.assertFalse(a.evaluate_one("openai_chat",503,{"error":"fail"},1)["model_generated_visible_text"])
    def test_declares_not_anthropic_original(self):
        x=a.evaluate_one("openai_responses",200,{"output_text":"MUSICJAPAN"},1.3)
        self.assertTrue(x["identity_not_original_anthropic"])
        self.assertTrue(x["exact_instruction_pass"])
    def test_reject_unknown_host_path(self):
        with self.assertRaises(ValueError):
            a.get_http("secret","GET","https://evil.test/",timeout=1)
    def test_no_credentials_in_test_messages(self):
        self.assertNotIn("api_key",json.dumps(a.CASES).lower())
    def test_dry_run_starts_no_model(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/"receipt.json"
            res=subprocess.run([
                sys.executable,str(Path(a.__file__).resolve()),
                "--binary","missing","--weights-dir",temp,"--out",str(out)
            ],capture_output=True,text=True,timeout=6)
            self.assertEqual(res.returncode,0)
            self.assertIn("DRY_RUN",res.stdout)
            self.assertFalse(out.exists())
if __name__=="__main__":unittest.main()
