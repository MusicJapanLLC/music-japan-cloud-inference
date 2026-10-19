import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cloud_owner_chat as chat
from cloud_probe import verify

class PublicCloudChatTests(unittest.TestCase):
    def test_model_pinned_to_real_official_public_weights(self):
        self.assertEqual(chat.CFG["model"],"gpt-oss-20b-MXFP4")
        self.assertEqual(len(chat.CFG["sha256"]),64)
        self.assertGreater(chat.CFG["bytes"],11_000_000_000)
    def test_question_restrictions(self):
        self.assertEqual(chat.validate_prompt(" What is 4*5? "),"What is 4*5?")
        for bad in ("","hi","a"*3500,"\0\0\0a"):
            with self.assertRaises(ValueError):chat.validate_prompt(bad)
    def test_reasoning_budget_bounded(self):
        chat.safe_budgets(3,3200,1500)
        for rounds,tokens,seconds in ((4,1200,500),(2,5000,500),(1,128,500),(2,600,8000)):
            with self.assertRaises(ValueError):chat.safe_budgets(rounds,tokens,seconds)
    def test_build_real_gpt_oss_high_effort_request(self):
        request=chat.model_message("Hello",1280,"high")
        self.assertEqual(request["model"],"gpt-oss-20b-MXFP4")
        self.assertEqual(request["reasoning_effort"],"high")
        self.assertEqual(request["chat_template_kwargs"]["reasoning_effort"],"high")
        self.assertEqual(request["max_tokens"],1280)
        self.assertNotIn("OPENAI_API_KEY",str(request))
    def test_invalid_effort_fails(self):
        with self.assertRaises(ValueError):chat.model_message("hello",1280,"ultra")
    def test_discussion_stages_are_distinct(self):
        first=chat.stage_prompt("What is 7x8?",[])
        second=chat.stage_prompt("What is 7x8?",["56"])
        third=chat.stage_prompt("What is 7x8?",["56","54"])
        self.assertIn("deep reasoning",first)
        self.assertIn("earlier answer",second)
        self.assertIn("SECOND ATTEMPT",third)
        self.assertNotIn("EXPECTED_ANSWER",first)
    def test_extract_model_output_requires_real_text(self):
        msg={"choices":[{"message":{"content":"42"}}],"usage":{"completion_tokens":8,"completion_tokens_details":{"reasoning_tokens":6}}}
        self.assertEqual(chat.extract_reply(200,msg),("42",8,6))
        with self.assertRaises(RuntimeError):chat.extract_reply(503,{"error":"x"})
        with self.assertRaises(RuntimeError):chat.extract_reply(200,{"choices":[{"message":{"reasoning_content":"hidden"}}]})
    def test_question_grounded_calculator_correction(self):
        out=chat.select_final("Calculate 314 times 29.",["9106","9130"])
        self.assertEqual(out["final_answer"],"9106")
        self.assertTrue(out["tool_corrected"])
        self.assertEqual(out["model_answer"],"9130")
    def test_unverified_consensus_not_overstated(self):
        out=chat.select_final("Which year was this novel published?",["not sure","unverified"])
        self.assertFalse(out["tool_corrected"])
        self.assertEqual(out["final_answer"],"unverified")
        self.assertEqual(out["verification"]["status"],"ABSTAIN")
    def test_request_path_allowlist(self):
        with self.assertRaises(ValueError):chat.request_local("key","https://other-service.invalid/",{})
    def test_default_offline_no_model_start(self):
        with tempfile.TemporaryDirectory() as t:
            result=subprocess.run([sys.executable,str(Path(chat.__file__).resolve()),
                "--binary","missing","--weights-dir",t,"--out",str(Path(t)/"receipt.json")],
                capture_output=True,text=True,timeout=6)
            self.assertEqual(result.returncode,0)
            self.assertIn("DRY_RUN",result.stdout)
            self.assertFalse((Path(t)/"receipt.json").exists())
    def test_fake_model_response_chain_receipt(self):
        class FakeProcess:
            def poll(self):return None
            def terminate(self):pass
            def wait(self,timeout):return 0
        answer={"choices":[{"message":{"content":"437"}}],"usage":{"completion_tokens":23,
               "completion_tokens_details":{"reasoning_tokens":10}}}
        def fake_http(key,path,body=None,timeout=15):
            if path=="/health":return 200,{"status":"ok"}
            if path=="/v1/models":return 200,{"data":[{"id":chat.CFG["model"]}]}
            return 200,answer
        with tempfile.TemporaryDirectory() as t, patch.object(chat,"open_model",return_value=FakeProcess()), patch.object(chat,"request_local",side_effect=fake_http):
            outfile=Path(t)/"out.json"
            result=chat.run_chat("Calculate 19 times 23.",Path(t)/"runtime",Path(t)/"weights",
                                 outfile,rounds=2,tokens=1280,effort="high",max_seconds=300)
            self.assertTrue(verify(result))
            self.assertEqual(result["rounds"],2)
            self.assertEqual(result["selection"]["final_answer"],"437")
            self.assertEqual(result["external_model_api_calls"],0)
            self.assertEqual(result["paid_compute_created"],False)
            self.assertEqual(result["selection"]["verification"]["status"],"VERIFIED")
            self.assertNotIn("Calculate 19 times 23.",outfile.read_text())
    def test_true_public_prompt_only_not_logged(self):
        import inspect
        source=inspect.getsource(chat.run_chat)
        self.assertNotIn("print(question",source)
        self.assertNotIn("print(stage",source)
if __name__=="__main__":unittest.main()
