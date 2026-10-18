import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import official120b_chat as chat
from cloud_probe import verify
class Official120BChatTests(unittest.TestCase):
    def test_exact_original_openai_public_model_identity(self):
        self.assertEqual(chat.model120b.ID,"gpt-oss-120b-MXFP4")
        self.assertEqual(chat.model120b.SIZE_BYTES,63387346208)
        self.assertEqual(len(chat.model120b.SHA256),64)
        self.assertIn("ggml-org/gpt-oss-120b",chat.model120b.SOURCE)
    def test_invalid_question(self):
        for item in ("","abc","\0hello","x"*4000):
            with self.assertRaises(ValueError):chat.check_prompt(item)
    def test_arbitrary_public_question(self):
        self.assertEqual(chat.check_prompt(" Calculate 314 times 29. "), "Calculate 314 times 29.")
    def test_reasoning_bounds(self):
        chat.check_limits(2,1536,"high")
        for rounds,tokens,effort in ((3,512,"high"),(1,2048,"high"),(1,32,"low"),(1,200,"ultra")):
            with self.assertRaises(ValueError):chat.check_limits(rounds,tokens,effort)
    def test_real_model_message_format(self):
        x=chat.model_body("What is 2+2?",[],"high",512)
        self.assertEqual(x["model"],"gpt-oss-120b-MXFP4")
        self.assertEqual(x["reasoning_effort"],"high")
        self.assertEqual(x["chat_template_kwargs"]["reasoning_effort"],"high")
        self.assertEqual(x["max_tokens"],512)
    def test_second_round_is_an_independent_recheck(self):
        x=chat.model_body("Compute 25*4.",["100"],"medium",192)
        self.assertIn("EARLIER PROPOSAL",x["messages"][1]["content"])
        self.assertIn("100",x["messages"][1]["content"])
    def test_max_uses_sha_bound_20b_independent_proposal(self):
        request=chat.model_body("Calculate 314 times 29.",[],"high",384,
                                proposal_context="9106")
        self.assertEqual(request["model"],"gpt-oss-120b-MXFP4")
        self.assertIn("9106",request["messages"][1]["content"])
        self.assertIn("INDEPENDENT PUBLIC OPEN-WEIGHT",request["messages"][1]["content"])
        self.assertEqual(request["reasoning_effort"],"high")

    def test_ultra_second_round_is_real_120b_reexamination(self):
        content=chat.prompt_for_round("Calculate 314 times 29.",["9106"],proposal_context="other model: 9106")
        self.assertIn("EARLIER PROPOSAL",content)
        self.assertIn("9106",content)
        self.assertIn("ORIGINAL QUESTION",content)

    def test_requires_true_model_response(self):
        payload={"choices":[{"message":{"content":"156"}}],"usage":{
            "completion_tokens":35,
            "completion_tokens_details":{"reasoning_tokens":30}
        }}
        self.assertEqual(chat.extract_answer(200,payload),("156",35,30))
        with self.assertRaises(RuntimeError):chat.extract_answer(503,{"error":"unavailable"})
        with self.assertRaises(RuntimeError):chat.extract_answer(200,{"choices":[{"message":{"reasoning_content":"private"}}]})
    def test_no_fake_gpt6_identity(self):
        src=Path(chat.__file__).read_text()
        self.assertIn("model_original_gpt6",src)
        self.assertIn('"--load-mode","mmap","--fit","off","--no-repack"',src)
    def test_default_dry_run_no_hosted_inference(self):
        with tempfile.TemporaryDirectory() as t:
            outfile=Path(t)/"out.json"
            res=subprocess.run([sys.executable,chat.__file__,
                "--binary","nonexistent","--weights-dir",t,"--out",str(outfile)],
                capture_output=True,text=True,timeout=6)
            self.assertEqual(res.returncode,0)
            self.assertIn("DRY_RUN",res.stdout)
            self.assertFalse(outfile.exists())
    def test_fake_local_reply_chain_acceptance(self):
        class FakeProcess:
            def poll(self):return None
            def terminate(self):pass
            def wait(self,timeout=10):return 0
        answer={"choices":[{"message":{"content":"9106"}}],"usage":{"completion_tokens":48}}
        def fake_call(key,path,body=None,timeout=10):
            if path=="/health":return 200,{"status":"ok"}
            if path=="/v1/models":return 200,{"data":[{"id":chat.model120b.ID}]}
            return 200,answer
        with tempfile.TemporaryDirectory() as t:
            binary=Path(t)/"llama-server";binary.touch()
            weights=Path(t)/"weights.gguf";weights.touch()
            with patch.object(chat.subprocess,"Popen",return_value=FakeProcess()),patch.object(chat.model120b,"client",side_effect=fake_call):
                r=chat.run("Calculate 314 times 29.",binary,weights,Path(t)/"answer.json",
                           rounds=2,tokens=192,effort="high",budget_seconds=300)
                self.assertTrue(verify(r))
                self.assertEqual(r["rounds"],2)
                self.assertEqual(r["selection"]["final_answer"],"9106")
                self.assertEqual(r["selection"]["verification"]["status"],"VERIFIED")
                self.assertEqual(r["model_provider_credits_used"],0)
                self.assertTrue(r["actual_local_120b_inference"])
    def test_no_remote_api_paths(self):
        with self.assertRaises(ValueError):
            chat.model120b.client("token","https://evil.test",timeout=1)
if __name__=="__main__":
    unittest.main()
