"""Proof and regression for multi-stage reasoning pipeline. No model/credits."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from cloud_probe import signed_record
from cloud_infer_frontier import CHALLENGES
import reasoning_mesh as mesh

def make_receipt(stage, answers=None, upstream=None, challenger=None, limit=12):
    items=[]
    for i,case in enumerate(CHALLENGES[:limit]):
        answer=answers[i] if answers is not None else case["expected"]
        items.append({"id":case["id"],"answer":answer,"latency_seconds":2.0,
                      "http_status":200,"output_tokens":10,
                      "request_budget_tokens":1024,
                      "observed_reasoning_tokens":0,
                      "generated_visible_text":bool(answer)})
    return signed_record({
        "schema":mesh.SCHEMA,"stage":stage,"model_profile":mesh.STAGES[stage],
        "model_id":"test","model_revision":"test","weights_sha256":"test",
        "reasoning_effort":"high",
        "requested_output_tokens_per_task":1024,
        "upstream_sha256":upstream,
        "challenge_sha256":challenger,
        "new_spend_usd":0,"proprietary_model_credits":0,
        "original_claude_or_gpt6":False,"public_runner_only":True,
        "items":items,"total_elapsed_seconds":24.0,
        "observed_at_utc":"2026-10-18T03:00:00+00:00"})

class MeshTests(unittest.TestCase):
    def test_genuine_existing_models(self):
        self.assertEqual(mesh.STAGES["propose"],"openai-20b")
        self.assertEqual(mesh.STAGES["challenge"],"fable-reconstruction-9b")
        self.assertEqual(mesh.STAGES["synthesize"],"openai-20b")
    def test_dry_run_does_not_need_credentials_or_weights(self):
        p=subprocess.run([sys.executable,str(Path(mesh.__file__).resolve()),
              "--stage","propose"],capture_output=True,text=True,timeout=7)
        self.assertEqual(p.returncode,0)
        self.assertIn('"inference": false',p.stdout)
    def test_high_effort_real_gpt_oss_payload(self):
        x=mesh.build_request("openai-20b","propose","Hello",effort="high",token_budget=1400)
        self.assertEqual(x["reasoning_effort"],"high")
        self.assertEqual(x["chat_template_kwargs"]["reasoning_effort"],"high")
        self.assertEqual(x["max_tokens"],1400)
        self.assertEqual(x["temperature"],1.0)
        self.assertNotIn("api_key",json.dumps(x).lower())
    def test_fable_challenger_does_not_forge_openai_effort(self):
        x=mesh.build_request("fable-reconstruction-9b","challenge","Hi",token_budget=1200)
        self.assertNotIn("reasoning_effort",x)
        self.assertEqual(x["max_tokens"],384)
        self.assertEqual(x["chat_template_kwargs"]["enable_thinking"],False)
    def test_invalid_profile_and_budget_fails(self):
        with self.assertRaises(ValueError):mesh.build_request("openai-20b","challenge","Hi")
        with self.assertRaises(ValueError):mesh.build_request("openai-20b","propose","Hi",token_budget=4000)
    def test_stage_prompts_do_not_leak_answer_keys(self):
        case=CHALLENGES[0]
        prompt=mesh.stage_instruction("propose",case)
        self.assertIn(case["prompt"],prompt)
        self.assertNotIn(case["expected"],prompt)
        self.assertNotIn("expected",prompt)
    def test_challenger_is_given_proposal(self):
        p=mesh.stage_instruction("challenge",CHALLENGES[0],"untrusted candidate")
        self.assertIn("untrusted candidate",p)
        self.assertIn("INDEPENDENT",p)
    def test_synthesis_contains_two_candidates(self):
        p=mesh.stage_instruction("synthesize",CHALLENGES[0],"FIRST","SECOND")
        self.assertIn("FIRST",p)
        self.assertIn("SECOND",p)
        self.assertNotIn("437",p)
    def test_stage_receipt_chain_and_comparison(self):
        a=make_receipt("propose")
        b=make_receipt("challenge",upstream=a["sha256"])
        c=make_receipt("synthesize",upstream=a["sha256"],challenger=b["sha256"])
        result=mesh.evaluate(a,b,c)
        self.assertEqual(result["scores"],{"propose":12,"challenge":12,"synthesize":12})
        self.assertEqual(result["change_from_proposal"],0)
        self.assertTrue(result["evaluation_only_not_an_oracle_router"])
        self.assertEqual(result["proprietary_model_credits_used"],0)
    def test_can_measure_real_improvement(self):
        wrong=["wrong"]*12
        a=make_receipt("propose",wrong)
        b=make_receipt("challenge",upstream=a["sha256"])
        c=make_receipt("synthesize",upstream=a["sha256"],challenger=b["sha256"])
        res=mesh.evaluate(a,b,c)
        self.assertEqual(res["change_from_proposal"],12)
        self.assertEqual(res["recommendation"],"MORE_REASONING_HELPED_THIS_EVAL")
    def test_overthinking_can_be_worse(self):
        a=make_receipt("propose")
        b=make_receipt("challenge",upstream=a["sha256"])
        c=make_receipt("synthesize",["wrong"]*12,upstream=a["sha256"],challenger=b["sha256"])
        r=mesh.evaluate(a,b,c)
        self.assertEqual(r["change_from_proposal"],-12)
        self.assertEqual(r["recommendation"],"MORE_REASONING_DID_NOT_IMPROVE_THIS_EVAL")
    def test_wrong_parent_sha_rejected(self):
        a=make_receipt("propose")
        b=make_receipt("challenge",upstream="bad")
        c=make_receipt("synthesize",upstream=a["sha256"],challenger=b["sha256"])
        with self.assertRaisesRegex(ValueError,"CHALLENGER_PARENT_MISMATCH"):
            mesh.evaluate(a,b,c)
    def test_tampered_receipt_rejected(self):
        a=make_receipt("propose")
        a["items"][0]["answer"]="hacked"
        self.assertFalse(mesh.receipt_ok(a))
    def test_wrong_profile_rejected(self):
        a=make_receipt("propose")
        a["model_profile"]="impostor"
        a=signed_record(a)
        self.assertFalse(mesh.receipt_ok(a))
    def test_wrong_stage_order_rejected(self):
        a=make_receipt("propose")
        a["items"]=list(reversed(a["items"]))
        a=signed_record(a)
        self.assertFalse(mesh.receipt_ok(a))
    def test_real_api_parser_does_not_forge_text(self):
        self.assertEqual(mesh.parse_answer({"choices":[{"message":{"content":"ok"}}]}),"ok")
        self.assertEqual(mesh.parse_answer({"choices":[{"message":{"reasoning_content":"secret"}}]}),"")
        self.assertEqual(mesh.parse_answer({"error":"failure"}),"")
    def test_remote_nonallowlisted_api_denied(self):
        with self.assertRaises(ValueError):
            mesh.infer_http("key","/v1/secrets",timeout=1)
    def test_not_exporting_model_reasoning_contents(self):
        code=Path(mesh.__file__).read_text("utf8")
        self.assertNotIn('reasoning_content',code)
        self.assertNotIn('OPENAI_API_KEY',code)
        self.assertNotIn('ANTHROPIC_API_KEY',code)
if __name__=="__main__":unittest.main()
