"""No-network unit tests for real-model adaptive quality control."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cloud_probe import signed_record,verify
from test_verify_max_ultra_chain import pair,QUESTION
import adaptive_supervisor as sup

class AdaptiveSupervisorTests(unittest.TestCase):
    def test_easy_arithmetic_preflight_uses_max_profile(self):
        policy=sup.plan_question(QUESTION)
        self.assertEqual(policy["preset"],"max")
        self.assertTrue(verify(policy))
        self.assertEqual(policy["20b_rounds"],2)
        self.assertEqual(policy["120b_rounds"],1)
    def test_complex_topic_chooses_ultra(self):
        question="Design a robust architecture for a meeting notes assistant with speaker recognition, permissions, and audits."
        policy=sup.plan_question(question)
        self.assertEqual(policy["preset"],"ultra")
        self.assertEqual(policy["120b_rounds"],2)
    def test_explicit_ultra_always_chooses_ultra(self):
        result=sup.plan_question(QUESTION,"force-ultra")
        self.assertEqual(result["preset"],"ultra")
        self.assertEqual(result["20b_rounds"],3)
    def test_real_draft_recipe_validation(self):
        draft,_=pair("max")
        p=sup.plan_question(QUESTION)
        self.assertEqual(sup.validate_draft(QUESTION,p,draft),"9106")
    def test_adaptive_early_exit_when_independently_verified(self):
        draft,_=pair("max")
        p=sup.plan_question(QUESTION)
        d=sup.decide(QUESTION,p,draft)
        self.assertFalse(d["escalate_to_120b"])
        answer=sup.finish(QUESTION,p,draft,d)
        self.assertTrue(verify(answer))
        self.assertEqual(answer["answer"],"9106")
        self.assertEqual(answer["models"],["gpt-oss-20b-MXFP4"])
        self.assertEqual(answer["verification_status"],"VERIFIED")
        self.assertEqual(answer["actual_model_rounds"],2)
    def test_tool_correction_of_wrong_raw_20b_triggers_escalation(self):
        draft,_=pair("max")
        draft["rounds_evidence"][-1]["answer"]="9150"
        draft=signed_record(draft)
        p=sup.plan_question(QUESTION)
        decision=sup.decide(QUESTION,p,draft)
        self.assertTrue(decision["escalate_to_120b"])
        self.assertEqual(decision["last_raw_20b_verification"],"REJECTED")
        self.assertEqual(decision["draft_verification"],"VERIFIED")

    def test_exhausted_model_thinking_budget_escalates(self):
        draft,_=pair("max")
        p=sup.plan_question(QUESTION)
        draft["rounds_evidence"][0]["output_tokens"]=p["20b_tokens"]
        draft=signed_record(draft)
        decision=sup.decide(QUESTION,p,draft)
        self.assertTrue(decision["escalate_to_120b"])
        self.assertTrue(decision["truncated_output_detected"])

    def test_force_ultra_always_calls_real_high_model(self):
        draft,final=pair("ultra")
        p=sup.plan_question(QUESTION,"force-ultra")
        d=sup.decide(QUESTION,p,draft)
        self.assertTrue(d["escalate_to_120b"])
        result=sup.finish(QUESTION,p,draft,d,final)
        self.assertEqual(result["models"],["gpt-oss-20b-MXFP4","gpt-oss-120b-MXFP4"])
        self.assertEqual(result["actual_model_rounds"],5)
        self.assertEqual(result["answer_source"],"REAL_120B_TOOL_VERIFIED")
        self.assertEqual(result["api_model_credits"],0)
        self.assertIsInstance(result["cross_model_chain_sha256"],str)
    def test_valid_20b_beats_incorrect_120b_math(self):
        draft,final=pair("ultra",answer="9150")
        p=sup.plan_question(QUESTION,"force-ultra")
        d=sup.decide(QUESTION,p,draft)
        report=sup.finish(QUESTION,p,draft,d,final)
        self.assertEqual(report["answer"],"9106")
        self.assertEqual(report["answer_source"],"REAL_20B_TOOL_VERIFIED")
    def test_unsupported_general_question_uses_120b(self):
        q="Explain a robust architecture for secure multilingual meeting summaries."
        p=sup.plan_question(q)
        draft,final=pair("ultra",answer="A detailed outline")
        draft["prompt_sha256"]=hashlib.sha256(q.encode("utf8")).hexdigest()
        draft=signed_record(draft)
        final["public_prompt_hash"]=hashlib.sha256(q.encode("utf8")).hexdigest()
        final["draft_proposal_receipt_sha256"]=draft["sha256"]
        final=signed_record(final)
        d=sup.decide(q,p,draft)
        self.assertTrue(d["escalate_to_120b"])
        report=sup.finish(q,p,draft,d,final)
        self.assertEqual(report["verification_status"],"ABSTAIN")
        self.assertEqual(report["answer_source"],"120B_RESULT_WITH_UNVERIFIED_CLAIMS")
    def test_wrong_120b_model_rejected(self):
        draft,final=pair("ultra")
        p=sup.plan_question(QUESTION,"force-ultra")
        d=sup.decide(QUESTION,p,draft)
        final["model_id"]="fake-120b"
        final=signed_record(final)
        with self.assertRaises(ValueError):sup.finish(QUESTION,p,draft,d,final)
    def test_wrong_parent_sha_rejected(self):
        draft,final=pair("ultra")
        p=sup.plan_question(QUESTION,"force-ultra")
        d=sup.decide(QUESTION,p,draft)
        final["draft_proposal_receipt_sha256"]="0"*64
        final=signed_record(final)
        with self.assertRaises(ValueError):sup.finish(QUESTION,p,draft,d,final)
    def test_mutated_policy_rejected(self):
        draft,_=pair("max")
        p=sup.plan_question(QUESTION)
        p["120b_tokens"]=2000
        p=signed_record(p)
        with self.assertRaisesRegex(ValueError,"POLICY_TAMPER"):
            sup.decide(QUESTION,p,draft)
    def test_escalation_change_rejected(self):
        draft,_=pair("max")
        p=sup.plan_question(QUESTION)
        d=sup.decide(QUESTION,p,draft)
        d["escalate_to_120b"]=True
        d=signed_record(d)
        with self.assertRaisesRegex(ValueError,"DECISION_REPLAY"):
            sup.finish(QUESTION,p,draft,d)
    def test_fake_120b_proof_for_early_exit_is_rejected(self):
        draft,final=pair("max")
        p=sup.plan_question(QUESTION)
        d=sup.decide(QUESTION,p,draft)
        with self.assertRaisesRegex(ValueError,"EXTRA_120B"):
            sup.finish(QUESTION,p,draft,d,final)
    def test_not_certify_model_agreement_as_factual_proof(self):
        q="What is the exact population of Mars?"
        self.assertEqual(sup.check_answer(q,"1 million")["status"],"ABSTAIN")
    def test_summary_html_escape(self):
        draft,final=pair("ultra")
        p=sup.plan_question(QUESTION,"force-ultra")
        d=sup.decide(QUESTION,p,draft)
        result=sup.finish(QUESTION,p,draft,d,final)
        result["answer"]="<script>alert(1)</script>"
        result=signed_record(result)
        safe=sup.safe_summary(result)
        self.assertIn("&lt;script&gt;",safe)
        self.assertNotIn("<script>",safe)
    def test_cli_offline_creates_signed_plan(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/"plan.json"
            env=dict(os.environ)
            env["MJ_PUBLIC_RESEARCH_QUESTION"]=QUESTION
            x=subprocess.run([sys.executable,str(Path(sup.__file__).resolve()),"plan","--out",str(out)],
                             capture_output=True,text=True,env=env,timeout=8)
            self.assertEqual(x.returncode,0,x.stderr)
            self.assertTrue(verify(json.loads(out.read_text("utf8"))))
    def test_invalid_modes_rejected(self):
        with self.assertRaises(ValueError):sup.plan_question(QUESTION,"use-paid-provider")
    def test_translation_equivalence(self):
        q="Translate Good morning into Japanese. Only the translation."
        self.assertEqual(sup.check_answer(q,"おはようございます")["status"],"VERIFIED")

if __name__=="__main__":unittest.main()
