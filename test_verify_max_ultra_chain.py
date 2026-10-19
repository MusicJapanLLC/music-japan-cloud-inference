"""Guard that MAX and ULTRA really executed both pinned models, not UI-only labels."""
import hashlib
import unittest
from cloud_probe import signed_record,verify
from cloud_owner_chat import CFG,SCHEMA as DRAFT_SCHEMA
from official120b_chat import SCHEMA as FINAL_SCHEMA
import experimental_gptoss120b as last
from reasoning_profiles import profile
from verify_max_ultra_chain import validate

QUESTION="Calculate 314 times 29. Answer only the integer."

def pair(mode="max", answer="9106"):
    preset=profile(mode)
    draft=signed_record({
        "schema":DRAFT_SCHEMA,
        "model_id":CFG["model"],
        "model_weight_sha256":CFG["sha256"],
        "model_revision":CFG["revision"],
        "real_local_linux_model_inference":True,
        "external_model_api_calls":0,
        "paid_compute_created":False,
        "rounds":preset["draft_rounds"],
        "reasoning_effort":preset["draft_effort"],
        "token_budget_per_round":preset["draft_tokens_per_round"],
        "prompt_sha256":hashlib.sha256(QUESTION.encode("utf8")).hexdigest(),
        "rounds_evidence":[{"official_model":CFG["model"],"answer":"9106",
                            "elapsed_seconds":10.0,"output_tokens":40,"round":i+1}
                           for i in range(preset["draft_rounds"])],
        "elapsed_total_seconds":90.0,
        "selection":{"final_answer":"9106"},
    })
    final=signed_record({
        "schema":FINAL_SCHEMA,
        "model_id":last.ID,
        "official_weight_sha256":last.SHA256,
        "model_original_openai_gpt_oss_120b":True,
        "actual_local_120b_inference":True,
        "research_mode":mode,
        "draft_proposal_receipt_sha256":draft["sha256"],
        "external_open_model_proposal_applied":True,
        "public_prompt_hash":hashlib.sha256(QUESTION.encode("utf8")).hexdigest(),
        "rounds":preset["final_rounds"],
        "reasoning_effort":preset["final_effort"],
        "requested_tokens_each_round":preset["final_tokens_per_round"],
        "round_proofs":[{"http_status":200,"model_answer":answer,
                         "reported_reasoning_tokens":21,"round":i+1}
                        for i in range(preset["final_rounds"])],
        "selection":{"final_answer":answer,
                    "verification":{"status":"VERIFIED"},"source":"model_with_independent_tool_verification"},
        "model_provider_credits_used":0,
        "paid_compute_allocated":False,
        "elapsed_total_seconds":300.0,
    })
    return draft,final

class ChainTests(unittest.TestCase):
    def test_max_real_two_model_chain(self):
        a,b=pair("max")
        receipt=validate(QUESTION,"max",a,b)
        self.assertTrue(verify(receipt))
        self.assertEqual(receipt["actual_total_generation_rounds"],3)
        self.assertEqual(receipt["verified_public_model_120b"],"gpt-oss-120b-MXFP4")
        self.assertEqual(receipt["independent_question_grounded_verification"],"VERIFIED")
        self.assertEqual(receipt["model_provider_credits_used"],0)
    def test_ultra_real_five_round_chain(self):
        a,b=pair("ultra")
        receipt=validate(QUESTION,"ultra",a,b)
        self.assertEqual(receipt["actual_total_generation_rounds"],5)
        self.assertGreater(receipt["total_measured_compute_seconds"],0)
        self.assertEqual(receipt["mode"],"ultra")
    def test_reject_unlinked_proposal(self):
        a,b=pair("max")
        b["draft_proposal_receipt_sha256"]="0"*64
        b=signed_record(b)
        with self.assertRaisesRegex(ValueError,"SOURCE_SHA_MISMATCH"):
            validate(QUESTION,"max",a,b)
    def test_refuse_forged_model(self):
        a,b=pair("max")
        b["model_id"]="claude-opus-5-5"
        b=signed_record(b)
        with self.assertRaisesRegex(ValueError,"ORIGINAL_120B_IDENTITY"):
            validate(QUESTION,"max",a,b)
    def test_wrong_question_rejected(self):
        a,b=pair("max")
        with self.assertRaisesRegex(ValueError,"QUESTION_IDENTITY"):
            validate("Calculate 27 times 33.","max",a,b)
    def test_ultra_cannot_be_renamed_max(self):
        a,b=pair("max")
        with self.assertRaises(ValueError):
            validate(QUESTION,"ultra",a,b)
    def test_model_inference_must_be_true(self):
        a,b=pair("max")
        b["actual_local_120b_inference"]=False
        b=signed_record(b)
        with self.assertRaisesRegex(ValueError,"ACTUAL_120B"):
            validate(QUESTION,"max",a,b)
    def test_preset_rounds_must_match(self):
        a,b=pair("ultra")
        b["round_proofs"].pop()
        b=signed_record(b)
        with self.assertRaisesRegex(ValueError,"INFERENCE_ROUNDS"):
            validate(QUESTION,"ultra",a,b)
    def test_not_discard_corrupted_rounds(self):
        a,b=pair("ultra")
        b["round_proofs"][1]=None
        b=signed_record(b)
        with self.assertRaisesRegex(ValueError,"RESPONSE_INVALID"):
            validate(QUESTION,"ultra",a,b)
    def test_reject_20b_paid_calls(self):
        a,b=pair("max")
        a["external_model_api_calls"]=1
        a=signed_record(a)
        b["draft_proposal_receipt_sha256"]=a["sha256"]
        b=signed_record(b)
        with self.assertRaisesRegex(ValueError,"DRAFT_POLICY"):
            validate(QUESTION,"max",a,b)
    def test_20b_empty_response_refused(self):
        a,b=pair("max")
        a["rounds_evidence"][0]["answer"]=""
        a=signed_record(a)
        b["draft_proposal_receipt_sha256"]=a["sha256"]
        b=signed_record(b)
        with self.assertRaisesRegex(ValueError,"DRAFT_RESPONSES"):
            validate(QUESTION,"max",a,b)

if __name__=="__main__":
    unittest.main()
