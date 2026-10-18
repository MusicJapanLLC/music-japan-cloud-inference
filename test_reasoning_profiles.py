"""Max/Ultra must correspond to actual extra inference and verified cross-model input."""
import hashlib
import unittest
from cloud_probe import signed_record
from cloud_owner_chat import CFG, SCHEMA as CHAT_SCHEMA
from reasoning_profiles import PRESETS, profile, validate_proposal

QUESTION="Calculate 314 times 29. Answer only the integer."

def signed_draft(mode="max",answer="9106",question=QUESTION):
    p=profile(mode)
    return signed_record({
        "schema":CHAT_SCHEMA,
        "model_id":CFG["model"],
        "model_weight_sha256":CFG["sha256"],
        "model_revision":CFG["revision"],
        "real_local_linux_model_inference":True,
        "external_model_api_calls":0,
        "paid_compute_created":False,
        "rounds":p["draft_rounds"],
        "reasoning_effort":p["draft_effort"],
        "token_budget_per_round":p["draft_tokens_per_round"],
        "prompt_sha256":hashlib.sha256(question.encode("utf8")).hexdigest(),
        "selection":{"final_answer":answer},
    })

class ReasoningProfilesTests(unittest.TestCase):
    def test_stronger_mode_is_more_compute_not_just_a_label(self):
        mx=profile("max")
        ultra=profile("ultra")
        self.assertGreater(ultra["draft_rounds"],mx["draft_rounds"])
        self.assertGreater(ultra["final_rounds"],mx["final_rounds"])
        self.assertGreater(ultra["draft_rounds"]*ultra["draft_tokens_per_round"]+
                           ultra["final_rounds"]*ultra["final_tokens_per_round"],
                           mx["draft_rounds"]*mx["draft_tokens_per_round"]+
                           mx["final_rounds"]*mx["final_tokens_per_round"])
        self.assertEqual(ultra["final_effort"],"high")
    def test_both_models_genuine_official_open_weights(self):
        for p in ("max","ultra"):
            x=profile(p)
            self.assertEqual(x["draft_model"],"openai-20b")
            self.assertEqual(x["final_model"],"openai-gpt-oss-120b")
            self.assertEqual(x["real_model_api_credits"],0)
            self.assertEqual(x["new_spend_usd"],0)
    def test_bad_preset_rejected(self):
        for x in ("astra-ultra","claude-opus5.5", "automatic", ""):
            with self.assertRaises(ValueError):
                profile(x)
    def test_verified_20b_proposal(self):
        for mode in ("max","ultra"):
            receipt=signed_draft(mode)
            self.assertEqual(validate_proposal(receipt,QUESTION,mode),"9106")
    def test_different_question_digest_rejected(self):
        x=signed_draft("max")
        with self.assertRaisesRegex(ValueError,"QUESTION_IDENTITY"):
            validate_proposal(x,"Calculate 77 times 99.", "max")
    def test_changed_model_name_rejected_even_with_new_receipt_hash(self):
        x=signed_draft("max")
        x["model_id"]="gpt-6-astra-ultra"
        x=signed_record(x)
        with self.assertRaisesRegex(ValueError,"WRONG_MODEL"):
            validate_proposal(x,QUESTION,"max")
    def test_changed_checkpoint_sha_rejected(self):
        x=signed_draft()
        x["model_weight_sha256"]="0"*64
        x=signed_record(x)
        with self.assertRaisesRegex(ValueError,"WRONG_MODEL"):
            validate_proposal(x,QUESTION,"max")
    def test_tamper_rejected(self):
        x=signed_draft()
        x["selection"]["final_answer"]="tampered"
        with self.assertRaisesRegex(ValueError,"UNVERIFIED_DRAFT_RECEIPT"):
            validate_proposal(x,QUESTION,"max")
    def test_preset_mismatch_rejected(self):
        x=signed_draft("max")
        with self.assertRaisesRegex(ValueError,"DRAFT_POLICY_MISMATCH"):
            validate_proposal(x,QUESTION,"ultra")
    def test_token_budget_mismatch_rejected(self):
        x=signed_draft()
        x["token_budget_per_round"]=5
        x=signed_record(x)
        with self.assertRaisesRegex(ValueError,"THINKING_BUDGET"):
            validate_proposal(x,QUESTION,"max")
    def test_paid_inference_or_not_real_blocked(self):
        for key,value in (("external_model_api_calls",1),
                          ("paid_compute_created",True),
                          ("real_local_linux_model_inference",False)):
            x=signed_draft()
            x[key]=value
            x=signed_record(x)
            with self.assertRaises(ValueError):
                validate_proposal(x,QUESTION,"max")
    def test_no_answer_no_forged_proposal(self):
        x=signed_draft(answer="")
        with self.assertRaisesRegex(ValueError,"NO_FINAL"):
            validate_proposal(x,QUESTION,"max")
    def test_long_answer_truncated_after_verification(self):
        x=signed_draft(answer="x"*2000)
        y=validate_proposal(x,QUESTION,"max")
        self.assertEqual(len(y),1600)
    def test_too_long_answer_not_accepted(self):
        x=signed_draft(answer="x"*30001)
        with self.assertRaisesRegex(ValueError,"TOO_LONG"):
            validate_proposal(x,QUESTION,"max")

if __name__=="__main__":
    unittest.main()
