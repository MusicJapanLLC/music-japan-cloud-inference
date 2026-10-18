import json
import tempfile
import unittest
from pathlib import Path
from cloud_probe import signed_record
import experimental_gptoss120b as model
import official120b_chat as original
import render_official120b as text

def synthetic(answer="437"):
    return signed_record({
        "schema":original.SCHEMA,
        "model_id":model.ID,
        "official_weight_sha256":model.SHA256,
        "model_original_openai_gpt_oss_120b":True,
        "actual_local_120b_inference":True,
        "model_provider_credits_used":0,
        "paid_compute_allocated":False,
        "reasoning_effort":"high","rounds":1,"elapsed_total_seconds":371.6,
        "selection":{"final_answer":answer,"source":"model_with_independent_tool_verification",
        "verification":{"status":"VERIFIED"}}
    })

class Original120BRenderer(unittest.TestCase):
    def read(self,rec):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"proof.json"
            p.write_text(json.dumps(rec),encoding="utf8")
            return text.read_verified(p)
    def test_authentic_reply(self):
        p=self.read(synthetic())
        self.assertEqual(p["model_id"],model.ID)
        self.assertIn("437",text.render(p))
    def test_attacker_html_escaped(self):
        report=text.render(synthetic("<script>alert(1)</script>"))
        self.assertIn("&lt;script&gt;",report)
        self.assertNotIn("<script>",report)
    def test_model_impersonation_blocked(self):
        rec=synthetic()
        rec["model_id"]="claude-opus-5-5"
        with self.assertRaisesRegex(ValueError,"IDENTITY"):
            self.read(signed_record(rec))
    def test_no_credit_condition_enforced(self):
        rec=synthetic()
        rec["model_provider_credits_used"]=1
        with self.assertRaisesRegex(ValueError,"NO_SPEND"):
            self.read(signed_record(rec))
    def test_sha_tamper_rejected(self):
        rec=synthetic()
        rec["selection"]["final_answer"]="I am GPT-6"
        with self.assertRaisesRegex(ValueError,"SHA"):
            self.read(rec)
    def test_model_inference_must_have_run(self):
        rec=synthetic()
        rec["actual_local_120b_inference"]=False
        with self.assertRaisesRegex(ValueError,"INFERENCE"):
            self.read(signed_record(rec))
    def test_blank_result_rejected(self):
        with self.assertRaisesRegex(ValueError,"VISIBLE_FINAL"):
            self.read(synthetic(""))
if __name__=="__main__":unittest.main()
