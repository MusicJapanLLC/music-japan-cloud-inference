import json
import tempfile
import unittest
from pathlib import Path
from cloud_probe import signed_record
from cloud_owner_chat import CFG,SCHEMA
import render_cloud_chat as reader

def proof(answer="hello"):
    return signed_record({
        "schema":SCHEMA,"model_id":CFG["model"],
        "model_weight_sha256":CFG["sha256"],
        "external_model_api_calls":0,"paid_compute_created":False,
        "real_local_linux_model_inference":True,"reasoning_effort":"high",
        "rounds":3,"elapsed_total_seconds":67.1,
        "rounds_evidence":[{"round":1},{"round":2},{"round":3}],
        "selection":{"final_answer":answer,"verification":{"status":"ABSTAIN"}}
    })

class SummaryTests(unittest.TestCase):
    def read(self,item):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/"proof.json"
            target.write_text(json.dumps(item))
            return reader.read_and_verify(target)
    def test_valid_proof(self):
        self.assertEqual(self.read(proof())["rounds"],3)
    def test_no_untrusted_html(self):
        result=reader.render(proof("<script>alert(1)</script>"))
        self.assertIn("&lt;script&gt;",result)
        self.assertNotIn("<script>",result)
    def test_model_identity_present(self):
        result=reader.render(proof("PONG"))
        self.assertIn("gpt-oss-20b",result)
        self.assertIn("PONG",result)
    def test_reject_modified_signature(self):
        rec=proof()
        rec["selection"]["final_answer"]="altered"
        with self.assertRaisesRegex(ValueError,"INTEGRITY"):
            self.read(rec)
    def test_reject_fake_frontier_model(self):
        rec=proof()
        rec["model_id"]="claude-opus-5-5"
        with self.assertRaisesRegex(ValueError,"MODEL_IDENTITY"):
            self.read(signed_record(rec))
    def test_reject_provider_api_usage(self):
        rec=proof()
        rec["external_model_api_calls"]=1
        with self.assertRaisesRegex(ValueError,"NONLOCAL"):
            self.read(signed_record(rec))
    def test_reject_empty_reply(self):
        with self.assertRaisesRegex(ValueError,"FINAL_TEXT"):
            self.read(proof(""))
    def test_reject_oversize(self):
        with self.assertRaisesRegex(ValueError,"TOO_LONG"):
            reader.render(proof("a"*30001))

if __name__=="__main__":unittest.main()
