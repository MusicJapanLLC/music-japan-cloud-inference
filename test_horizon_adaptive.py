import json
import tempfile
import unittest
from pathlib import Path
from cloud_probe import signed_record,verify
from test_horizon_quorum import QUESTION,horizon,frontier
import horizon_adaptive as m

class AdaptiveTests(unittest.TestCase):
    def test_actual_397b_alone_is_real_not_consensus(self):
        r=m.select_available(QUESTION,{"frontier-397b-iq1s":frontier("frontier-397b-iq1s")})
        self.assertTrue(verify(r))
        self.assertEqual(r["actual_completed_worker_count"],1)
        self.assertEqual(r["confidence"],"independently_verified")
        self.assertEqual(r["selected_profile"],"frontier-397b-iq1s")
        self.assertEqual(r["model_final_answer"],"156")
        self.assertEqual(r["vendor_api_inference_credits"],0)
    def test_397b_does_not_suppress_missing_235b(self):
        data={"oss120":horizon("oss120"),"frontier-397b-iq1s":frontier("frontier-397b-iq1s")}
        a=m.select_available(QUESTION,data,minimum_workers=2)
        self.assertEqual(a["actual_completed_worker_count"],2)
        self.assertEqual(a["decision"],"OBJECTIVE_VERIFICATION_PASSED")
    def test_real_four_workers_once_all_proven(self):
        data={"oss120":horizon("oss120"),"qwen122":horizon("qwen122"),
              "thinking-235b-iq2m":frontier("thinking-235b-iq2m"),
              "frontier-397b-iq1s":frontier("frontier-397b-iq1s")}
        r=m.select_available(QUESTION,data,minimum_workers=4)
        self.assertEqual(r["actual_completed_worker_count"],4)
        self.assertEqual(len(r["per_model_evidence"]),4)
        self.assertEqual(r["confidence"],"independently_verified")
    def test_limited_workers_does_not_forge_four(self):
        with self.assertRaisesRegex(ValueError,"MINIMUM"):
            m.select_available(QUESTION,{"frontier-397b-iq1s":frontier("frontier-397b-iq1s")},minimum_workers=4)
    def test_bad_provenance_rejected(self):
        v=frontier("frontier-397b-iq1s")
        v["model_repo"]="fake/private-model"
        v=signed_record(v)
        with self.assertRaisesRegex(ValueError,"MODEL_REPO"):
            m.select_available(QUESTION,{"frontier-397b-iq1s":v})
    def test_wrong_question_rejected(self):
        with self.assertRaisesRegex(ValueError,"FRONTIER_FIXED_QUESTION"):
            m.select_available("What is the user private key?",{"frontier-397b-iq1s":frontier("frontier-397b-iq1s")})
    def test_disagreement_marks_uncertainty(self):
        q="Which private outcome will happen in the future?"
        r=m.select_available(q,{"oss120":horizon("oss120","A",q),
                                 "qwen122":horizon("qwen122","B",q)},minimum_workers=2)
        self.assertEqual(r["confidence"],"requires_external_verification")
        self.assertEqual(r["decision"],"SINGLE_MODEL_OR_DISAGREEMENT")
    def test_unverified_majority_not_independent(self):
        q="Why will this unannounced private decision occur?"
        r=m.select_available(q,{"oss120":horizon("oss120","Guess",q),
                                 "qwen122":horizon("qwen122","Guess",q)})
        self.assertEqual(r["confidence"],"consensus_only")
    def test_reject_fake_gpt6_identity(self):
        v=horizon("oss120")
        v["model_id"]="gpt6-astra-proprietary"
        v=signed_record(v)
        with self.assertRaisesRegex(ValueError,"MODEL_ID"):
            m.select_available(QUESTION,{"oss120":v})
    def test_reject_zero_model_proofs(self):
        with self.assertRaisesRegex(ValueError,"NO_REAL"):
            m.select_available(QUESTION,{})
    def test_tampered_sha_rejected(self):
        v=frontier("frontier-397b-iq1s")
        v["results"][0]["visible_answer"]="0000"
        with self.assertRaisesRegex(ValueError,"RECORD_HASH"):
            m.select_available(QUESTION,{"frontier-397b-iq1s":v})
    def test_question_is_not_exported(self):
        r=m.select_available(QUESTION,{"oss120":horizon("oss120")})
        self.assertNotIn(QUESTION,json.dumps(r))
        self.assertFalse(r["question_text_exported"])
    def test_receipts_are_no_io_selector(self):
        import inspect
        code=inspect.getsource(m.select_available)
        self.assertNotIn("requests.",code)
        self.assertNotIn("open(",code)

if __name__=="__main__":
    unittest.main()
