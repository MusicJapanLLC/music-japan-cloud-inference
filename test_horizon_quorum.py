"""Synthetic quorum validation; never calls proprietary or local models."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from cloud_probe import signed_record,verify
from horizon_fanout import SCHEMA as HORIZON_SCHEMA,config
from beyond120_openweight import SCHEMA as FRONTIER_SCHEMA,MODELS,PROMPTS
import horizon_quorum as h

QUESTION=PROMPTS[0]

def horizon(profile,answer="156",q=QUESTION):
    c=config(profile)
    return signed_record({
        "schema":HORIZON_SCHEMA,
        "profile":profile,
        "model_id":c["id"],"model_repo":c["repo"],
        "model_revision":c["revision"],
        "weights_integrity_digest":c["sha256"],
        "weight_total_bytes":c["bytes"],
        "public_question_sha256":hashlib.sha256(q.encode("utf8")).hexdigest(),
        "actual_original_public_weight_inference":True,
        "proprietary_vendor_model":False,
        "model_answer":answer,"model_answer_present":True,
        "http_status":200,"github_linux_cpu_ephemeral":True,
        "paid_cloud_jobs":0,"external_llm_model_credits":0,
        "question_in_receipt":False,"generation_seconds":45.3,
    })

def frontier(profile,answer="156"):
    c=MODELS[profile]
    return signed_record({
        "schema":FRONTIER_SCHEMA,"profile":profile,"model_id":c["model_alias"],
        "model_repo":c["repo"],"revision":c["revision"],
        "quantization":c["quant"],
        "model_shards":[{"path":n,"size":s,"sha256":digest}
                        for n,s,digest in c["parts"]],
        "model_loaded":True,"real_inference":True,
        "results":[{"id":1,"http_status":200,"visible_answer":answer,
                    "actual_model_generated_text":True,
                    "completion_tokens":6,"inference_seconds":60.1}],
        "public_github_linux_cpu_only":True,"not_a_proprietary_model":True,
        "model_provider_credits":0,"paid_cloud_allocations":0,
    })

def evidence(oss="156",qwen="156",include=False):
    rows={"oss120":horizon("oss120",oss),
          "qwen122":horizon("qwen122",qwen)}
    if include:
        rows["thinking-235b-iq2m"]=frontier("thinking-235b-iq2m","156")
        rows["frontier-397b-iq1s"]=frontier("frontier-397b-iq1s","154")
    return rows

class QuorumTests(unittest.TestCase):
    def test_required_two_real_models(self):
        r=h.select(QUESTION,evidence())
        self.assertTrue(verify(r))
        self.assertEqual(r["model_workers"],2)
        self.assertEqual(r["decision"],"INDEPENDENT_OBJECTIVE_TOOL_VERIFIED")
        self.assertEqual(r["tool_verified_model_count"],2)
        self.assertFalse(r["model_weights_claimed_to_be_claude_gpt6_or_gemini_proprietary"])
    def test_four_worker_quorum_verified(self):
        r=h.select(QUESTION,evidence(include=True),require_all=True)
        self.assertEqual(r["model_workers"],4)
        self.assertEqual(r["tool_verified_model_count"],3)
        self.assertEqual(r["final_answer"],"156")
        self.assertEqual(r["selected_model"],"oss120")
        self.assertEqual(r["model_api_credits"],0)
    def test_verified_secondary_wins_wrong_primary(self):
        r=h.select(QUESTION,evidence(oss="152"))
        self.assertEqual(r["selected_model"],"qwen122")
        self.assertEqual(r["decision"],"INDEPENDENT_OBJECTIVE_TOOL_VERIFIED")
        self.assertEqual(r["final_answer"],"156")
    def test_disagreement_is_not_factual_proof(self):
        q="Explain an unknown privately held future decision."
        pair={"oss120":horizon("oss120","A",q),
              "qwen122":horizon("qwen122","B",q)}
        r=h.select(q,pair)
        self.assertEqual(r["confidence"],"requires_review")
        self.assertEqual(r["decision"],"NO_INDEPENDENT_VERIFICATION_CONFLICT")
    def test_consensus_not_mislabeled_fact(self):
        q="What is the secret reason for a private decision?"
        pair={"oss120":horizon("oss120","Unknown",q),
              "qwen122":horizon("qwen122","Unknown",q)}
        r=h.select(q,pair)
        self.assertEqual(r["confidence"],"model_consensus_only")
    def test_missing_real_model_rejected(self):
        with self.assertRaisesRegex(ValueError,"MISSING_REQUIRED"):
            h.select(QUESTION,{"oss120":horizon("oss120")})
    def test_tampered_genuine_receipt_rejected(self):
        pair=evidence()
        pair["oss120"]["model_answer"]="spoofed"
        with self.assertRaisesRegex(ValueError,"HASH_INVALID"):
            h.select(QUESTION,pair)
    def test_fake_original_closed_model_rejected(self):
        pair=evidence()
        pair["oss120"]["model_id"]="claude-fable-5-1"
        pair["oss120"]=signed_record(pair["oss120"])
        with self.assertRaisesRegex(ValueError,"MODEL_ID"):
            h.select(QUESTION,pair)
    def test_different_question_rejected(self):
        with self.assertRaisesRegex(ValueError,"QUESTION"):
            h.select("Calculate 314 times 29. Answer only the number.",evidence())
    def test_deceptive_397b_only_source_rejected(self):
        items=evidence(include=True)
        items["frontier-397b-iq1s"]["model_shards"][0]["sha256"]="0"*64
        items["frontier-397b-iq1s"]=signed_record(items["frontier-397b-iq1s"])
        with self.assertRaisesRegex(ValueError,"SHARD"):
            h.select(QUESTION,items)
    def test_397b_cannot_claim_inference_without_answer(self):
        items=evidence(include=True)
        items["frontier-397b-iq1s"]["results"][0]["actual_model_generated_text"]=False
        items["frontier-397b-iq1s"]=signed_record(items["frontier-397b-iq1s"])
        with self.assertRaisesRegex(ValueError,"INFERENCE"):
            h.select(QUESTION,items)
    def test_four_required_exact(self):
        with self.assertRaisesRegex(ValueError,"ALL_FOUR"):
            h.select(QUESTION,evidence(),require_all=True)
    def test_no_token_private_exposure(self):
        r=h.select(QUESTION,evidence(include=True))
        self.assertTrue(r["question_in_receipt"] is False)
        self.assertNotIn(QUESTION,json.dumps(r))
        self.assertFalse(r["dedicated_gpu_allocated"])
        self.assertEqual(r["additional_spend_usd"],0)
    def test_cli_rejects_no_files(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as d:
            p=subprocess.run([sys.executable,h.__file__,
                "--oss120",str(Path(d)/"none1"),
                "--qwen122",str(Path(d)/"none2"),
                "--question",QUESTION,
                "--out",str(Path(d)/"out.json")],
                capture_output=True,text=True,timeout=8)
            self.assertEqual(p.returncode,2)
            self.assertFalse((Path(d)/"out.json").exists())

if __name__=="__main__":
    unittest.main()
