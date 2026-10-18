import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from cloud_probe import signed_record,verify
from horizon_fanout import SCHEMA,PROFILES,config,inference_request,DEFAULT_QUESTION
import horizon_referee as council

QUESTION="Compute ((72 / 8) + 17) times 6. Reply only with the integer."

def make_receipt(profile:str,answer:str="156",question:str=QUESTION)->dict:
    cfg=config(profile)
    return signed_record({
        "schema":SCHEMA,"profile":profile,"model_id":cfg["id"],
        "model_repo":cfg["repo"],"model_revision":cfg["revision"],
        "weights_integrity_digest":cfg["sha256"],"weight_total_bytes":cfg["bytes"],
        "public_question_sha256":hashlib.sha256(question.encode("utf8")).hexdigest(),
        "actual_original_public_weight_inference":True,
        "proprietary_vendor_model":False,
        "reasoning_effort":"high" if profile=="oss120" else "format_only_no_thinking",
        "model_answer":answer,"model_answer_present":True,
        "model_generated_tokens":20,"http_status":200,
        "model_load_seconds":180.0,"generation_seconds":90.0,"total_seconds":270.0,
        "github_linux_cpu_ephemeral":True,
        "paid_cloud_jobs":0,"external_llm_model_credits":0,
        "question_in_receipt":False,
    })

class HorizonTests(unittest.TestCase):
    def test_genuinely_original_large_models(self):
        self.assertEqual(PROFILES,("oss120","qwen122"))
        self.assertEqual(config("oss120")["id"],"gpt-oss-120b-MXFP4")
        self.assertEqual(config("qwen122")["id"],"Qwen3.5-122B-A10B-IQ4_XS")
        self.assertGreater(config("qwen122")["bytes"],60_000_000_000)

    def test_model_high_effort_assigned_only_when_supported(self):
        openai=inference_request("oss120",QUESTION)
        self.assertEqual(openai["reasoning_effort"],"high")
        self.assertEqual(openai["max_tokens"],512)
        self.assertEqual(openai["chat_template_kwargs"]["reasoning_effort"],"high")
        qwen=inference_request("qwen122",QUESTION)
        self.assertNotIn("reasoning_effort",qwen)
        self.assertEqual(qwen["chat_template_kwargs"]["enable_thinking"],False)
        self.assertEqual(qwen["max_tokens"],256)

    def test_disallowed_profiles_and_credential_strings(self):
        with self.assertRaises(ValueError):
            config("fake-original-claude")
        with self.assertRaises(ValueError):
            inference_request("oss120","hello sk-ant-notarealtoken01234567890123456789")

    def test_actual_model_requires_sha_and_exact_question(self):
        r=make_receipt("oss120")
        self.assertEqual(council.verified_receipt(r,"oss120",QUESTION),r)
        with self.assertRaises(ValueError):
            council.verified_receipt(r,"oss120","Calculate 314 times 29. Reply with integer only.")
        r["model_answer"]="changed"
        with self.assertRaises(ValueError):
            council.verified_receipt(r,"oss120",QUESTION)

    def test_math_cross_model_selects_independently_correct(self):
        report=council.decide(QUESTION,{"oss120":make_receipt("oss120","156"),
                                        "qwen122":make_receipt("qwen122","154")})
        self.assertEqual(report["final_answer"],"156")
        self.assertEqual(report["decision"],"INDEPENDENT_TOOL_VERIFIED")
        self.assertEqual(report["tool_verified_model_count"],1)
        self.assertTrue(verify(report))

    def test_tool_checked_122b_wins_when_120b_wrong(self):
        report=council.decide(QUESTION,{"oss120":make_receipt("oss120","154"),
                                        "qwen122":make_receipt("qwen122","156")})
        self.assertEqual(report["selected_model"],"qwen122")
        self.assertEqual(report["final_answer"],"156")

    def test_unseen_general_knowledge_unverified_consensus(self):
        q="Who wrote the earliest known version of this obscure song?"
        data={"oss120":make_receipt("oss120","Unknown",q),
              "qwen122":make_receipt("qwen122","Unknown",q)}
        report=council.decide(q,data)
        self.assertEqual(report["decision"],"UNVERIFIED_CONSENSUS")
        self.assertEqual(report["final_answer"],"Unknown")
        self.assertEqual(report["tool_verified_model_count"],0)

    def test_unseen_disagreement_cannot_be_marked_verified(self):
        q="Explain an unusual architectural decision in the 1990s."
        data={"oss120":make_receipt("oss120","Choice A",q),
              "qwen122":make_receipt("qwen122","Choice B",q)}
        report=council.decide(q,data)
        self.assertEqual(report["decision"],"UNRESOLVED_MODEL_DISAGREEMENT")
        self.assertFalse(report["global_shared_ram_gib_claimed"])

    def test_wrong_model_origin_fails(self):
        r=make_receipt("qwen122")
        r["model_id"]="claude-fable-5-1"
        r=signed_record(r)
        with self.assertRaises(ValueError):
            council.verified_receipt(r,"qwen122",QUESTION)

    def test_credit_usage_or_nonfree_job_refused(self):
        a=make_receipt("oss120")
        a["paid_cloud_jobs"]=1
        a=signed_record(a)
        with self.assertRaises(ValueError):
            council.verified_receipt(a,"oss120",QUESTION)

    def test_both_receipts_required(self):
        with self.assertRaises(ValueError):
            council.decide(QUESTION,{"oss120":make_receipt("oss120")})

    def test_exact_question_not_in_proof(self):
        r=council.decide(QUESTION,{"oss120":make_receipt("oss120"),
                                   "qwen122":make_receipt("qwen122")})
        self.assertNotIn(QUESTION,json.dumps(r))

    def test_worker_offline_no_cloud_or_model_calls(self):
        import horizon_fanout as worker
        with tempfile.TemporaryDirectory() as t:
            p=subprocess.run([sys.executable,worker.__file__,
                "--profile","oss120","--binary","unused","--weights-dir",t,
                "--out",str(Path(t)/"answer.json")],capture_output=True,text=True,timeout=8)
            self.assertEqual(p.returncode,0)
            self.assertIn("DRY_RUN",p.stdout)
            self.assertFalse((Path(t)/"answer.json").exists())

if __name__=="__main__":
    unittest.main()
