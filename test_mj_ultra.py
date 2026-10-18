"""Ultra remote cloud launcher: synthetic tests, NO actual model-credit request."""
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import mj_ultra as u
from test_verify_max_ultra_chain import QUESTION, pair
from verify_max_ultra_chain import validate
from cloud_probe import signed_record

URL="https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/39098211563"
RUN=39098211563

def fixtures(mode="max"):
    a,b=pair(mode)
    c=validate(QUESTION,mode,a,b)
    return {"draft":a,"final":b,"chain":c}

def result_for(id=RUN,state="completed",conclusion="success"):
    return json.dumps({"databaseId":id,"status":state,"conclusion":conclusion,"url":URL,
      "name":"Music Japan MAX ULTRA — verified open 20B + 120B reasoning",
      "headBranch":u.REF,"event":"workflow_dispatch"})

class UltraControlTests(unittest.TestCase):
    def test_public_safety_gate(self):
        self.assertEqual(u.public_question("  Calculate 314 times 29.  "),"Calculate 314 times 29.")
        for value in (None,"","ab","x"*3400,"a\0b\nc",
                      "Here is my key sk-proj-aaaaaaaaaaaaaaaaaaaaaaaa",
                      "ghp_aabbccddeeffgghhiijjkkllmmnn",
                      "-----BEGIN PRIVATE KEY-----\nabc"):
            with self.assertRaises(ValueError):
                u.public_question(value)
    def test_mode_validation(self):
        with self.assertRaises(ValueError):
            u.dispatch("private","Hello there.",invoke=lambda *args,**kwargs: "noop")
    def test_dispatch_exposes_only_intended_official_workflow(self):
        captured=[]
        def gh(args,timeout):
            captured.append((args,timeout))
            return "Triggered workflow\n"+URL
        obj=u.dispatch("max",QUESTION,invoke=gh)
        self.assertEqual(obj["run_id"],RUN)
        self.assertEqual(obj["mode"],"max")
        cmd=captured[0][0]
        self.assertEqual(cmd[:5],["workflow","run",u.WORKFLOW,"--repo",u.REPO])
        self.assertEqual(cmd[cmd.index("--ref")+1],u.REF)
        self.assertIn("execute=true",cmd)
        self.assertIn("mode=max",cmd)
        self.assertEqual(sum(1 for x in cmd if x.startswith("public_question=")),1)
    def test_concurrent_workflow_ambiguity_not_guessed(self):
        with self.assertRaisesRegex(RuntimeError,"WITHOUT_CONFIRMED_RUN_URL"):
            u.dispatch("max",QUESTION,invoke=lambda *args,**kwargs:"workflow dispatched")
    def test_run_id_only_numeric(self):
        self.assertEqual(u.run_id(str(RUN)),RUN)
        for item in (0,-5,True,"notanid","9999","12345abc","1"*17):
            with self.assertRaises(ValueError):u.run_id(item)
    def test_gh_command_no_shell_and_strips_error_output(self):
        captured=[]
        class Outcome:
            returncode=0
            stdout="  ok\n"
            stderr="private stuff"
        def fake(cmd,**opts):
            captured.append((cmd,opts))
            return Outcome()
        self.assertEqual(u.gh(["run","view",str(RUN)],invoke=fake),"ok")
        self.assertEqual(captured[0][0][0],"gh")
        self.assertTrue(captured[0][1]["capture_output"])
        self.assertNotIn("shell",captured[0][1])
        self.assertEqual(captured[0][1]["encoding"],"utf-8")
        self.assertEqual(captured[0][1]["errors"],"replace")
    def test_gh_error_redacted(self):
        class Outcome:
            returncode=2
            stdout=""
            stderr="SECRET FROM USER"
        with self.assertRaisesRegex(RuntimeError,"GITHUB_CLI_EXIT_2") as ctx:
            u.gh(["run","view",str(RUN)],invoke=lambda *args,**kw:Outcome())
        self.assertNotIn("SECRET FROM USER",str(ctx.exception))
    def test_status_real_sha_pinned_branch(self):
        r=u.status(RUN,invoke=lambda *args,**kwargs:result_for())
        self.assertEqual(r["status"],"completed")
        self.assertEqual(r["conclusion"],"success")
    def test_status_wrong_branch_rejected(self):
        row=json.loads(result_for())
        row["headBranch"]="evil-controlled"
        with self.assertRaisesRegex(ValueError,"BRANCH_OR_EVENT"):
            u.status(RUN,invoke=lambda *args,**kwargs:json.dumps(row))
    def test_status_wrong_run_refused(self):
        with self.assertRaisesRegex(ValueError,"RUN_ID_MISMATCH"):
            u.status(RUN,invoke=lambda *args,**kwargs:result_for(123456789))
    def test_watch_complete_success(self):
        r=u.watch(RUN,seconds=60,invoke=lambda *args,**kwargs:result_for(),
                 clock=lambda:0,sleep=lambda x:None)
        self.assertEqual(r["conclusion"],"success")
    def test_watch_failed_run_errors(self):
        with self.assertRaisesRegex(RuntimeError,"FINISHED_FAILURE"):
            u.watch(RUN,seconds=60,invoke=lambda *args,**kwargs:result_for(conclusion="failure"),
                    clock=lambda:0,sleep=lambda x:None)
    def test_watch_never_ignores_budget(self):
        with self.assertRaises(ValueError):u.watch(RUN,seconds=3)
        with self.assertRaises(ValueError):u.watch(RUN,seconds=20000)
    def test_actual_two_model_proof(self):
        real=u.prove("max",QUESTION,fixtures())
        self.assertTrue(real["verified"])
        self.assertEqual(real["rounds"],3)
        self.assertEqual(real["real_model_api_credits_used"],0)
        self.assertEqual(real["new_spend_usd"],0)
        self.assertEqual(real["models"],["gpt-oss-20b-MXFP4","gpt-oss-120b-MXFP4"])
        self.assertEqual(real["response"],"9106")
    def test_ultra_really_five_model_rounds(self):
        r=u.prove("ultra",QUESTION,fixtures("ultra"))
        self.assertEqual(r["mode"],"ultra")
        self.assertEqual(r["rounds"],5)
    def test_no_unauthenticated_fake_final(self):
        f=fixtures()
        f["final"]["actual_local_120b_inference"]=False
        f["final"]=signed_record(f["final"])
        with self.assertRaisesRegex(ValueError,"ACTUAL_120B_INFERENCE_NOT_VERIFIED"):
            u.prove("max",QUESTION,f)
    def test_tampered_chain_rejected(self):
        f=fixtures()
        f["chain"]["final_answer"]="spoofed"
        with self.assertRaisesRegex(ValueError,"HASH_FAILED_CHAIN"):
            u.prove("max",QUESTION,f)
    def test_wrong_original_question_fails_closed(self):
        with self.assertRaises(ValueError):
            u.prove("max","Calculate 33 times 44. Answer only the integer.",fixtures())
    def test_collect_existing_run_works(self):
        with patch.object(u,"status",return_value={"status":"completed","conclusion":"success","url":URL}),\
             patch.object(u,"download_artifacts",return_value=fixtures()):
            result=u.collect(RUN,"max",QUESTION)
        self.assertEqual(result["run_id"],RUN)
        self.assertEqual(result["run_url"],URL)
        self.assertEqual(result["proof_sha256"],validate(QUESTION,"max",*pair("max"))["sha256"])
    def test_partial_job_cannot_return_success(self):
        with patch.object(u,"status",return_value={"status":"in_progress","conclusion":None,"url":URL}):
            with self.assertRaisesRegex(ValueError,"RUN_NOT_VERIFIED"):
                u.collect(RUN,"max",QUESTION)
    def test_requires_public_flag_to_start(self):
        with self.assertRaises(SystemExit) as ctx:
            u.cli(["start","--mode","max","--public-question",QUESTION])
        self.assertEqual(ctx.exception.code,2)

if __name__=="__main__":
    unittest.main()
