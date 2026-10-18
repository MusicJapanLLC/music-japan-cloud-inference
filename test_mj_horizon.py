import json
import subprocess
import sys
import unittest
from pathlib import Path
from cloud_probe import signed_record
from horizon_referee import decide
from test_horizon_federation import QUESTION, make_receipt
import mj_horizon as cli

RUN=39104472325

def fake_invoke_factory(tamper=False):
    x={"oss120":make_receipt("oss120","156"),
       "qwen122":make_receipt("qwen122","154")}
    x["final"]=decide(QUESTION,x)
    if tamper:
        x["final"]["final_answer"]="false claim"
    def fake(command,timeout=45):
        if command[:2]==["workflow","run"]:
            return "https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/"+str(RUN)
        if command[:2]==["run","view"]:
            return json.dumps({
                "databaseId":RUN,"status":"completed","conclusion":"success",
                "url":"https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/"+str(RUN),
                "name":"Music Japan Horizon — Genuine 120B x 122B Cloud Linux Federation",
                "headBranch":cli.REF,"event":"workflow_dispatch"})
        if command[:2]==["run","download"]:
            idx=command.index("--name")+1
            artifact=command[idx]
            dest=Path(command[command.index("--dir")+1])
            for kind,(name,file) in cli.ARTIFACTS.items():
                if name==artifact:
                    dest.mkdir(exist_ok=True,parents=True)
                    (dest/file).write_text(json.dumps(x[kind]),encoding="utf8")
                    return ""
            raise AssertionError("unexpected artifact")
        raise AssertionError("unexpected subprocess call")
    return fake

class ClientTests(unittest.TestCase):
    def test_dispatch_authorized_existing_gh(self):
        res=cli.dispatch(QUESTION,fake_invoke_factory())
        self.assertEqual(res["run_id"],RUN)
        self.assertIn("actions/runs/",res["url"])
        self.assertEqual(len(res["models"]),2)

    def test_real_receipt_collection_and_answer(self):
        out=cli.collect(RUN,QUESTION,fake_invoke_factory())
        self.assertTrue(out["completed"])
        self.assertEqual(out["final_answer"],"156")
        self.assertEqual(out["independent_validation"],"INDEPENDENT_TOOL_VERIFIED")
        self.assertEqual(out["model_credits"],0)

    def test_wrong_question_cannot_replay_evidence(self):
        with self.assertRaises(ValueError):
            cli.collect(RUN,"Compute 315 times 29. Answer with only integer.",fake_invoke_factory())

    def test_tampered_final_rejected_even_with_model_proofs(self):
        with self.assertRaises(ValueError):
            cli.collect(RUN,QUESTION,fake_invoke_factory(tamper=True))

    def test_wrong_git_branch_rejected(self):
        def bad(command,timeout=45):
            return json.dumps({"databaseId":RUN,"status":"completed","conclusion":"success",
                "name":"Music Japan Horizon — Cloud","headBranch":"main","event":"workflow_dispatch"})
        with self.assertRaises(ValueError):cli.status(RUN,bad)

    def test_prompt_goes_to_public_action_without_executing_shell(self):
        observed=[]
        def fake(cmd,timeout=45):
            observed.append(cmd)
            return "https://github.com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/"+str(RUN)
        question="Calculate (9+4)*3."
        cli.dispatch(question,fake)
        self.assertIn("public_question="+question,observed[0])
        self.assertEqual(observed[0][:2],["workflow","run"])
        self.assertNotIn("powershell",observed[0])

    def test_invalid_run_id_rejected(self):
        with self.assertRaises(ValueError):cli.status(100,fake_invoke_factory())
    def test_no_public_flag_aborts_without_dispatch(self):
        res=subprocess.run([sys.executable,str(Path(cli.__file__).resolve()),"start",
             "--public-question",QUESTION],capture_output=True,text=True,timeout=6)
        self.assertNotEqual(res.returncode,0)
        self.assertIn("PUBLIC_CONFIRMATION_REQUIRED",res.stderr)

if __name__=="__main__":unittest.main()
