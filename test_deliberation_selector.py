import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cloud_probe import signed_record, verify
from cloud_infer_frontier import CHALLENGES, MODEL_MAP
import reasoning_mesh as mesh
import deliberation_selector as sel


def sample_stage(stage:str,answers=None,parent=None,challenger=None,count=12):
    profile=mesh.STAGES[stage]
    cfg=MODEL_MAP[profile]
    selected_answers=answers if answers is not None else ["unverified text"]*count
    items=[
        {
            "id":CHALLENGES[i]["id"],
            "answer":selected_answers[i],
            "http_status":200,
            "latency_seconds":2.0,
            "output_tokens":50,
            "request_budget_tokens":1024,
            "observed_reasoning_tokens":10,
            "generated_visible_text":bool(selected_answers[i]),
        }
        for i in range(count)
    ]
    return signed_record({
        "schema":mesh.SCHEMA,
        "stage":stage,
        "model_profile":profile,
        "model_id":cfg["model"],
        "model_revision":cfg["revision"],
        "weights_sha256":cfg["sha256"],
        "reasoning_effort":"high" if profile=="openai-20b" else "not_supported_by_this_checkpoint",
        "requested_output_tokens_per_task":1024,
        "upstream_sha256":parent,
        "challenge_sha256":challenger,
        "new_spend_usd":0,
        "proprietary_model_credits":0,
        "original_claude_or_gpt6":False,
        "public_runner_only":True,
        "items":items,
        "total_elapsed_seconds":24.0,
        "observed_at_utc":"2026-10-18T02:00:00+00:00",
    })

def stages(proposal_answers=None,critique_answers=None,synthesis_answers=None,count=12):
    p=sample_stage("propose",proposal_answers,count=count)
    c=sample_stage("challenge",critique_answers,parent=p["sha256"],count=count)
    s=sample_stage("synthesize",synthesis_answers,parent=p["sha256"],challenger=c["sha256"],count=count)
    return p,c,s


class ReasonerSelectorTest(unittest.TestCase):
    def test_math_corrects_without_benchmark_label(self):
        answer=sel.independently_select(
            "Calculate 314 times 29. Answer with integer only.",
            "9150","fake answer","9200")
        self.assertEqual(answer["answer"],"9106")
        self.assertEqual(answer["source"],"independent_tool_correction")
        self.assertTrue(answer["tool_correction"])
        self.assertFalse(answer["model_output_used"])

    def test_prefers_verified_synthesis(self):
        out=sel.independently_select("Calculate 7 times 8.","56","wrong","56")
        self.assertEqual(out["source_stage"],"synthesize")
        self.assertEqual(out["source"],"tool_verified_model")

    def test_prefers_verified_first_if_synthesis_wrong(self):
        out=sel.independently_select("Calculate 7 times 8.","56","55","57")
        self.assertEqual(out["answer"],"56")
        self.assertEqual(out["source_stage"],"propose")

    def test_japanese_polite_and_casual_both_valid(self):
        prompt="Translate Good morning into Japanese. Only the translation."
        r=sel.independently_select(prompt,"おはようございます","おはよう","おはようございます")
        self.assertEqual(r["confidence"],"independently_verified")
        self.assertEqual(r["source"],"curated_translation_reference")
        self.assertFalse(r["tool_correction"])
        self.assertEqual(r["answer"],"おはようございます")

    def test_unseen_translation_variants_are_not_fabricated(self):
        r=sel.independently_select("Translate leverage into Japanese.","レバレッジ","てこ","梃子")
        self.assertEqual(r["confidence"],"requires_independent_review")

    def test_majority_is_not_verification(self):
        out=sel.independently_select("What is the capital of France?","Paris","Berlin","Paris")
        self.assertEqual(out["confidence"],"unverified_consensus")
        self.assertFalse(out["question_grounded"])

    def test_genuine_disagreement_not_confident(self):
        out=sel.independently_select("Write a poem.","a","b","c")
        self.assertEqual(out["source"],"unverified_synthesis")
        self.assertEqual(out["confidence"],"requires_independent_review")

    def test_tool_algebra_without_gold(self):
        out=sel.independently_select(
            "Solve 3x - 8 = 13. Number only.","0","other","2")
        self.assertEqual(out["answer"],"7")
        self.assertEqual(out["verification_domain"],"linear_equation")

    def test_stage_chain_model_identity(self):
        p,c,s=stages()
        self.assertEqual(sel.validate_stage_chain(p,c,s),12)
        c["model_id"]="fake-claude-6"
        c=signed_record(c)
        with self.assertRaisesRegex(ValueError,"MODEL_ID_MISMATCH"):
            sel.validate_stage_chain(p,c,s)

    def test_wrong_weights_rejected(self):
        p,c,s=stages()
        p["weights_sha256"]="0"*64
        p=signed_record(p)
        with self.assertRaisesRegex(ValueError,"MODEL_WEIGHT_MISMATCH"):
            sel.validate_stage_chain(p,c,s)

    def test_wrong_parent_proof_rejected(self):
        p,c,s=stages()
        s["challenge_sha256"]="0"*64
        s=signed_record(s)
        with self.assertRaisesRegex(ValueError,"PARENTS_SHA_MISMATCH"):
            sel.validate_stage_chain(p,c,s)

    def test_no_model_run_selector_only(self):
        p,c,s=stages()
        rec=sel.select_from_receipts(p,c,s)
        self.assertTrue(verify(rec))
        self.assertEqual(rec["model_outputs_generated_in_this_selection"],0)
        self.assertEqual(rec["model_credits_used"],0)
        self.assertEqual(rec["total_cases"],12)

    def test_posthoc_grading_is_separate(self):
        gold=[case["expected"] for case in CHALLENGES]
        p,c,s=stages(proposal_answers=gold,critique_answers=gold,synthesis_answers=gold)
        result=sel.select_from_receipts(p,c,s)
        self.assertEqual(result["total_cases"],12)
        summary=sel.test_score_after_selection(result)
        self.assertEqual(summary["total"],12)
        # Japanese strings and JSON can remain unverified even if correct.
        self.assertGreaterEqual(summary["passed"],10)

    def test_both_strict_and_reference_aware_scores(self):
        answers=[case["expected"] for case in CHALLENGES]
        answers[-1]="おはようございます"
        p,c,s=stages(answers,answers,answers)
        rec=sel.select_from_receipts(p,c,s)
        audit=sel.test_score_after_selection(rec)
        self.assertEqual(audit["passed"],11)
        self.assertEqual(audit["reference_aware_passed"],12)
        self.assertEqual(audit["rows"][-1]["equivalence_rule"],"curated_multilingual_equivalence")

    def test_token_budget_exceeded(self):
        p,c,s=stages()
        for x in p["items"]: x["output_tokens"]=999999
        p=signed_record(p)
        c["upstream_sha256"]=p["sha256"]
        c=signed_record(c)
        s["upstream_sha256"]=p["sha256"]
        s["challenge_sha256"]=c["sha256"]
        s=signed_record(s)
        with self.assertRaisesRegex(ValueError,"BUDGET_EXCEEDED"):
            sel.validate_stage_chain(p,c,s)

    def test_no_arbitrary_code_execution(self):
        x=sel.independently_select(
            "Write a poem on autumn.",
            "__import__('os').system('echo hacked')",
            "fall leaves","winter words")
        self.assertNotEqual(x["source"],"independent_tool_correction")
        self.assertEqual(x["confidence"],"requires_independent_review")

    def test_no_staged_label_embedded_in_choice(self):
        import inspect
        source=inspect.getsource(sel.independently_select)
        self.assertNotIn("CHALLENGES",source)
        self.assertNotIn("expected",source)
        self.assertNotIn("grade(",source)

    def test_local_no_model_dry_run(self):
        run=subprocess.run([sys.executable,str(Path(sel.__file__).resolve()),"--self-check"],
                           text=True,capture_output=True,timeout=7)
        self.assertEqual(run.returncode,0)
        self.assertIn('"model_calls": 0',run.stdout)

if __name__=="__main__":
    unittest.main()
