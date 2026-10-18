#!/usr/bin/env python3
"""Music Japan Deliberation Selector: model council + independent tool reasoning.

Three recorded REAL open-weight model runs:
  1. gpt-oss-20b high-effort proposal
  2. independently trained public Qwable-Fable 9B critique
  3. gpt-oss-20b high-effort synthesis

Selection is based on deterministic tools/independent agreement, NOT hidden
benchmark answer keys. Gives unsupported tasks lower confidence rather than
fabricating certainty. Does not call any model/API or access private data.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from cloud_probe import normalized, signed_record, verify
from cloud_infer_frontier import CHALLENGES, MODEL_MAP
from objective_verifier import apply_verifier, verify_answer
from reasoning_mesh import read_verified, STAGES, receipt_ok

SCHEMA = "mj.deliberation-final/1.0"
MAX_ITEMS = 64
MAX_OUTPUT_CHARS = 2000
MAX_TOTAL_MODEL_TOKENS = 300_000
MAX_TOTAL_LATENCY_SECONDS = 12_000

def clean_answer(answer:str)->str:
    if not isinstance(answer,str) or len(answer)>MAX_OUTPUT_CHARS:
        raise ValueError("INVALID_MODEL_REPLY")
    return answer.strip()

def independently_select(question:str, proposal:str, critique:str, synthesis:str)->dict:
    """Select answer WITHOUT reference to benchmark gold labels."""
    if not isinstance(question,str) or len(question)>8000:
        raise ValueError("INVALID_QUESTION")
    answers = {
        "propose": clean_answer(proposal),
        "challenge": clean_answer(critique),
        "synthesize": clean_answer(synthesis),
    }
    grades = {name: verify_answer(question,answer) for name,answer in answers.items()}
    applicable = any(x["status"] in ("VERIFIED","REJECTED") for x in grades.values())
    if applicable:
        # Prefer final synthesis if independently verified, otherwise keep
        # an earlier independently verified model answer.
        for name in ("synthesize","propose","challenge"):
            if grades[name]["status"]=="VERIFIED":
                return {
                    "answer":answers[name],
                    "source":"tool_verified_model",
                    "source_stage":name,
                    "confidence":"independently_verified",
                    "model_output_used":True,
                    "tool_correction":False,
                    "verification_domain":grades[name]["domain"],
                    "question_grounded":True,
                }
        # The tool computes the result from the original question only,
        # not from stored benchmark answer labels.
        derived = apply_verifier(question,answers["synthesize"])
        if derived["source"]!="deterministic_tool_correction":
            raise ValueError("INDEPENDENT_TOOL_CORRECTION_UNAVAILABLE")
        return {
            "answer":derived["final_answer"],
            "source":"independent_tool_correction",
            "source_stage":None,
            "confidence":"independently_verified",
            "model_output_used":False,
            "tool_correction":True,
            "verification_domain":derived["verification"]["domain"],
            "question_grounded":True,
        }
    # For general open-ended questions a majority is evidence of agreement,
    # but NOT proof of factual correctness.
    if answers["synthesize"] and (answers["synthesize"]==answers["propose"] or
                                   answers["synthesize"]==answers["challenge"]):
        return {"answer":answers["synthesize"],"source":"model_council_agreement",
                "source_stage":"synthesize","confidence":"unverified_consensus",
                "model_output_used":True,"tool_correction":False,
                "verification_domain":None,"question_grounded":False}
    if answers["propose"] and answers["propose"]==answers["challenge"]:
        return {"answer":answers["propose"],"source":"model_council_agreement",
                "source_stage":"propose","confidence":"unverified_consensus",
                "model_output_used":True,"tool_correction":False,
                "verification_domain":None,"question_grounded":False}
    return {"answer":answers["synthesize"],"source":"unverified_synthesis",
            "source_stage":"synthesize","confidence":"requires_independent_review",
            "model_output_used":bool(answers["synthesize"]),
            "tool_correction":False, "verification_domain":None,
            "question_grounded":False}

def validate_stage_chain(proposal:dict, critique:dict, synthesis:dict)->int:
    for stage, rec in (("propose",proposal),("challenge",critique),("synthesize",synthesis)):
        if not receipt_ok(rec, stage):
            raise ValueError("STAGE_INVALID_"+stage)
        profile = STAGES[stage]
        expected = MODEL_MAP[profile]
        if rec.get("model_id") != expected["model"]:
            raise ValueError("MODEL_ID_MISMATCH_"+stage)
        if rec.get("weights_sha256") != expected["sha256"]:
            raise ValueError("MODEL_WEIGHT_MISMATCH_"+stage)
        if rec.get("model_revision") != expected["revision"]:
            raise ValueError("MODEL_REVISION_MISMATCH_"+stage)
    if critique["upstream_sha256"]!=proposal["sha256"]:
        raise ValueError("CRITIQUE_PARENT_SHA_MISMATCH")
    if synthesis["upstream_sha256"]!=proposal["sha256"] or synthesis["challenge_sha256"]!=critique["sha256"]:
        raise ValueError("SYNTHESIS_PARENTS_SHA_MISMATCH")
    count = len(proposal["items"])
    if not (1 <= count <= min(MAX_ITEMS,len(CHALLENGES))) or count != len(critique["items"]) or count != len(synthesis["items"]):
        raise ValueError("TASK_LENGTH_NOT_COMPARABLE")
    tokens = sum(int(x.get("output_tokens",0)) for r in (proposal,critique,synthesis) for x in r["items"])
    latency = sum(float(x.get("latency_seconds",0)) for r in (proposal,critique,synthesis) for x in r["items"])
    if tokens<0 or tokens>MAX_TOTAL_MODEL_TOKENS or latency>MAX_TOTAL_LATENCY_SECONDS:
        raise ValueError("EXPERIMENT_REASONING_BUDGET_EXCEEDED")
    return count

def select_from_receipts(proposal:dict,critique:dict,synthesis:dict)->dict:
    count=validate_stage_chain(proposal,critique,synthesis)
    items=[]
    for index in range(count):
        # Deliberately read the question only; gold case["expected"] is not
        # available to the decision function.
        question = CHALLENGES[index]["prompt"]
        trio = (proposal["items"][index],critique["items"][index],synthesis["items"][index])
        item=independently_select(question,*(row["answer"] for row in trio))
        item.update({
            "id":CHALLENGES[index]["id"],
            "latency_seconds":round(sum(float(x["latency_seconds"]) for x in trio),2),
            "actual_model_output_tokens":sum(int(x["output_tokens"]) for x in trio),
            "participating_models":[MODEL_MAP[STAGES[x]]["model"] for x in ("propose","challenge","synthesize")],
        })
        items.append(item)
    record=signed_record({
        "schema":SCHEMA,
        "decision_policy":"question_grounded_verification_then_council_then_abstain_from_certainty",
        "source_evidence":{
            "proposal":proposal["sha256"],
            "critique":critique["sha256"],
            "synthesis":synthesis["sha256"],
        },
        "model_credits_used":0,
        "additional_cloud_spend":0,
        "model_original_fable_or_gpt6":False,
        "model_outputs_generated_in_this_selection":0,
        "total_cases":len(items),
        "independently_verified":sum(x["confidence"]=="independently_verified" for x in items),
        "tool_corrections":sum(x["tool_correction"] for x in items),
        "unverified_consensus":sum(x["confidence"]=="unverified_consensus" for x in items),
        "needs_review":sum(x["confidence"]=="requires_independent_review" for x in items),
        "items":items,
    })
    return record

def test_score_after_selection(record:dict)->dict:
    """Offline audit only: test labels NEVER enter the answer selector."""
    if not verify(record) or record.get("schema")!=SCHEMA:
        raise ValueError("OUTPUT_NOT_VERIFIED")
    from cloud_infer_frontier import grade
    if len(record["items"])>len(CHALLENGES):raise ValueError("EVAL_OUT_OF_RANGE")
    scores=[{"id":CHALLENGES[i]["id"],"passed":bool(grade(item["answer"],CHALLENGES[i])),
             "source":item["source"]} for i,item in enumerate(record["items"])]
    return {
        "total":len(scores),"passed":sum(x["passed"] for x in scores),"rows":scores,
        "note":"Tool-derived answers scored separately from raw model-only results; small public synthetic suite",
    }

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--proposal",type=Path)
    p.add_argument("--critique",type=Path)
    p.add_argument("--synthesis",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--audit",action="store_true")
    p.add_argument("--self-check",action="store_true")
    args=p.parse_args(argv)
    if args.self_check:
        print(json.dumps({"schema":SCHEMA,"source":"local_question_grounded",
            "model_calls":0, "new_spend_usd":0},sort_keys=True))
        return 0
    if not all((args.proposal,args.critique,args.synthesis,args.out)):
        p.error("provide all 3 signed stage receipts and --out")
    result=select_from_receipts(
        read_verified(args.proposal,"propose"),
        read_verified(args.critique,"challenge"),
        read_verified(args.synthesis,"synthesize"))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(normalized(result)+b"\n")
    summary={"receipt_sha256":result["sha256"],
             "cases":result["total_cases"],
             "verified":result["independently_verified"],
             "tool_corrections":result["tool_corrections"],
             "unverified":result["unverified_consensus"]+result["needs_review"]}
    if args.audit:summary["posthoc_evaluation"]=test_score_after_selection(result)
    print(json.dumps(summary,sort_keys=True,ensure_ascii=False))
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(json.dumps({"status":"FAILED","reason":str(error)[:150]}),file=sys.stderr)
        sys.exit(2)
