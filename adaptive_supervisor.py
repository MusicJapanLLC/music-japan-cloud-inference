#!/usr/bin/env python3
"""Adaptive zero-credit controller of REAL official-open gpt-oss-20b and gpt-oss-120b.

Plan -> verified 20B draft -> escalate based on actual results -> verified 120B
-> choose evidence-grounded winner. Only public synthetic/research prompts on
GitHub Actions. This controller itself performs no inference or remote calls.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
import re
import sys
from pathlib import Path
from cloud_probe import signed_record, normalized, verify
from cloud_owner_chat import CFG as MODEL20, validate_prompt
from reasoning_profiles import profile, validate_proposal
from objective_verifier import verify_answer
from language_equivalence import verify_translation
from verify_max_ultra_chain import validate as check_real_chain

PLAN="mj.adaptive-supervisor.plan/1.0"
DECISION="mj.adaptive-supervisor.decision/1.0"
ANSWER="mj.adaptive-supervisor.answer/1.0"
MODES=("auto","force-ultra")
COMPLEX=re.compile(r"\b(architecture|security|proof|explain|compare|evaluate|debug|design|reason|research|strategy|counterexample|complex|failure|implement|code|database)\b|設計|開発|推論|コード|研究|複雑|原因|反証|検証|戦略|セキュリティ|最適化|自律|品質",re.I)

def question_hash(q:str)->str:
    return hashlib.sha256(validate_prompt(q).encode("utf8")).hexdigest()

def check_answer(q:str,a:str)->dict:
    result=verify_answer(q,a)
    if result["status"]=="ABSTAIN":
        other=verify_translation(q,a)
        if other["status"]=="VERIFIED":
            return {"status":"VERIFIED","domain":other["domain"],"independent":True}
    return result

def plan_question(q:str, requested:str="auto")->dict:
    q=validate_prompt(q)
    if requested not in MODES:raise ValueError("UNKNOWN_MODE")
    looks_checkable=check_answer(q,"WRONG")["status"]=="REJECTED"
    advanced=bool(COMPLEX.search(q)) or len(q)>220
    chosen="ultra" if requested=="force-ultra" or (advanced and not looks_checkable) else "max"
    p=profile(chosen)
    return signed_record({
        "schema":PLAN,"question_sha256":question_hash(q),
        "requested_mode":requested,"preset":chosen,
        "reason":"FORCED_HIGHEST_THINKING" if requested=="force-ultra" else
                 ("COMPLEX_TASK" if advanced else "QUESTION_GROUNDED_FAST_CHECK"),
        "20b_rounds":p["draft_rounds"],"20b_tokens":p["draft_tokens_per_round"],
        "120b_rounds":p["final_rounds"],"120b_tokens":p["final_tokens_per_round"],
        "effort":"high","new_spend_usd":0,"proprietary_model_credits":0})

def validate_plan(q:str,p:dict)->str:
    if not verify(p) or p.get("schema")!=PLAN or p.get("question_sha256")!=question_hash(q):
        raise ValueError("PLAN_OR_QUESTION_MISMATCH")
    if p.get("requested_mode") not in MODES or p.get("preset") not in ("max","ultra"):
        raise ValueError("UNKNOWN_POLICY")
    expected=plan_question(q,p["requested_mode"])
    for key in ("preset","20b_rounds","20b_tokens","120b_rounds","120b_tokens","effort"):
        if p.get(key)!=expected[key]:raise ValueError("POLICY_TAMPER")
    if p.get("new_spend_usd")!=0 or p.get("proprietary_model_credits")!=0:
        raise ValueError("BILLING_TAMPER")
    return p["preset"]

def validate_draft(q:str,p:dict,d:dict)->str:
    return validate_proposal(d,validate_prompt(q),validate_plan(q,p))

def decide(q:str,p:dict,d:dict)->dict:
    output=validate_draft(q,p,d)
    independent=check_answer(q,output)
    last=(d.get("rounds_evidence") or [{}])[-1].get("answer","")
    if not isinstance(last,str) or not last.strip():raise ValueError("DRAFT_TEXT_EMPTY")
    bypass=(p["requested_mode"]=="auto" and independent["status"]=="VERIFIED")
    return signed_record({
        "schema":DECISION,"question_sha256":p["question_sha256"],
        "plan_sha256":p["sha256"],"draft_sha256":d["sha256"],
        "real_20b":MODEL20["model"],
        "draft_verification":independent["status"],
        "last_raw_20b_verification":check_answer(q,last)["status"],
        "escalate_to_120b":not bypass,"route":"EARLY_VERIFIED" if bypass else "REAL_120B",
        "preset":p["preset"],"new_spend_usd":0,"model_api_credits":0})

def validate_decision(q:str,p:dict,d:dict,c:dict)->bool:
    validate_draft(q,p,d)
    if not verify(c) or c.get("schema")!=DECISION:raise ValueError("DECISION_UNVERIFIED")
    recomputed=decide(q,p,d)
    if c!=recomputed:raise ValueError("DECISION_REPLAY_MISMATCH")
    return bool(c["escalate_to_120b"])

def finish(q:str,p:dict,d:dict,c:dict,final:dict|None=None)->dict:
    needs120=validate_decision(q,p,d,c)
    first=(d.get("selection") or {}).get("final_answer")
    if not isinstance(first,str) or not first.strip():raise ValueError("DRAFT_MISSING")
    if needs120:
        if not isinstance(final,dict):raise ValueError("120B_PROOF_REQUIRED")
        provenance=check_real_chain(validate_prompt(q),p["preset"],d,final)
        ultimate=(final.get("selection") or {}).get("final_answer")
        if not isinstance(ultimate,str) or not ultimate.strip():raise ValueError("120B_TEXT_EMPTY")
        status120=check_answer(q,ultimate)
    else:
        if final is not None:raise ValueError("EXTRA_120B_NOT_APPROVED_BY_POLICY")
        provenance=None;ultimate=None;status120=None
    status20=check_answer(q,first)
    if ultimate is not None and status120["status"]=="VERIFIED":
        answer,why,check=ultimate,"REAL_120B_TOOL_VERIFIED",status120
    elif status20["status"]=="VERIFIED":
        answer,why,check=first,"REAL_20B_TOOL_VERIFIED",status20
    elif ultimate is None:
        raise ValueError("EARLY_COMPLETION_WITHOUT_TOOL_PROOF")
    else:
        answer,why,check=ultimate,"120B_RESULT_WITH_UNVERIFIED_CLAIMS",status120
    return signed_record({
        "schema":ANSWER,"question_sha256":p["question_sha256"],
        "policy_sha256":p["sha256"],"route_sha256":c["sha256"],
        "20b_sha256":d["sha256"],"120b_sha256":final["sha256"] if final else None,
        "cross_model_chain_sha256":provenance["sha256"] if provenance else None,
        "models":[MODEL20["model"]]+(["gpt-oss-120b-MXFP4"] if ultimate is not None else []),
        "reasoning_preset":p["preset"],
        "actual_model_rounds":len(d["rounds_evidence"])+(
                               len(final.get("round_proofs",[])) if final else 0),
        "model_agreement":ultimate is None or first==ultimate,
        "answer_source":why,"answer":answer[:30000],
        "verification_status":check["status"],"verification_domain":check.get("domain"),
        "api_model_credits":0,"new_spend_usd":0,
        "equivalent_to_closed_gpt6_or_claude_claimed":False})

def safe_summary(result:dict)->str:
    if not verify(result) or result.get("schema")!=ANSWER:
        raise ValueError("INVALID_FINAL_RESULT")
    if len(result["answer"])>30000:raise ValueError("TOO_LONG")
    return ("## Music Japan Adaptive Supervisor — actual public model response\n\n"
            "**Models:** "+html.escape(" → ".join(result["models"]))+
            "  \n**Observed inference rounds:** "+str(result["actual_model_rounds"])+
            "  \n**Selected:** "+html.escape(result["answer_source"])+
            "  \n**Independent verification:** "+html.escape(result["verification_status"])+
            "  \n**Provider inference credits:** 0\n\n"
            "### Answer\n\n<pre>"+html.escape(result["answer"])+"</pre>\n\n"
            "Evidence SHA256: "+result["sha256"]+"\n")

def load(path:Path)->dict:
    obj=json.loads(path.read_text("utf8"))
    if not verify(obj):raise ValueError("INVALID_RECEIPT")
    return obj

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("phase",choices=("plan","decide","finish"))
    ap.add_argument("--mode",choices=MODES,default="auto")
    ap.add_argument("--plan",type=Path)
    ap.add_argument("--draft",type=Path)
    ap.add_argument("--decision",type=Path)
    ap.add_argument("--final",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--github-summary",action="store_true")
    args=ap.parse_args()
    question=validate_prompt(os.getenv("MJ_PUBLIC_RESEARCH_QUESTION",""))
    if args.phase=="plan": data=plan_question(question,args.mode)
    elif args.phase=="decide":
        if not args.plan or not args.draft:raise ValueError("DRAFT_AND_PLAN_REQUIRED")
        data=decide(question,load(args.plan),load(args.draft))
    else:
        if not args.plan or not args.draft or not args.decision:
            raise ValueError("RECEIPTS_REQUIRED")
        data=finish(question,load(args.plan),load(args.draft),load(args.decision),
                    load(args.final) if args.final else None)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(normalized(data)+b"\n")
    if args.phase=="finish" and args.github_summary:
        path=os.getenv("GITHUB_STEP_SUMMARY")
        if not path:raise ValueError("GITHUB_SUMMARY_NOT_CONFIGURED")
        with open(path,"a",encoding="utf8") as out:out.write(safe_summary(data))
    print(json.dumps({"phase":args.phase,"schema":data["schema"],
       "sha256":data["sha256"],"preset":data.get("preset",data.get("reasoning_preset")),
       "escalate":data.get("escalate_to_120b"),
       "selected_source":data.get("answer_source"),
       "model_rounds":data.get("actual_model_rounds")},sort_keys=True))
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({"status":"ERROR","code":type(exc).__name__,
                          "reason":str(exc)[:150]}),file=sys.stderr)
        sys.exit(2)
