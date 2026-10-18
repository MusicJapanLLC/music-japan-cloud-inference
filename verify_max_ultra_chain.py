#!/usr/bin/env python3
"""Independent 20B -> 120B Max/Ultra evidence chain validator.

Reads verified local GitHub Actions artifacts; no model inference, remote
account access, provider credits, or confidential prompt collection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from cloud_probe import verify,signed_record,normalized
from reasoning_profiles import profile,validate_proposal
from render_official120b import read_verified as read_120b

SCHEMA="mj.max-ultra-evidence-chain/1.0"

def validate(question:str,mode:str,raw_proposal:dict,raw_final:dict)->dict:
    preset=profile(mode)
    _proposal=validate_proposal(raw_proposal,question,mode)
    if (not verify(raw_final) or
        raw_final.get("schema")!="mj.official-open-120b-chat/1.0" or
        raw_final.get("model_original_openai_gpt_oss_120b") is not True or
        raw_final.get("actual_local_120b_inference") is not True):
        raise ValueError("ACTUAL_120B_INFERENCE_NOT_VERIFIED")
    if raw_final.get("model_id")!="gpt-oss-120b-MXFP4":
        raise ValueError("ORIGINAL_120B_IDENTITY_FAILED")
    if raw_final.get("research_mode")!=mode:
        raise ValueError("MAX_ULTRA_PRESET_CHANGED")
    if (raw_final.get("external_open_model_proposal_applied") is not True or
        raw_final.get("draft_proposal_receipt_sha256")!=raw_proposal.get("sha256")):
        raise ValueError("CROSS_MODEL_SOURCE_SHA_MISMATCH")
    digest=hashlib.sha256(question.encode("utf8")).hexdigest()
    if raw_final.get("public_prompt_hash")!=digest:
        raise ValueError("FINAL_PROMPT_IDENTITY_FAILED")
    if (raw_final.get("rounds")!=preset["final_rounds"] or
        raw_final.get("reasoning_effort")!=preset["final_effort"] or
        raw_final.get("requested_tokens_each_round")!=preset["final_tokens_per_round"]):
        raise ValueError("FINAL_THINKING_PRESET_NOT_EXECUTED")
    if (raw_final.get("model_provider_credits_used")!=0 or
        raw_final.get("paid_compute_allocated") is not False):
        raise ValueError("NONZERO_CREDITS_OR_BILLING")
    answer=(raw_final.get("selection") or {}).get("final_answer")
    if not isinstance(answer,str) or not answer.strip():
        raise ValueError("FINAL_ANSWER_MISSING")
    if not isinstance(raw_final.get("round_proofs"),list) or len(raw_final["round_proofs"])!=preset["final_rounds"]:
        raise ValueError("INFERENCE_ROUNDS_NOT_PROVEN")
    if not all(isinstance(x,dict) and x.get("http_status")==200
               and isinstance(x.get("model_answer"),str) and bool(x["model_answer"].strip())
               for x in raw_final["round_proofs"]):
        raise ValueError("INFERENCE_ROUND_RESPONSE_INVALID")
    draft=raw_proposal.get("rounds_evidence",[])
    if not isinstance(draft,list) or len(draft)!=preset["draft_rounds"]:
        raise ValueError("DRAFT_ROUNDS_NOT_PROVEN")
    if not all(isinstance(x,dict) and x.get("official_model")=="gpt-oss-20b-MXFP4"
               and isinstance(x.get("answer"),str) and bool(x["answer"].strip())
               for x in draft):
        raise ValueError("DRAFT_RESPONSES_NOT_PROVEN")
    detail=raw_final.get("selection",{})
    result=signed_record({
        "schema":SCHEMA,
        "mode":mode,
        "verified_public_model_20b":"gpt-oss-20b-MXFP4",
        "verified_public_model_120b":"gpt-oss-120b-MXFP4",
        "actual_inference_rounds_20b":len(draft),
        "actual_inference_rounds_120b":len(raw_final["round_proofs"]),
        "actual_total_generation_rounds":len(draft)+len(raw_final["round_proofs"]),
        "total_measured_compute_seconds":round(
            float(raw_proposal.get("elapsed_total_seconds",0))+
            float(raw_final.get("elapsed_total_seconds",0)),2),
        "native_gpt_oss_reasoning_effort":"high",
        "independent_question_grounded_verification":detail.get("verification",{}).get("status","ABSTAIN"),
        "final_answer":answer[:20000],
        "final_answer_source":detail.get("source","unknown"),
        "proposal_receipt_sha256":raw_proposal["sha256"],
        "final_receipt_sha256":raw_final["sha256"],
        "new_spend_usd":0,
        "model_provider_credits_used":0,
        "claimed_private_gpt6_or_anthropic_model":False,
    })
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("max","ultra"),required=True)
    p.add_argument("--proposal",type=Path,required=True)
    p.add_argument("--final",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    import os
    question=os.environ.get("MJ_PUBLIC_RESEARCH_QUESTION","").strip()
    if not 4<=len(question)<=3200:
        raise ValueError("PUBLIC_QUESTION_INVALID")
    draft=json.loads(args.proposal.read_text("utf8"))
    final=read_120b(args.final)
    result=validate(question,args.mode,draft,final)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(normalized(result)+b"\n")
    print(json.dumps({
        "mode":result["mode"],"all_rounds":result["actual_total_generation_rounds"],
        "models":[result["verified_public_model_20b"],result["verified_public_model_120b"]],
        "answer_present":bool(result["final_answer"]),
        "independent_verification":result["independent_question_grounded_verification"],
        "evidence_sha256":result["sha256"],"model_credits":0,
    },sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
