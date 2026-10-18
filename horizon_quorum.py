#!/usr/bin/env python3
"""Music Japan Horizon Quorum: merge independently executed >120B public models.

One AI-facing answer from 2-4 separately verified GitHub Linux model runs.
- gpt-oss-120b + Qwen3.5-122B are required when using this quorum.
- Qwen3-235B Thinking and Qwen3.5-397B optional after real inference.
- Independent objective verification overrides model consensus only on supported tasks.
- All model provenance and exact question evidence must match.
- Fail closed on fake / incomplete / unauthenticated *execution evidence*.
- No proprietary model API, no copied credentials, no billing, no network calls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import sys
from pathlib import Path

from cloud_probe import normalized, signed_record, verify
from horizon_fanout import SCHEMA as HORIZON_WORKER_SCHEMA, config as horizon_config
from beyond120_openweight import SCHEMA as FRONTIER_WORKER_SCHEMA, MODELS, PROMPTS
from objective_verifier import verify_answer, apply_verifier
from mj_ultra import public_question

SCHEMA = "mj.horizon.quorum/1.0"
REQUIRED = ("oss120", "qwen122")
OPTIONAL = ("thinking-235b-iq2m", "frontier-397b-iq1s")
MAX_ALL = 4
MAX_ANSWER_CHARS = 12000


def read_json(path:Path)->dict:
    if not path.is_file() or path.stat().st_size>100_000:
        raise ValueError("EVIDENCE_FILE_MISSING_OR_OVERSIZED")
    data=json.loads(path.read_text(encoding="utf8"))
    if not isinstance(data,dict):raise ValueError("EVIDENCE_NOT_JSON_OBJECT")
    return data


def verify_horizon(profile:str, record:dict, question:str)->dict:
    if profile not in REQUIRED:
        raise ValueError("WRONG_HORIZON_MODEL")
    if not verify(record) or record.get("schema")!=HORIZON_WORKER_SCHEMA:
        raise ValueError("HORIZON_RECORD_HASH_INVALID")
    cfg=horizon_config(profile)
    fields={
        "profile":profile,
        "model_id":cfg["id"],
        "model_repo":cfg["repo"],
        "model_revision":cfg["revision"],
        "weights_integrity_digest":cfg["sha256"],
        "weight_total_bytes":cfg["bytes"],
        "public_question_sha256":hashlib.sha256(question.encode("utf8")).hexdigest(),
        "actual_original_public_weight_inference":True,
        "proprietary_vendor_model":False,
        "model_answer_present":True,
        "http_status":200,
        "github_linux_cpu_ephemeral":True,
        "paid_cloud_jobs":0,
        "external_llm_model_credits":0,
        "question_in_receipt":False,
    }
    for key,value in fields.items():
        if record.get(key)!=value:
            raise ValueError("HORIZON_FIELD_MISMATCH_"+key.upper())
    answer=record.get("model_answer")
    if not isinstance(answer,str) or not answer.strip() or len(answer)>MAX_ANSWER_CHARS:
        raise ValueError("HORIZON_ANSWER_INVALID")
    return {
        "profile":profile,"model_id":cfg["id"],"answer":answer.strip(),
        "receipt_sha256":record["sha256"],
        "latency_seconds":float(record.get("generation_seconds",0)),
        "source":"official_original_open_weight",
        "verified_execution":True,
    }


def verify_frontier(profile:str,record:dict,question:str)->dict:
    if profile not in OPTIONAL:
        raise ValueError("WRONG_FRONTIER_MODEL")
    if question!=PROMPTS[0]:
        raise ValueError("FRONTIER_FIXED_QUESTION_NOT_SAME")
    if not verify(record) or record.get("schema")!=FRONTIER_WORKER_SCHEMA:
        raise ValueError("FRONTIER_RECORD_HASH_INVALID")
    cfg=MODELS[profile]
    fields={
        "profile":profile,
        "model_id":cfg["model_alias"],
        "model_repo":cfg["repo"],
        "revision":cfg["revision"],
        "quantization":cfg["quant"],
        "model_loaded":True,
        "real_inference":True,
        "public_github_linux_cpu_only":True,
        "not_a_proprietary_model":True,
        "model_provider_credits":0,
        "paid_cloud_allocations":0,
    }
    for key,value in fields.items():
        if record.get(key)!=value:
            raise ValueError("FRONTIER_FIELD_MISMATCH_"+key.upper())
    expected_parts=[
        {"path":name,"size":size,"sha256":digest}
        for name,size,digest in cfg["parts"]
    ]
    if record.get("model_shards")!=expected_parts:
        raise ValueError("FRONTIER_SHARD_PROVENANCE_MISMATCH")
    rows=record.get("results")
    if not isinstance(rows,list) or not rows:
        raise ValueError("FRONTIER_RESULTS_MISSING")
    head=rows[0]
    if not isinstance(head,dict) or head.get("id")!=1 or head.get("http_status")!=200 or head.get("actual_model_generated_text") is not True:
        raise ValueError("FRONTIER_INFERENCE_NOT_VERIFIED")
    answer=head.get("visible_answer")
    if not isinstance(answer,str) or not answer.strip() or len(answer)>400:
        raise ValueError("FRONTIER_ANSWER_NOT_AVAILABLE")
    if head.get("completion_tokens") is not None and (
        not isinstance(head["completion_tokens"],int) or head["completion_tokens"]<1
    ):
        raise ValueError("FRONTIER_COMPLETION_TOKENS_INVALID")
    return {
        "profile":profile,"model_id":cfg["model_alias"],"answer":answer.strip(),
        "receipt_sha256":record["sha256"],
        "latency_seconds":float(head.get("inference_seconds",0)),
        "source":"community_quantized_official_open_weight",
        "verified_execution":True,
    }


def select(question:str,evidence:dict[str,dict], require_all:bool=False)->dict:
    question=public_question(question)
    if not all(x in evidence for x in REQUIRED):
        raise ValueError("MISSING_REQUIRED_INDEPENDENT_MODEL_WORKERS")
    if any(x not in (*REQUIRED,*OPTIONAL) for x in evidence):
        raise ValueError("UNSUPPORTED_MODEL_PROFILE")
    if require_all and set(evidence)!=set((*REQUIRED,*OPTIONAL)):
        raise ValueError("ALL_FOUR_PROFILES_REQUIRED")
    if not 2<=len(evidence)<=MAX_ALL:
        raise ValueError("QUORUM_SIZE_INVALID")
    verified={}
    for profile,record in evidence.items():
        verified[profile]=(verify_horizon(profile,record,question)
                          if profile in REQUIRED else verify_frontier(profile,record,question))
    checks={profile:verify_answer(question,item["answer"]) for profile,item in verified.items()}
    passing=[profile for profile in (*REQUIRED,*OPTIONAL)
             if profile in verified and checks[profile]["status"]=="VERIFIED"]
    if passing:
        selected=passing[0]
        answer=verified[selected]["answer"]
        decision="INDEPENDENT_OBJECTIVE_TOOL_VERIFIED"
        confidence="tool_verified"
    else:
        counts={}
        for profile,item in verified.items():
            key=item["answer"].strip()
            counts.setdefault(key,[]).append(profile)
        most=max(counts.values(),key=len)
        if len(most)>len(verified)/2:
            selected=most[0];answer=verified[selected]["answer"]
            decision="UNVERIFIED_MULTI_MODEL_MAJORITY"
            confidence="model_consensus_only"
        else:
            selected="oss120";answer=verified[selected]["answer"]
            decision="NO_INDEPENDENT_VERIFICATION_CONFLICT"
            confidence="requires_review"
    report=signed_record({
        "schema":SCHEMA,
        "question_sha256":hashlib.sha256(question.encode("utf8")).hexdigest(),
        "question_in_receipt":False,
        "observed_utc":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "model_workers":len(verified),
        "official_open_weights_actual_outputs":True,
        "model_api_credits":0,
        "additional_spend_usd":0,
        "dedicated_gpu_allocated":False,
        "model_weights_claimed_to_be_claude_gpt6_or_gemini_proprietary":False,
        "individual_model_answers":verified,
        "independent_checks":checks,
        "tool_verified_model_count":len(passing),
        "selected_model":selected,
        "final_answer":answer,
        "decision":decision,"confidence":confidence,
    })
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oss120",type=Path,required=True)
    parser.add_argument("--qwen122",type=Path,required=True)
    parser.add_argument("--thinking235",type=Path)
    parser.add_argument("--frontier397",type=Path)
    parser.add_argument("--question",required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--require-four",action="store_true")
    args=parser.parse_args()
    evidence={"oss120":read_json(args.oss120),"qwen122":read_json(args.qwen122)}
    if args.thinking235:evidence["thinking-235b-iq2m"]=read_json(args.thinking235)
    if args.frontier397:evidence["frontier-397b-iq1s"]=read_json(args.frontier397)
    report=select(args.question,evidence,require_all=args.require_four)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(normalized(report)+b"\n")
    print(json.dumps({
       "decision":report["decision"],
       "workers":report["model_workers"],
       "selected":report["selected_model"],
       "sha256":report["sha256"],
       "tool_verified":report["tool_verified_model_count"],
       "credits_used":0
    },sort_keys=True))
    return 0


if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(json.dumps({"status":"FAILED","type":type(error).__name__,
                          "reason":str(error)[:140]}),file=sys.stderr)
        sys.exit(2)
