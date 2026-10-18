"""Validate actual public 120B and 122B model outputs as an evidence-linked council.

Never reads benchmark answer keys. Independently derives math/date checks from
the PUBLIC user question. Model disagreement without tool verification is
explicitly not asserted true. It does NOT call an LLM itself.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

from cloud_probe import normalized, signed_record, verify
from objective_verifier import verify_answer
from mj_ultra import public_question
from horizon_fanout import SCHEMA as WORKER_SCHEMA, config

SCHEMA="mj.horizon.federated-reasoning/1.0"
REQUIRED=("oss120","qwen122")

def verified_receipt(record:dict,profile:str,question:str)->dict:
    question=public_question(question)
    cfg=config(profile)
    if not verify(record) or record.get("schema")!=WORKER_SCHEMA:
        raise ValueError("MODEL_PROOF_SHA256_INVALID")
    expected={
        "profile":profile,
        "model_id":cfg["id"],
        "model_repo":cfg["repo"],
        "model_revision":cfg["revision"],
        "weights_integrity_digest":cfg["sha256"],
        "weight_total_bytes":cfg["bytes"],
        "public_question_sha256":hashlib.sha256(question.encode("utf8")).hexdigest(),
        "actual_original_public_weight_inference":True,
        "proprietary_vendor_model":False,
        "http_status":200,
        "model_answer_present":True,
        "paid_cloud_jobs":0,
        "external_llm_model_credits":0,
        "question_in_receipt":False,
        "github_linux_cpu_ephemeral":True,
    }
    for key,value in expected.items():
        if record.get(key)!=value:
            raise ValueError("MODEL_SOURCE_OR_QUESTION_MISMATCH_"+key.upper())
    text=record.get("model_answer")
    if not isinstance(text,str) or not text.strip() or len(text)>12000:
        raise ValueError("NO_VERIFIED_MODEL_TEXT")
    return record

def decide(question:str,inputs:dict)->dict:
    question=public_question(question)
    if set(inputs)!=set(REQUIRED):
        raise ValueError("EXACT_TWO_MODELS_REQUIRED")
    records={m:verified_receipt(inputs[m],m,question) for m in REQUIRED}
    assessments={model:verify_answer(question,r["model_answer"]) for model,r in records.items()}
    correct=[p for p in REQUIRED if assessments[p]["status"]=="VERIFIED"]
    if correct:
        # An independently checked answer is preferable to a popular guess.
        winner=correct[0]   # original 120B is first when both are verified
        status="INDEPENDENT_TOOL_VERIFIED"
    else:
        winner="oss120"
        if records["oss120"]["model_answer"].strip()==records["qwen122"]["model_answer"].strip():
            status="UNVERIFIED_CONSENSUS"
        else:
            status="UNRESOLVED_MODEL_DISAGREEMENT"
    report=signed_record({
        "schema":SCHEMA,
        "observed_utc":max((records[p].get("observed_utc","") for p in REQUIRED), default=""),
        "public_question_sha256":hashlib.sha256(question.encode("utf8")).hexdigest(),
        "original_public_models":[records[p]["model_id"] for p in REQUIRED],
        "source_receipt_sha256":{p:records[p]["sha256"] for p in REQUIRED},
        "model_answers":{p:records[p]["model_answer"] for p in REQUIRED},
        "independent_checks":assessments,
        "tool_verified_model_count":len(correct),
        "selected_model":winner,
        "final_answer":records[winner]["model_answer"],
        "decision":status,
        "single_model_equivalence_with_gpt6_claimed":False,
        "model_credits_used":0,
        "additional_cloud_spend":0,
        "model_count":len(REQUIRED),
        "inference_done_during_referee":False,
        "total_original_generation_seconds":round(sum(
            records[p]["generation_seconds"] for p in REQUIRED),2),
        "model_concurrent_workers":2,
        "global_shared_ram_gib_claimed":False,
        "dedicated_gpu_allocated":False,
    })
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--oss120",type=Path,required=True)
    p.add_argument("--qwen122",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--public-question",required=True)
    args=p.parse_args()
    data={
        "oss120":json.loads(args.oss120.read_text(encoding="utf8")),
        "qwen122":json.loads(args.qwen122.read_text(encoding="utf8")),
    }
    result=decide(args.public_question,data)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(normalized(result)+b"\n")
    print(json.dumps({
        "status":"ACTUAL_120B_122B_FEDERATION_VERIFIED",
        "selected":result["selected_model"],
        "independent_check":result["decision"],
        "sha256":result["sha256"],
        "model_count":2,
        "external_credits":0,
    },sort_keys=True))
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except (ValueError,OSError,KeyError,TypeError) as e:
        print(json.dumps({"error":str(e)[:160]}),file=sys.stderr)
        sys.exit(2)
