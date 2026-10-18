#!/usr/bin/env python3
"""Music Japan adaptive frontier — incrementally combine real 120B..397B output.

No single straggler failure can suppress another completed model. Each provider
receipt is bound to original-public weights, source GitHub run and question,
then independently checkable claims are validated from question content.
A single available answer is *not* presented as multi-model consensus.

No model downloads, external inference, spending, tokens or personal content.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from cloud_probe import normalized, signed_record, verify
from horizon_quorum import verify_horizon, verify_frontier, REQUIRED, OPTIONAL, read_json
from objective_verifier import verify_answer

SCHEMA = "mj.horizon.adaptive/1.0"
PROFILE_ORDER = ("oss120","qwen122","thinking-235b-iq2m","frontier-397b-iq1s")
MAX_ANSWER_CHARS = 12000
WEIGHTS = {"oss120":120,"qwen122":122,"thinking-235b-iq2m":235,"frontier-397b-iq1s":397}

def select_available(question:str, receipts:dict[str,dict], minimum_workers:int=1) -> dict:
    if not isinstance(question,str) or not 5 <= len(question) <= 3000:
        raise ValueError("PUBLIC_QUESTION_LENGTH_INVALID")
    if not isinstance(receipts,dict) or not receipts:
        raise ValueError("NO_REAL_MODEL_PROOFS_AVAILABLE")
    if type(minimum_workers) is not int or not 1<=minimum_workers<=len(PROFILE_ORDER):
        raise ValueError("INVALID_WORKER_REQUIREMENT")
    if any(key not in PROFILE_ORDER for key in receipts):
        raise ValueError("UNREGISTERED_MODEL_PROFILE")
    if len(receipts)<minimum_workers:
        raise ValueError("MINIMUM_VERIFIED_MODEL_COUNT_NOT_MET")
    validated={}
    for profile in PROFILE_ORDER:
        if profile not in receipts:
            continue
        if profile in REQUIRED:
            validated[profile]=verify_horizon(profile,receipts[profile],question)
        else:
            validated[profile]=verify_frontier(profile,receipts[profile],question)
        if validated[profile]["answer"]=="":
            raise ValueError("EMPTY_REAL_MODEL_OUTPUT")
    checks={model:verify_answer(question,record["answer"]) for model,record in validated.items()}
    verified=[name for name,record in checks.items() if record["status"]=="VERIFIED"]
    if verified:
        # Tool-proven correct answers outrank any unverified disagreement.
        winner=min(verified,key=lambda key:(validated[key]["latency_seconds"],PROFILE_ORDER.index(key)))
        answer=validated[winner]["answer"]
        status="OBJECTIVE_VERIFICATION_PASSED"
        confidence="independently_verified"
    else:
        votes={}
        for name,record in validated.items():
            votes.setdefault(record["answer"],[]).append(name)
        winning_votes=max(votes.values(),key=len)
        if len(validated)>1 and len(winning_votes)>len(validated)/2:
            winner=winning_votes[0]
            answer=validated[winner]["answer"]
            status="UNVERIFIED_MULTI_MODEL_CONSENSUS"
            confidence="consensus_only"
        else:
            winner=next(iter(validated))
            answer=validated[winner]["answer"]
            status="SINGLE_MODEL_OR_DISAGREEMENT"
            confidence="requires_external_verification"
    result=signed_record({
        "schema":SCHEMA,
        "question_sha256":hashlib.sha256(question.encode("utf8")).hexdigest(),
        "question_text_exported":False,
        "observed_utc":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "actual_completed_worker_count":len(validated),
        "model_profiles":list(validated),
        "parameter_total_b_per_worker":{k:WEIGHTS[k] for k in validated},
        "real_model_provenance_checked":True,
        "same_model_as_proprietary_chatgpt_or_anthropic":False,
        "new_paid_compute_usd":0,
        "vendor_api_inference_credits":0,
        "model_final_answer":answer,
        "selected_profile":winner,
        "decision":status,
        "confidence":confidence,
        "independent_checks":checks,
        "per_model_evidence":validated,
        "total_generation_seconds_across_workers":round(sum(max(0,x["latency_seconds"]) for x in validated.values()),2),
        "note":"Independent hosted CPU research workers, not a shared GPU or single downloaded closed model.",
    })
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--oss120",type=Path)
    p.add_argument("--qwen122",type=Path)
    p.add_argument("--thinking235",type=Path)
    p.add_argument("--frontier397",type=Path)
    p.add_argument("--question",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--min-workers",type=int,default=1)
    args=p.parse_args()
    files={
        "oss120":args.oss120,
        "qwen122":args.qwen122,
        "thinking-235b-iq2m":args.thinking235,
        "frontier-397b-iq1s":args.frontier397,
    }
    receipts={key:read_json(value) for key,value in files.items() if value is not None}
    response=select_available(args.question,receipts,args.min_workers)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(normalized(response)+b"\n")
    print(json.dumps({"status":"PROVENANCE_VALIDATED","workers":response["actual_completed_worker_count"],
                      "confidence":response["confidence"],"decision":response["decision"],
                      "selected":response["selected_profile"],"response_sha256":response["sha256"],
                      "external_model_credits":0},sort_keys=True))
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except (ValueError,OSError,KeyError,TypeError) as e:
        print(json.dumps({"status":"FAILED","reason":str(e)[:160]}),file=sys.stderr)
        sys.exit(2)
