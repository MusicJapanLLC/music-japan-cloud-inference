#!/usr/bin/env python3
"""Music Japan 120B Open Reasoning — real original OpenAI public model on cloud Linux.

Uses SHA-256 pinned gpt-oss-120b-MXFP4 with the tested mmap, fit-off,
no-repack settings that generated a real HTTP 200 answer with 15.61GiB RAM.

This provides arbitrary *public research* prompt inference, optionally with a
second high-reasoning revision. Zero proprietary API credits, no account
secrets or private business content. The cloud Linux job is ephemeral.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

import experimental_gptoss120b as model120b
from cloud_probe import normalized,signed_record
from objective_verifier import apply_verifier
from reasoning_profiles import profile as reasoning_profile, validate_proposal

SCHEMA="mj.official-open-120b-chat/1.0"
MAX_PUBLIC_PROMPT_CHARS=3200
MAX_TOKENS=1536
MAX_ROUNDS=2
MODEL_WAIT_SECONDS=390
TOTAL_RUN_SECONDS=1800

def check_prompt(question:str)->str:
    if not isinstance(question,str):raise ValueError("PUBLIC_PROMPT_STRING_REQUIRED")
    question=question.strip()
    if not 4<=len(question)<=MAX_PUBLIC_PROMPT_CHARS or "\0" in question:
        raise ValueError("PUBLIC_PROMPT_INVALID")
    return question

def check_limits(rounds:int, tokens:int, effort:str)->None:
    if type(rounds) is not int or not 1<=rounds<=MAX_ROUNDS:
        raise ValueError("INVALID_ROUND_COUNT")
    if type(tokens) is not int or not 64<=tokens<=MAX_TOKENS:
        raise ValueError("INVALID_MODEL_TOKEN_LIMIT")
    if effort not in ("low","medium","high"):
        raise ValueError("INVALID_THINKING_LEVEL")

def prompt_for_round(question:str, earlier:list[str], proposal_context:str="")->str:
    if not earlier:
        external = ("\n\nEARLIER INDEPENDENT PUBLIC OPEN-WEIGHT MODEL PROPOSAL (may be wrong):\n"+
                    proposal_context[:1600]) if proposal_context else ""
        return ("Solve the following problem thoroughly. Check arithmetic, "
                "evidence, edge cases and instructions before giving your final answer. "
                "Do not claim certainty without support.\nQUESTION:\n"+question+external)
    return ("Reconsider this from first principles. The previous proposal may "
            "contain subtle errors. Identify contradictions, re-evaluate all "
            "assumptions, and return only your strongest final response.\n"
            "ORIGINAL QUESTION:\n"+question+
            "\nEARLIER PROPOSAL:\n"+earlier[-1][:4000])

def model_body(question:str, answers:list[str], effort:str, tokens:int,
               proposal_context:str="")->dict:
    check_limits(1,tokens,effort)
    return {
        "model":model120b.ID,
        "messages":[
            {"role":"system","content":"You are Music Japan's open-weight research reasoning model. Be accurate and mark uncertainties."},
            {"role":"user","content":prompt_for_round(question,answers,proposal_context)}
        ],
        "reasoning_effort":effort,
        "chat_template_kwargs":{"reasoning_effort":effort},
        "max_tokens":tokens,
        "temperature":1.0,
        "stream":False,
    }

def extract_answer(status:int, data:dict)->tuple[str,int,int]:
    if status!=200 or not isinstance(data,dict):
        raise RuntimeError("MODEL_HTTP_FAILURE")
    try:
        message=data["choices"][0]["message"]
        text=message.get("content") or ""
        if not isinstance(text,str) or not text.strip():
            raise RuntimeError("NO_VISIBLE_MODEL_RESPONSE")
        usage=data.get("usage") or {}
        completion=int(usage.get("completion_tokens") or 0)
        reasoning=int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        return text.strip(),completion,reasoning
    except (KeyError,IndexError,TypeError):
        raise RuntimeError("MODEL_REPLY_FORMAT_INVALID")

def run(question:str,exe:Path,weights:Path,outfile:Path,
        rounds:int=1,tokens:int=512,effort:str="high",
        budget_seconds:int=TOTAL_RUN_SECONDS,
        proposal_context:str="",proposal_receipt_sha:str|None=None,
        research_mode:str|None=None)->dict:
    question=check_prompt(question)
    check_limits(rounds,tokens,effort)
    if not exe.is_file() or not weights.is_file():
        raise FileNotFoundError("VERIFIED_WEIGHTS_AND_LLAMA_CPP_REQUIRED")
    if not 300<=budget_seconds<=TOTAL_RUN_SECONDS:
        raise ValueError("RUN_TIME_BUDGET_INVALID")
    outfile.parent.mkdir(parents=True,exist_ok=True)
    key=secrets.token_urlsafe(48)
    env=dict(os.environ);env["LLAMA_API_KEY"]=key
    start=time.monotonic()
    with (outfile.parent/"official120b-runtime.log").open("wb") as log:
        command=[
            str(exe),"-m",str(weights),"-a",model120b.ID,
            "--host","127.0.0.1","--port",str(model120b.PORT),
            "--no-webui","--no-mmproj","--cache-ram","0",
            "--load-mode","mmap","--fit","off","--no-repack",
            "-c","1536","-b","64","-ub","32","-t","4","-np","1","-ngl","0",
        ]
        p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,
                           stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(MODEL_WAIT_SECONDS//3):
                if p.poll() is not None:
                    raise RuntimeError("OFFICIAL_120B_MODEL_START_FAILED")
                try:
                    st,_=model120b.client(key,"/health",timeout=3)
                    if st==200:break
                except (OSError,TimeoutError,ValueError):
                    pass
                time.sleep(3)
            else:
                raise TimeoutError("OFFICIAL_120B_READY_TIMEOUT")
            st,models=model120b.client(key,"/v1/models")
            if st!=200 or model120b.ID not in [m.get("id") for m in models.get("data",[]) if isinstance(m,dict)]:
                raise RuntimeError("REAL_120B_ID_NOT_VERIFIED")
            outputs=[]
            details=[]
            for round_index in range(rounds):
                elapsed=time.monotonic()-start
                if elapsed>=budget_seconds-80:
                    raise TimeoutError("HIGH_REASONING_TIME_BUDGET_EXCEEDED")
                payload=model_body(question,outputs,effort,tokens,proposal_context)
                before=time.monotonic()
                st,res=model120b.client(key,"/v1/chat/completions",payload,
                         timeout=min(900,max(80,int(budget_seconds-elapsed))))
                answer,completion,reasoning=extract_answer(st,res)
                seconds=round(time.monotonic()-before,2)
                outputs.append(answer)
                details.append({
                    "round":round_index+1,"model_answer":answer[:10000],
                    "seconds":seconds,"completion_tokens":completion,
                    "reported_reasoning_tokens":reasoning,
                    "http_status":st,
                })
                print(json.dumps({"round":round_index+1,"status":st,
                      "visible_output":bool(answer),"seconds":seconds,
                      "completion_tokens":completion},sort_keys=True),flush=True)
            result=apply_verifier(question,outputs[-1])
            proof=signed_record({
                "schema":SCHEMA,"model_id":model120b.ID,
                "model_repo":model120b.REPO,"model_revision":model120b.REV,
                "official_weight_sha256":model120b.SHA256,
                "model_size_bytes":model120b.SIZE_BYTES,
                "model_original_gpt6":False,
                "model_original_openai_gpt_oss_120b":True,
                "research_mode":research_mode or "direct_120b",
                "draft_proposal_receipt_sha256":proposal_receipt_sha,
                "external_open_model_proposal_applied":bool(proposal_context),
                "actual_local_120b_inference":True,
                "public_prompt_hash":hashlib.sha256(question.encode("utf8")).hexdigest(),
                "raw_user_prompt_saved":False,
                "reasoning_effort":effort,
                "requested_tokens_each_round":tokens,
                "rounds":rounds,
                "round_proofs":details,
                "selection":{
                    "model_last_answer":outputs[-1],
                    "final_answer":result["final_answer"],
                    "source":result["source"],
                    "verification":result["verification"],
                },
                "elapsed_total_seconds":round(time.monotonic()-start,2),
                "observed_utc":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "model_provider_credits_used":0,
                "paid_compute_allocated":False,
            })
            outfile.write_bytes(normalized(proof)+b"\n")
            print(json.dumps({"status":"OFFICIAL_120B_CHAT_GENERATED",
               "model":model120b.ID,"rounds":rounds,
               "receipt_sha256":proof["sha256"]},sort_keys=True),flush=True)
            return proof
        finally:
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--weights-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--rounds",type=int,choices=(1,2),default=1)
    parser.add_argument("--effort",choices=("low","medium","high"),default="high")
    parser.add_argument("--tokens",type=int,default=512)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--mode",choices=("max","ultra"),help="Use 20B draft evidence then official open 120B with configured high-effort budget")
    parser.add_argument("--proposal-receipt",type=Path,help="SHA-verified 20B source model evidence for the same public question")
    args=parser.parse_args(argv)
    if args.mode:
        preset=reasoning_profile(args.mode)
        args.rounds=preset["final_rounds"]
        args.tokens=preset["final_tokens_per_round"]
        args.effort=preset["final_effort"]
    check_limits(args.rounds,args.tokens,args.effort)
    if not args.execute:
        print(json.dumps({"status":"DRY_RUN","model":model120b.ID,
            "provider_model_credits":0,"inference":False,"rounds":args.rounds}))
        return 0
    question=check_prompt(os.getenv("MJ_PUBLIC_RESEARCH_QUESTION",""))
    proposal_text=""
    proposal_sha=None
    if args.mode:
        if args.proposal_receipt is None or not args.proposal_receipt.is_file():
            raise ValueError("MAX_ULTRA_REQUIRE_VERIFIED_20B_PROPOSAL")
        source=json.loads(args.proposal_receipt.read_text("utf8"))
        proposal_text=validate_proposal(source,question,args.mode)
        proposal_sha=source["sha256"]
    args.weights_dir.mkdir(parents=True,exist_ok=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    obs=model120b.local_resources(args.weights_dir)
    print(json.dumps({"phase":"compute_verified","ram_gib":obs["ram_gib"],
         "free_disk_gib":obs["disk_free_gib"]}),flush=True)
    loaded=model120b.retrieve_public_weights(args.weights_dir/model120b.FILENAME)
    print(json.dumps({"phase":"sha_verified","size":loaded["bytes"],
        "sha256":loaded["sha256"]}),flush=True)
    run(question,args.binary,args.weights_dir/model120b.FILENAME,args.out,
        rounds=args.rounds,tokens=args.tokens,effort=args.effort,
        proposal_context=proposal_text,proposal_receipt_sha=proposal_sha,
        research_mode=args.mode)
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,TimeoutError,subprocess.CalledProcessError) as exc:
        print(json.dumps({"status":"FAILED","type":type(exc).__name__,
              "reason":str(exc)[:150]}),file=sys.stderr)
        sys.exit(2)
