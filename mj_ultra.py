#!/usr/bin/env python3
"""Music Japan Ultra — one command controls actual GitHub-hosted 20B/120B reasoning.

An authenticated owner's EXISTING GitHub CLI session dispatches a workflow
to FREE standard public Linux runners. This tool never handles GitHub
tokens, obtains credentials, or calls provider inference APIs.

Usage:
  python mj_ultra.py start --mode max --public-question "Calculate 314 times 29." --public
  python mj_ultra.py status RUN_ID
  python mj_ultra.py collect RUN_ID --mode max --public-question "Calculate 314 times 29."
  python mj_ultra.py run --mode max --public-question "Calculate 314 times 29." --public

PUBLIC ONLY: GitHub Actions inputs/results in this repository are visible to
others; never submit private data, company messages, customer files or keys.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from cloud_probe import verify
from verify_max_ultra_chain import validate as verify_chain
from reasoning_profiles import profile

REPO="MusicJapanLLC/music-japan-cloud-inference"
WORKFLOW="max-ultra-reasoning.yml"
REF="feature/mj-frontier-research-factory-20261018"
RUN_PATTERN=re.compile(
    r"https://github[.]com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/([0-9]{7,16})"
)
MAX_QUESTION_CHARS=3200
DEFAULT_WATCH_SECONDS=7200
ARTIFACT_NAMES={
    "draft":("mj-max-ultra-real-20b-proposal","20b-proposal.json"),
    "final":("mj-max-ultra-real-120b-final","120b-final.json"),
    "chain":("mj-max-ultra-independent-verification","max-ultra-evidence.json"),
}

def public_question(value:object)->str:
    if not isinstance(value,str):
        raise ValueError("PUBLIC_QUESTION_REQUIRED")
    q=value.strip()
    if len(q)<4 or len(q)>MAX_QUESTION_CHARS or "\x00" in q:
        raise ValueError("PUBLIC_QUESTION_INVALID")
    # This repository is PUBLIC: reject plausible credential/private-key paste.
    patterns=(r"-----BEGIN [A-Z ]*PRIVATE KEY-----",r"\bghp_[A-Za-z0-9]{15,}",
              r"\bgithub_pat_[A-Za-z0-9_]{15,}",
              r"\bsk-proj-[A-Za-z0-9_-]{15,}",
              r"\bsk-ant-[A-Za-z0-9_-]{15,}")
    if any(re.search(p,q,re.I) for p in patterns):
        raise ValueError("PUBLIC_QUESTION_APPEARS_TO_CONTAIN_CREDENTIALS")
    return q

def run_id(value:object)->int:
    if isinstance(value,bool):
        raise ValueError("RUN_ID_INVALID")
    s=str(value)
    if not re.fullmatch(r"[1-9][0-9]{6,15}",s):
        raise ValueError("RUN_ID_INVALID")
    return int(s)

def gh(args:list[str],timeout=45, invoke=subprocess.run)->str:
    # Only the installed official gh CLI is used; never a shell/powershell.
    cmd=["gh",*args]
    try:
        completed=invoke(cmd,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout,check=False)
    except FileNotFoundError as e:
        raise RuntimeError("GITHUB_CLI_NOT_INSTALLED") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("GITHUB_CLI_TIMEOUT") from e
    if completed.returncode:
        # Do not print raw stderr: it could contain a user's public question.
        raise RuntimeError("GITHUB_CLI_EXIT_"+str(completed.returncode))
    if not isinstance(completed.stdout,str):
        raise RuntimeError("GITHUB_CLI_OUTPUT_DECODE_FAILURE")
    return completed.stdout.strip()

def dispatch(mode:str,question:str,invoke=None)->dict:
    profile(mode)
    q=public_question(question)
    call=gh if invoke is None else invoke
    result=call(["workflow","run",WORKFLOW,"--repo",REPO,"--ref",REF,
                 "-f","execute=true","-f","mode="+mode,
                 "-f","public_question="+q],timeout=70)
    m=RUN_PATTERN.search(result)
    if not m:
        # Do not guess a run ID when concurrent workers may exist.
        raise RuntimeError("DISPATCH_RESPONSE_WITHOUT_CONFIRMED_RUN_URL")
    identifier=run_id(m.group(1))
    return {"run_id":identifier,"url":m.group(0),"mode":mode,
            "expected_workflow":WORKFLOW,"public_only":True}

def status(identifier:int,invoke=None)->dict:
    num=run_id(identifier)
    call=gh if invoke is None else invoke
    raw=call(["run","view",str(num),"--repo",REPO,"--json",
              "databaseId,status,conclusion,url,name,headBranch,event"],timeout=35)
    try:
        data=json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError("GITHUB_STATUS_NOT_JSON") from e
    if int(data.get("databaseId") or 0)!=num:
        raise ValueError("RUN_ID_MISMATCH")
    if data.get("headBranch")!=REF or data.get("event")!="workflow_dispatch":
        raise ValueError("RUN_BRANCH_OR_EVENT_MISMATCH")
    # Runtime output confirms conclusion; no assumptions based on URL existence.
    result={
        "run_id":num,
        "status":data.get("status"),
        "conclusion":data.get("conclusion"),
        "url":data.get("url"),
        "name":data.get("name"),
    }
    return result

def watch(identifier:int,seconds:int=DEFAULT_WATCH_SECONDS,interval=30,
          invoke=None,sleep=time.sleep,clock=time.monotonic)->dict:
    if not 30<=seconds<=14400 or not 3<=interval<=120:
        raise ValueError("WATCH_BUDGET_INVALID")
    start=clock()
    while clock()-start<seconds:
        result=status(identifier,invoke)
        if result["status"]=="completed":
            if result["conclusion"]!="success":
                raise RuntimeError("ULTRA_RUN_FINISHED_"+str(result["conclusion"]).upper())
            return result
        sleep(interval)
    raise TimeoutError("ULTRA_RUN_WAIT_BUDGET_EXPIRED")

def download_artifacts(identifier:int,folder:Path,invoke=None)->dict:
    num=run_id(identifier)
    call=gh if invoke is None else invoke
    result={}
    for key,(artifact,file) in ARTIFACT_NAMES.items():
        dest=folder/key
        dest.mkdir(parents=True,exist_ok=True)
        call(["run","download",str(num),"--repo",REPO,
              "--name",artifact,"--dir",str(dest)],timeout=110)
        path=dest/file
        if not path.is_file() or path.stat().st_size>2_000_000:
            raise ValueError("VERIFIED_ARTIFACT_FILE_MISSING_OR_TOO_LARGE_"+key.upper())
        try:
            result[key]=json.loads(path.read_text(encoding="utf8"))
        except json.JSONDecodeError as e:
            raise ValueError("ARTIFACT_NOT_JSON_"+key.upper()) from e
    return result

def prove(mode:str,question:str,artifacts:dict)->dict:
    profile(mode)
    q=public_question(question)
    try:
        proposal=artifacts["draft"]
        final=artifacts["final"]
        chain=artifacts["chain"]
    except (KeyError,TypeError) as e:
        raise ValueError("THREE_SIGNED_MODEL_ARTIFACTS_REQUIRED") from e
    for name,record in (("draft",proposal),("final",final),("chain",chain)):
        if not verify(record):
            raise ValueError("MODEL_PROVENANCE_HASH_FAILED_"+name.upper())
    expected=verify_chain(q,mode,proposal,final)
    if expected!=chain:
        raise ValueError("REAL_MODEL_CHAIN_PROOF_MISMATCH")
    answer=expected["final_answer"]
    if not isinstance(answer,str) or not answer.strip():
        raise ValueError("MODEL_FINAL_TEXT_MISSING")
    return {
        "mode":mode,"models":[expected["verified_public_model_20b"],expected["verified_public_model_120b"]],
        "verified":True,"real_model_api_credits_used":0,"new_spend_usd":0,
        "rounds":expected["actual_total_generation_rounds"],
        "actual_compute_seconds":expected["total_measured_compute_seconds"],
        "independent_check":expected["independent_question_grounded_verification"],
        "response":answer,
        "proof_sha256":expected["sha256"],
    }

def collect(identifier:int,mode:str,question:str,invoke=None)->dict:
    observed=status(identifier,invoke)
    if observed["status"]!="completed" or observed["conclusion"]!="success":
        raise ValueError("RUN_NOT_VERIFIED_COMPLETE_SUCCESS")
    with tempfile.TemporaryDirectory(prefix="mj-ultra-proof-") as temp:
        artifacts=download_artifacts(identifier,Path(temp),invoke)
        confirmed=prove(mode,question,artifacts)
        confirmed["run_id"]=run_id(identifier)
        confirmed["run_url"]=observed["url"]
        return confirmed

def cli(argv=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="action",required=True)
    for name in ("start","run","collect"):
        p=subs.add_parser(name)
        p.add_argument("--mode",choices=("max","ultra"),required=True)
        p.add_argument("--public-question",required=True)
        if name in ("start","run"):
            p.add_argument("--public",action="store_true",
                help="Confirm input may be visible on public GitHub Actions")
        if name=="run":
            p.add_argument("--wait-seconds",type=int,default=DEFAULT_WATCH_SECONDS)
        if name=="collect":
            p.add_argument("id",type=int)
    st=subs.add_parser("status")
    st.add_argument("id",type=int)
    args=parser.parse_args(argv)
    if args.action in ("start","run"):
        if not args.public:
            parser.error("Explicit --public consent is required for PUBLIC GitHub Actions prompts")
        job=dispatch(args.mode,args.public_question)
        print(json.dumps(job,ensure_ascii=False,sort_keys=True),flush=True)
        if args.action=="start":
            return 0
        watch(job["run_id"],seconds=args.wait_seconds)
        data=collect(job["run_id"],args.mode,args.public_question)
    elif args.action=="status":
        data=status(args.id)
    else:
        data=collect(args.id,args.mode,args.public_question)
    print(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2))
    return 0

if __name__=="__main__":
    try:sys.exit(cli())
    except (ValueError,RuntimeError,TimeoutError,OSError) as e:
        print(json.dumps({"status":"ERROR","code":str(e)[:120]}),file=sys.stderr)
        sys.exit(2)
