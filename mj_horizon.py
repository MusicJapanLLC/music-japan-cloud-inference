#!/usr/bin/env python3
"""Music Japan Horizon: one command to use authentic public 120B+122B cloud models.

Uses owner's existing GitHub CLI login; never reads or copies credentials.
GitHub Actions inputs are public; ONLY submit public research questions.

start:   python mj_horizon.py start --public-question "Calculate 19 times 23." --public
status:  python mj_horizon.py status RUN_ID
collect: python mj_horizon.py collect RUN_ID --public-question "Calculate 19 times 23."
run:     python mj_horizon.py run --public-question "Calculate 19 times 23." --public
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import tempfile
import time

from cloud_probe import verify
from horizon_referee import decide,SCHEMA as RESULT_SCHEMA
from mj_ultra import gh, public_question, run_id

REPO="MusicJapanLLC/music-japan-cloud-inference"
WORKFLOW="horizon-federated-cloud.yml"
REF="feature/mj-beyond-120b-235b-397b-cloud-20261018"
ARTIFACTS={
    "oss120":("horizon-oss120-proof","horizon-oss120.json"),
    "qwen122":("horizon-qwen122-proof","horizon-qwen122.json"),
    "final":("horizon-actual-combined-answer","horizon-final.json")
}
RUN_PATTERN=re.compile(
    r"https://github[.]com/MusicJapanLLC/music-japan-cloud-inference/actions/runs/([0-9]{7,16})")

def dispatch(question:str,invoke=gh)->dict:
    q=public_question(question)
    output=invoke(["workflow","run",WORKFLOW,"--repo",REPO,"--ref",REF,
                   "-f","run_federation=true","-f","public_question="+q],timeout=100)
    match=RUN_PATTERN.search(output)
    if not match:raise ValueError("WORKFLOW_RUN_URL_NOT_RETURNED")
    return {"run_id":int(match[1]),"url":match[0],"public_question":True,
            "models":["gpt-oss-120b-MXFP4","Qwen3.5-122B-A10B-IQ4_XS"]}

def status(value:int,invoke=gh)->dict:
    rid=run_id(value)
    raw=invoke(["run","view",str(rid),"--repo",REPO,"--json",
            "databaseId,status,conclusion,url,name,headBranch,event"],timeout=35)
    result=json.loads(raw)
    if (int(result.get("databaseId") or 0)!=rid or
        result.get("headBranch")!=REF or result.get("event")!="workflow_dispatch"):
        raise ValueError("NOT_OUR_FEDERATION_WORKFLOW_RUN")
    if not str(result.get("name","")).startswith("Music Japan Horizon"):
        raise ValueError("WORKFLOW_NAME_INVALID")
    return {"run_id":rid,"status":result.get("status"),
            "conclusion":result.get("conclusion"),"url":result.get("url")}

def collect(value:int,question:str,invoke=gh)->dict:
    run=status(value,invoke)
    if run["status"]!="completed" or run["conclusion"]!="success":
        raise ValueError("MODEL_WORKFLOW_NOT_PROVEN_COMPLETED")
    question=public_question(question)
    with tempfile.TemporaryDirectory(prefix="mj-horizon-proofs-") as t:
        items={}
        for key,(artifact,filename) in ARTIFACTS.items():
            target=Path(t)/key
            target.mkdir(exist_ok=True)
            invoke(["run","download",str(run["run_id"]),"--repo",REPO,
                    "--name",artifact,"--dir",str(target)],timeout=150)
            filepath=target/filename
            if not filepath.is_file() or filepath.stat().st_size>2_000_000:
                raise ValueError("ARTIFACT_NOT_PRESENT_"+key)
            items[key]=json.loads(filepath.read_text("utf8"))
        for name,row in items.items():
            if not verify(row):raise ValueError("TAMPERED_MODEL_PROOF_"+name)
        checked=decide(question,{"oss120":items["oss120"],"qwen122":items["qwen122"]})
        stable_fields=(
            "schema","public_question_sha256","original_public_models",
            "source_receipt_sha256","model_answers","independent_checks",
            "tool_verified_model_count","selected_model","final_answer",
            "decision","model_credits_used","additional_cloud_spend",
            "model_count","inference_done_during_referee",
            "total_original_generation_seconds","model_concurrent_workers",
            "global_shared_ram_gib_claimed","dedicated_gpu_allocated",
        )
        if checked["schema"]!=RESULT_SCHEMA or any(
            checked.get(field)!=items["final"].get(field) for field in stable_fields
        ):
            raise ValueError("HASH_CHAIN_OR_DECISION_DIVERGENCE")
        return {
            "completed":True,"url":run["url"],
            "models":checked["original_public_models"],
            "final_answer":checked["final_answer"],
            "independent_validation":checked["decision"],
            "proof_sha256":checked["sha256"],
            "model_credits":0,"new_spend_usd":0,
            "machines":"four ephemeral GitHub Linux jobs, separate RAM, no allocated GPU",
        }

def watch(value:int,seconds:int=4500,invoke=gh,sleep=time.sleep,clock=time.monotonic)->dict:
    if not 30<=seconds<=5400:raise ValueError("INVALID_WATCH_DURATION")
    started=clock()
    while clock()-started<seconds:
        current=status(value,invoke)
        if current["status"]=="completed":
            if current["conclusion"]!="success":raise RuntimeError("HORIZON_RUN_FAILED")
            return current
        sleep(30)
    raise TimeoutError("CLOUD_JOB_WAIT_EXCEEDED")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for command in ("start","run","collect"):
        item=sub.add_parser(command)
        item.add_argument("--public-question",required=True)
        if command in ("start","run"):
            item.add_argument("--public",action="store_true",
                help="Confirm public GitHub Actions research prompt")
        if command=="collect":
            item.add_argument("run_id",type=int)
        if command=="run":
            item.add_argument("--wait",type=int,default=4500)
    item=sub.add_parser("status")
    item.add_argument("run_id",type=int)
    args=p.parse_args()
    if args.command=="status":
        print(json.dumps(status(args.run_id),ensure_ascii=False,indent=2))
        return
    if args.command=="collect":
        print(json.dumps(collect(args.run_id,args.public_question),ensure_ascii=False,indent=2))
        return
    if not args.public:
        p.error("PUBLIC_CONFIRMATION_REQUIRED")
    launched=dispatch(args.public_question)
    if args.command=="start":
        print(json.dumps(launched,ensure_ascii=False,indent=2))
        return
    watch(launched["run_id"],seconds=args.wait)
    print(json.dumps(collect(launched["run_id"],args.public_question),ensure_ascii=False,indent=2))

if __name__=="__main__":
    try:main()
    except (ValueError,RuntimeError,TimeoutError,OSError,json.JSONDecodeError) as err:
        print(json.dumps({"status":"ERROR","code":str(err)[:140]}),file=sys.stderr)
        sys.exit(2)
