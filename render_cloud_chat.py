#!/usr/bin/env python3
"""Safe GitHub Actions Summary renderer for verified public-model inference."""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path
from cloud_probe import verify
from cloud_owner_chat import SCHEMA, CFG

def read_and_verify(path:Path)->dict:
    record=json.loads(path.read_text("utf8"))
    if not verify(record) or record.get("schema")!=SCHEMA:
        raise ValueError("INTEGRITY_FAILURE")
    if record.get("model_id")!=CFG["model"] or record.get("model_weight_sha256")!=CFG["sha256"]:
        raise ValueError("MODEL_IDENTITY_FAILURE")
    if record.get("paid_compute_created") is not False or record.get("external_model_api_calls")!=0:
        raise ValueError("NONLOCAL_OR_PAID_ACCESS")
    if record.get("real_local_linux_model_inference") is not True:
        raise ValueError("INFERENCE_NOT_VERIFIED")
    selected=record.get("selection",{})
    if not isinstance(selected,dict) or not isinstance(selected.get("final_answer"),str) or not selected["final_answer"].strip():
        raise ValueError("FINAL_TEXT_MISSING")
    return record

def render(record:dict)->str:
    selected=record["selection"]
    answer=selected["final_answer"]
    if len(answer)>30000:
        raise ValueError("FINAL_TEXT_TOO_LONG")
    rows=record["rounds_evidence"]
    return (
        "## Music Japan: Real Cloud LLM Answer\n\n"
        "Model: **"+html.escape(record["model_id"])+"**\n\n"
        "Thinking effort: **"+html.escape(record["reasoning_effort"])+"**\n\n"
        "Actual reasoning rounds: **"+str(len(rows))+"**\n\n"
        "Total inference time: **"+str(record["elapsed_total_seconds"])+" seconds**\n\n"
        "Independent verifier: **"+html.escape(str(selected.get("verification",{}).get("status","ABSTAIN")))+"**\n\n"
        "External proprietary model API credits: **0**\n\n"
        "Receipt SHA-256: "+html.escape(record["sha256"])+"\n\n"
        "### Final answer\n\n<pre>"+html.escape(answer)+"</pre>\n\n"
        "_Public research inputs only. This GitHub report is not a confidential messaging system._\n"
    )

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    data=read_and_verify(args.receipt)
    formatted=render(data)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(formatted,encoding="utf8")
    else:
        path=os.getenv("GITHUB_STEP_SUMMARY")
        if not path:raise ValueError("SUMMARY_FILE_NOT_SET")
        with Path(path).open("a",encoding="utf8") as f:f.write(formatted)
    print(json.dumps({"summary_ready":True,"model":data["model_id"],
        "rounds":data["rounds"],"receipt_sha256":data["sha256"]},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
