#!/usr/bin/env python3
"""Acceptance-test a real public Fable-inspired model via three LOCAL llama.cpp API surfaces.

Zero proprietary model API credits; test ONLY 127.0.0.1 on an ephemeral free
GitHub standard Linux runner. Requires explicit --execute for model download and
inference. Distinguishes "HTTP endpoint accepted" from "model produced a reply".
"""
from __future__ import annotations
import argparse
import datetime as dt
import http.client
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from cloud_probe import normalized, signed_record
from cloud_infer_frontier import MODEL_MAP, available, download

VERSION="mj.api-reconstruction-acceptance/1.0"
PROFILE="fable-reconstruction-9b"
MODEL=MODEL_MAP[PROFILE]["model"]
PORT=18887
MODELS_PATH="/v1/models"
CASE_TEXT="Return only the word MUSICJAPAN, in uppercase, with no punctuation."

CASES=[
    ("openai_chat", "/v1/chat/completions", {
        "model": MODEL,
        "messages": [{"role":"user", "content":CASE_TEXT}],
        "temperature":0,
        "max_tokens":128,
        "chat_template_kwargs":{"enable_thinking":False},
        "stream":False,
    }),
    ("anthropic_messages", "/v1/messages", {
        "model":MODEL,
        "max_tokens":128,
        "system":"Follow the formatting instruction, respond with the final answer only.",
        "messages":[{"role":"user","content":CASE_TEXT}],
        "stream":False,
    }),
    ("openai_responses", "/v1/responses", {
        "model":MODEL,
        "input":CASE_TEXT,
        "max_output_tokens":128,
        "temperature":0,
        "stream":False,
    }),
]

def body_text(label:str,payload:dict)->str:
    """Parse model output without inventing a result or silently swallowing no text."""
    if not isinstance(payload,dict):
        return ""
    if label == "openai_chat":
        choices=payload.get("choices",[])
        if not isinstance(choices,list) or not choices:
            return ""
        msg=choices[0].get("message",{})
        content=msg.get("content","") if isinstance(msg,dict) else ""
        return content if isinstance(content,str) else ""
    if label == "anthropic_messages":
        blocks=payload.get("content",[])
        if not isinstance(blocks,list):
            return ""
        return "".join(block.get("text","") for block in blocks
                       if isinstance(block,dict) and block.get("type")=="text"
                       and isinstance(block.get("text"),str))
    if label == "openai_responses":
        if isinstance(payload.get("output_text"),str):
            return payload["output_text"]
        blocks=payload.get("output",[])
        if not isinstance(blocks,list):
            return ""
        return "".join(content.get("text","")
                       for item in blocks if isinstance(item,dict)
                       for content in item.get("content",[])
                       if isinstance(content,dict)
                       and content.get("type")=="output_text"
                       and isinstance(content.get("text"),str))
    return ""

def get_http(key:str,method:str,path:str,body=None,timeout=12):
    if path not in (MODELS_PATH,"/health",*[c[1] for c in CASES]):
        raise ValueError("UPSTREAM_NOT_ALLOWLISTED")
    headers={"Authorization":"Bearer "+key}
    if body is not None:
        headers["Content-Type"]="application/json"
    conn=http.client.HTTPConnection("127.0.0.1",PORT,timeout=timeout)
    try:
        conn.request(method,path,body=json.dumps(body,ensure_ascii=False).encode("utf8") if body else None,
                     headers=headers)
        res=conn.getresponse()
        raw=res.read(2_000_000)
        if len(raw)>=2_000_000:
            raise ValueError("UPSTREAM_BODY_TOO_LARGE")
        return res.status,json.loads(raw)
    finally:
        conn.close()

def evaluate_one(label:str,status:int,data:dict,seconds:float)->dict:
    answer=body_text(label,data).strip()
    return {"api":label,"http_status":status,
            "response_text":answer[:200],
            "model_generated_visible_text":bool(answer),
            "exact_instruction_pass":answer=="MUSICJAPAN",
            "latency_seconds":round(seconds,2),
            "identity_not_original_anthropic":True}

def start_server(exe:Path,weights:Path,log:Path,key:str):
    env=dict(os.environ)
    env["LLAMA_API_KEY"]=key
    return subprocess.Popen(
        [str(exe),"-m",str(weights),"-a",MODEL,"--host","127.0.0.1",
         "--port",str(PORT),"-c","2048","--no-webui",
         "--no-mmproj","--cache-ram","0","-t","4","-ngl","0","-np","1"],
        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
    )

def actual_acceptance(exe:Path,weights:Path,out:Path)->int:
    if not exe.is_file():
        raise FileNotFoundError("LLAMA_SERVER_NOT_FOUND")
    out.parent.mkdir(parents=True,exist_ok=True)
    key=secrets.token_urlsafe(40)
    with (out.parent/"server.log").open("wb") as log:
        child=start_server(exe,weights,log,key)
        try:
            started=time.monotonic()
            while time.monotonic()-started<180:
                if child.poll() is not None:
                    raise RuntimeError("SERVER_EXITED_BEFORE_READY")
                try:
                    status,_=get_http(key,"GET","/health",timeout=3)
                    if status==200:
                        break
                except (OSError,TimeoutError,ValueError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("SERVER_LOADING_TIMEOUT")
            status,info=get_http(key,"GET",MODELS_PATH)
            listed=[x.get("id") for x in info.get("data",[]) if isinstance(x,dict)]
            if status!=200 or MODEL not in listed:
                raise RuntimeError("WRONG_REAL_MODEL_OR_RESPONSE")
            results=[]
            for label,path,body in CASES:
                start=time.monotonic()
                try:
                    http_status,result=get_http(key,"POST",path,body,timeout=140)
                except (OSError,TimeoutError,ValueError,json.JSONDecodeError):
                    http_status,result=0,{}
                item=evaluate_one(label,http_status,result,time.monotonic()-start)
                results.append(item)
                print(json.dumps({"api":label,"status":item["http_status"],
                                  "visible_text":item["model_generated_visible_text"],
                                  "exact_instruction":item["exact_instruction_pass"]},
                                 ensure_ascii=False),flush=True)
            proof=signed_record({
                "schema":VERSION,
                "observed_utc":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "model":MODEL,
                "model_repo":MODEL_MAP[PROFILE]["repo"],
                "model_revision":MODEL_MAP[PROFILE]["revision"],
                "gguf_sha256":MODEL_MAP[PROFILE]["sha256"],
                "original_anthropic_model":False,
                "actual_model_loaded":True,
                "backend":"llama.cpp/127.0.0.1",
                "external_provider_inference_calls":0,
                "new_spend_usd":0,
                "public_http_endpoint_exposed":False,
                "api_tests":results,
                "supported_with_visible_text":[x["api"] for x in results
                                               if x["http_status"]==200
                                               and x["model_generated_visible_text"]],
                "full_compatibility_with_upstream_claimed":False,
            })
            out.write_bytes(normalized(proof)+b"\n")
            print(json.dumps({"proof_sha256":proof["sha256"],
                              "successful_endpoints":len(proof["supported_with_visible_text"]),
                              "total_endpoints":len(CASES)}),flush=True)
            # At least actual Chat Completions text is required to report usable inference.
            if results[0]["http_status"]!=200 or not results[0]["model_generated_visible_text"]:
                return 3
            return 0
        finally:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--weights-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if not args.execute:
        print(json.dumps({"status":"DRY_RUN","model":MODEL,
                          "api_tests":[x[0] for x in CASES],
                          "paid_calls":0,"loaded":False}),flush=True)
        return 0
    args.weights_dir.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"hardware_gate":available(MODEL_MAP[PROFILE],args.weights_dir)}),flush=True)
    weights=args.weights_dir/MODEL_MAP[PROFILE]["filename"]
    proof=download(MODEL_MAP[PROFILE],weights)
    print(json.dumps({"weight_sha_verified":proof["sha256"]==MODEL_MAP[PROFILE]["sha256"],
                      "bytes":proof["bytes"]}),flush=True)
    return actual_acceptance(args.binary,weights,args.out)

if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,TimeoutError,subprocess.CalledProcessError) as exc:
        print(json.dumps({"error_type":type(exc).__name__,"error":str(exc)[:120]}),file=sys.stderr)
        sys.exit(2)
