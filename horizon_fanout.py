#!/usr/bin/env python3
"""Music Japan Horizon: real 120B and 122B models, two independent Linux machines.

Each stage executes ORIGINAL PUBLIC open-weight inference with SHA-pinned GGUF.
No paid model API, no unauthorized GPU, and no input from Music Japan private DB.

Inputs are only PUBLIC synthetic research questions in GitHub Actions.
A separate referee consumes both hashed model receipts after real inference.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

from cloud_probe import normalized, signed_record
import experimental_gptoss120b as oss
import experimental_qwen122b as qwen
from mj_ultra import public_question

SCHEMA="mj.horizon.two-frontier-workers/1.0"
PORT=18961
MAX_OUTPUT_TOKENS={"oss120":512,"qwen122":256}
PROFILES=("oss120","qwen122")
DEFAULT_QUESTION="Compute ((72 / 8) + 17) times 6. Reply only with the integer."

def config(profile:str) -> dict:
    if profile=="oss120":
        return {"id":oss.ID,"revision":oss.REV,"repo":oss.REPO,
                "quantization":"MXFP4","sha256":oss.SHA256,"bytes":oss.SIZE_BYTES,
                "system":"You are a careful long-reasoning model. Verify assumptions, then give only the requested answer.",
                "max_tokens":MAX_OUTPUT_TOKENS[profile], "load_wait_seconds":480}
    if profile=="qwen122":
        return {"id":qwen.MODEL_ID,"revision":qwen.REV,"repo":qwen.REPO,
                "quantization":"IQ4_XS",
                "sha256":hashlib.sha256("|".join(p["sha256"] for p in qwen.PARTS).encode("ascii")).hexdigest(),
                "bytes":qwen.MODEL_BYTES,
                "system":"Solve independently. Carefully test computations and give only the requested final answer.",
                "max_tokens":MAX_OUTPUT_TOKENS[profile],"load_wait_seconds":480}
    raise ValueError("ONLY_TWO_PINNED_PUBLIC_FRONTIER_PROFILES")

def weight_prepare(profile:str,root:Path)->dict:
    cfg=config(profile)
    if profile=="oss120":
        obs=oss.local_resources(root)
        proof=oss.retrieve_public_weights(root/oss.FILENAME)
        if proof["sha256"]!=oss.SHA256 or proof["bytes"]!=oss.SIZE_BYTES:
            raise ValueError("ORIGINAL_120B_WEIGHTS_NOT_VERIFIED")
        return {"resource":obs,"model_file":root/oss.FILENAME,
                "parts":[{"name":oss.FILENAME,"bytes":proof["bytes"],"sha256":proof["sha256"]}]}
    obs=qwen.validate_resources(root)
    verified=[]
    for p in qwen.PARTS:
        proof=qwen.download_part(p,root)
        if proof["sha256"]!=p["sha256"] or proof["bytes"]!=p["bytes"]:
            raise ValueError("QWEN122B_SHARD_NOT_VERIFIED")
        verified.append(proof)
    return {"resource":obs,"model_file":root/qwen.PARTS[0]["name"],"parts":verified}

def local_request(key:str,path:str,body=None,timeout:int=10) -> tuple[int,dict]:
    if path not in ("/health","/v1/models","/v1/chat/completions"):
        raise ValueError("UNKNOWN_LOCAL_API_PATH")
    conn=http.client.HTTPConnection("127.0.0.1",PORT,timeout=timeout)
    headers={"Authorization":"Bearer "+key}
    if body is not None:headers["Content-Type"]="application/json"
    try:
        conn.request("POST" if body is not None else "GET",path,
          body=json.dumps(body,ensure_ascii=False).encode("utf8") if body is not None else None,
          headers=headers)
        response=conn.getresponse()
        content=response.read(2_000_000)
        if len(content)>=2_000_000:raise ValueError("LOCAL_RESPONSE_TOO_LARGE")
        return response.status,json.loads(content)
    finally:conn.close()

def inference_request(profile:str,question:str,extra_context:str="") -> dict:
    cfg=config(profile)
    question=public_question(question)
    query=question
    if extra_context:
        query+=("\n\nA prior model suggested this answer, which might be false. "
                "Verify independently and return a corrected final answer:\n"+extra_context[:1000])
    payload={
        "model":cfg["id"],
        "messages":[{"role":"system","content":cfg["system"]},
                    {"role":"user","content":query}],
        "max_tokens":cfg["max_tokens"],
        "stream":False,
        "temperature":1.0 if profile=="oss120" else 0.0,
    }
    if profile=="oss120":
        payload["reasoning_effort"]="high"
        payload["chat_template_kwargs"]={"reasoning_effort":"high"}
    else:
        payload["chat_template_kwargs"]={"enable_thinking":False}
    return payload

def run(profile:str,binary:Path,weights:Path,question:str,out:Path)->dict:
    cfg=config(profile)
    question=public_question(question)
    if not binary.is_file() or not weights.is_file():
        raise FileNotFoundError("VERIFIED_BIN_OR_MODEL_FILE_NOT_FOUND")
    out.parent.mkdir(parents=True,exist_ok=True)
    token=secrets.token_urlsafe(48)
    env=dict(os.environ)
    env["LLAMA_API_KEY"]=token
    begun=time.monotonic()
    with (out.parent/"horizon-runtime.log").open("wb") as logs:
        p=subprocess.Popen([
            str(binary),"-m",str(weights),"-a",cfg["id"],
            "--host","127.0.0.1","--port",str(PORT),
            "--no-webui","--no-mmproj","--load-mode","mmap","--fit","off",
            "--no-repack","--cache-ram","0","-c","1280","-b","64",
            "-ub","32","-t","4","-np","1","-ngl","0"
        ],env=env,stdin=subprocess.DEVNULL,stdout=logs,stderr=subprocess.STDOUT)
        try:
            loaded=False
            while time.monotonic()-begun<cfg["load_wait_seconds"]:
                if p.poll() is not None:raise RuntimeError("MODEL_PROCESS_EXITED_BEFORE_READY")
                try:
                    state,_=local_request(token,"/health",timeout=3)
                    if state==200:loaded=True;break
                except (OSError,TimeoutError,ValueError,json.JSONDecodeError):
                    pass
                time.sleep(3)
            if not loaded:raise TimeoutError("MODEL_NOT_READY_ON_FREE_VM")
            state,models=local_request(token,"/v1/models",timeout=10)
            if state!=200 or cfg["id"] not in [v.get("id") for v in models.get("data",[]) if isinstance(v,dict)]:
                raise ValueError("ORIGINAL_MODEL_ID_NOT_VERIFIED")
            start=time.monotonic()
            body=inference_request(profile,question)
            code,response=local_request(token,"/v1/chat/completions",body,timeout=1250)
            if code!=200 or not isinstance(response,dict):
                raise RuntimeError("MODEL_HTTP_ERROR")
            messages=response.get("choices") or []
            ans=messages[0].get("message",{}).get("content") if messages else None
            if not isinstance(ans,str) or not ans.strip():
                raise RuntimeError("EMPTY_MODEL_FINAL_TEXT")
            usage=response.get("usage",{})
            if not isinstance(usage,dict):usage={}
            result=signed_record({
                "schema":SCHEMA,"profile":profile,"model_id":cfg["id"],
                "model_repo":cfg["repo"],"model_revision":cfg["revision"],
                "weights_integrity_digest":cfg["sha256"],
                "weight_total_bytes":cfg["bytes"],
                "public_question_sha256":hashlib.sha256(question.encode("utf8")).hexdigest(),
                "actual_original_public_weight_inference":True,
                "proprietary_vendor_model":False,
                "reasoning_effort":"high" if profile=="oss120" else "format_only_no_thinking",
                "model_answer":ans.strip()[:12000],
                "model_answer_present":True,
                "model_generated_tokens":int(usage.get("completion_tokens") or 0),
                "http_status":code,
                "model_load_seconds":round(start-begun,2),
                "generation_seconds":round(time.monotonic()-start,2),
                "total_seconds":round(time.monotonic()-begun,2),
                "github_linux_cpu_ephemeral":True,
                "paid_cloud_jobs":0,"external_llm_model_credits":0,
                "question_in_receipt":False,
                "observed_utc":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            })
            out.write_bytes(normalized(result)+b"\n")
            print(json.dumps({"model_id":cfg["id"],"verified_real_output":True,
               "answer_present":True,"elapsed_seconds":result["total_seconds"],
               "receipt_sha256":result["sha256"],"llm_credits":0}),flush=True)
            return result
        finally:
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill()

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile",choices=PROFILES,required=True)
    ap.add_argument("--binary",type=Path,required=True)
    ap.add_argument("--weights-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--execute",action="store_true")
    args=ap.parse_args(argv)
    if not args.execute:
        print(json.dumps({"state":"DRY_RUN","model":config(args.profile)["id"],"model_api_credits":0}))
        return 0
    question=public_question(os.environ.get("MJ_PUBLIC_RESEARCH_QUESTION",DEFAULT_QUESTION))
    args.weights_dir.mkdir(parents=True,exist_ok=True)
    obs=weight_prepare(args.profile,args.weights_dir)
    print(json.dumps({"phase":"WEIGHTS_VERIFIED","model":config(args.profile)["id"],
                      "shards":len(obs["parts"]),"disk_free_before_model_gib":obs["resource"]["disk_free_gib"]}),flush=True)
    run(args.profile,args.binary,obs["model_file"],question,args.out)
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except (ValueError,OSError,RuntimeError,TimeoutError,subprocess.CalledProcessError) as err:
        print(json.dumps({"error_code":type(err).__name__,"message":str(err)[:160]}),file=sys.stderr)
        sys.exit(2)
