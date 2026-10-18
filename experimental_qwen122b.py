#!/usr/bin/env python3
"""One bounded attempt to run truly public Qwen3.5-122B-A10B IQ4_XS MoE GGUF.

Official Qwen Apache-2.0 open base + AesSedai third-party quantization.
All three GGUF parts are SHA-256 and revision pinned before llama.cpp loads.
No paid API, host privilege escalation, account tokens or private prompts.
Does NOT claim to be GPT-6, Claude or original model precision.
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
import shutil
import subprocess
import sys
import time
from urllib.parse import quote
from cloud_probe import signed_record, normalized, linux_memory_gib

SCHEMA="mj.open-qwen122b-mmap/1.0"
REPO="AesSedai/Qwen3.5-122B-A10B-GGUF"
REV="0af6191991b6e42d7057656e932fea0457f49911"
MODEL_ID="Qwen3.5-122B-A10B-IQ4_XS"
PORT=18939
PARTS=[
    {"name":"IQ4_XS/Qwen3.5-122B-A10B-IQ4_XS-00001-of-00003.gguf",
     "bytes":10_943_424,
     "sha256":"b5c8e0c6a0f0592d2283c868052e8c2885a735f36f9a2b88fdf42cdb05a396b5"},
    {"name":"IQ4_XS/Qwen3.5-122B-A10B-IQ4_XS-00002-of-00003.gguf",
     "bytes":49_928_180_096,
     "sha256":"c1143c2dc6b5ecf76c4acd1c44bbc01093a8daab1584123a9f8da7c823796de3"},
    {"name":"IQ4_XS/Qwen3.5-122B-A10B-IQ4_XS-00003-of-00003.gguf",
     "bytes":13_178_122_240,
     "sha256":"8dab1537cff890e81569c8cd7e460deca4df24d98ab8440c9825c4dc81c4dc00"},
]
PROMPT="Calculate 19 times 23. Answer only the integer."
MODEL_BYTES=sum(x["bytes"] for x in PARTS)
MIN_FREE_GIB=MODEL_BYTES/2**30+8
MIN_RAM_GIB=12

def url(part:dict)->str:
    if part not in PARTS:
        raise ValueError("PART_NOT_ALLOWLISTED")
    return f"https://huggingface.co/{REPO}/resolve/{REV}/"+quote(part["name"],safe="/")

def validate_resources(root:Path)->dict:
    ram=linux_memory_gib()
    free=shutil.disk_usage(root).free/2**30
    if ram is None or ram<MIN_RAM_GIB:
        raise RuntimeError("RAM_REQUIREMENTS_NOT_OBSERVED")
    if free<MIN_FREE_GIB:
        raise RuntimeError("FILE_SIZE_PLUS_DISK_RESERVE_NOT_AVAILABLE")
    return {"ram_gib":ram,"disk_free_gib":round(free,2),
            "disk_required_gib":round(MIN_FREE_GIB,2)}

def digest(path:Path)->tuple[int,str]:
    sha=hashlib.sha256()
    count=0
    with path.open("rb") as f:
        while chunk:=f.read(8*1024*1024):
            sha.update(chunk)
            count+=len(chunk)
    return count,sha.hexdigest()

def verify_part(part:dict, file:Path)->dict:
    size,hashval=digest(file)
    if size!=part["bytes"] or hashval!=part["sha256"]:
        raise ValueError("GGUF_PART_SHA256_OR_SIZE_INVALID")
    return {"filename":part["name"],"bytes":size,"sha256":hashval}

def download_part(part:dict,root:Path)->dict:
    name=part["name"]
    if not name.startswith("IQ4_XS/") or ".." in name or ":" in name:
        raise ValueError("INVALID_PINNED_FILE_PATH")
    target=root/name
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file():
        return verify_part(part,target)
    temp=target.with_suffix(".partial")
    temp.unlink(missing_ok=True)
    try:
        subprocess.run(["curl","--fail","--location","--silent","--show-error",
            "--retry","2","--retry-delay","4","--max-time","950",
            "--output",str(temp),url(part)],
            check=True,timeout=975)
        verified=verify_part(part,temp)
        os.replace(temp,target)
        return verified
    finally:
        temp.unlink(missing_ok=True)

def http(key:str,route:str,payload=None,timeout=10):
    if route not in ("/health","/v1/models","/v1/chat/completions"):
        raise ValueError("NON_ALLOWLISTED_LOCAL_API")
    c=http.client.HTTPConnection("127.0.0.1",PORT,timeout=timeout)
    headers={"Authorization":"Bearer "+key}
    if payload is not None:headers["Content-Type"]="application/json"
    try:
        c.request("POST" if payload is not None else "GET",route,
            body=json.dumps(payload,ensure_ascii=False).encode("utf8") if payload is not None else None,
            headers=headers)
        r=c.getresponse()
        data=r.read(1024*1024)
        return r.status,json.loads(data)
    finally:c.close()

def inference(binary:Path,root:Path,out:Path)->int:
    if not binary.is_file(): raise FileNotFoundError("INFERENCE_RUNTIME_MISSING")
    first=root/PARTS[0]["name"]
    if not first.is_file(): raise FileNotFoundError("GGUF_FIRST_SHARD_MISSING")
    key=secrets.token_urlsafe(48)
    env=dict(os.environ)
    env["LLAMA_API_KEY"]=key
    start=time.monotonic()
    log_path=out.parent/"qwen122-runtime.txt"
    with log_path.open("wb") as log:
        proc=subprocess.Popen([
            str(binary),"-m",str(first),"-a",MODEL_ID,
            "--host","127.0.0.1","--port",str(PORT),
            "--no-webui","--no-mmproj","--load-mode","mmap","--fit","off",
            "--no-repack","--cache-ram","0",
            "-c","1024","-b","64","-ub","32","-t","4","-np","1","-ngl","0"
        ],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        result={
            "schema":SCHEMA,"model_id":MODEL_ID,
            "upstream_base":"Qwen/Qwen3.5-122B-A10B",
            "quantization_repo":REPO,"quantization_revision":REV,
            "verified_parts":[{"name":p["name"],"sha256":p["sha256"],"bytes":p["bytes"]} for p in PARTS],
            "model_weight_size_bytes":MODEL_BYTES,
            "official_closed_model_clone":False,
            "original_full_precision_base":False,
            "real_inference":False,
            "new_spend_usd":0,"model_api_credits":0,
            "cpu_only":True,"api_http_status":None,
            "response_text":None,"model_loaded":False,
        }
        try:
            t=time.monotonic()
            for _ in range(150):
                if proc.poll() is not None:break
                try:
                    status,_=http(key,"/health",timeout=3)
                    if status==200:break
                except (OSError,TimeoutError,ValueError):
                    pass
                time.sleep(3)
            result["load_seconds"]=round(time.monotonic()-t,2)
            if proc.poll() is not None:
                result["reason_code"]="MODEL_PROCESS_EXITED"
                return 3
            status,models=http(key,"/v1/models")
            result["model_loaded"]=status==200 and MODEL_ID in [x.get("id") for x in models.get("data",[])]
            if not result["model_loaded"]:
                result["reason_code"]="MODEL_ID_NOT_VERIFIED"
                return 3
            t=time.monotonic()
            body={
              "model":MODEL_ID,
              "messages":[{"role":"system","content":"Answer only in the requested format, no explanation."},
                          {"role":"user","content":PROMPT}],
              "temperature":0,"max_tokens":128,
              "chat_template_kwargs":{"enable_thinking":False},
              "stream":False,
            }
            status,response=http(key,"/v1/chat/completions",body,timeout=400)
            answer=response.get("choices",[{}])[0].get("message",{}).get("content","") if status==200 else ""
            if not isinstance(answer,str):answer=""
            result.update({"inference_seconds":round(time.monotonic()-t,2),
                    "api_http_status":status,
                    "response_text":answer[:200],
                    "real_inference":bool(answer.strip()),
                    "correct_sample":answer.strip()=="437",
                    "generated_tokens":response.get("usage",{}).get("completion_tokens",0)})
            return 0 if result["real_inference"] else 3
        except (OSError,TimeoutError,ValueError,IndexError,KeyError):
            result["reason_code"]="INFERENCE_TIMEOUT_OR_FAILED"
            return 3
        finally:
            result["total_elapsed_seconds"]=round(time.monotonic()-start,2)
            result["process_exit_code"]=proc.poll()
            result["checked_utc"]=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            out.write_bytes(normalized(signed_record(result))+b"\n")
            proc.terminate()
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:proc.kill()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--weights-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if not a.execute:
        print(json.dumps({"status":"DRY_RUN","model":MODEL_ID,
           "part_count":len(PARTS),"total_bytes":MODEL_BYTES,
           "model_api_calls":0,"model_loaded":False}))
        return 0
    a.weights_dir.mkdir(parents=True,exist_ok=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"resources":validate_resources(a.weights_dir)}),flush=True)
    for part in PARTS:
        r=download_part(part,a.weights_dir)
        print(json.dumps({"phase":"verified_shard","filename":r["filename"],
                          "bytes":r["bytes"],"sha256":r["sha256"]}),flush=True)
    return inference(a.binary,a.weights_dir,a.out)

if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,RuntimeError,ValueError,subprocess.CalledProcessError,TimeoutError) as e:
        print(json.dumps({"status":"FAILED","reason":str(e)[:150],
            "error_type":type(e).__name__}),file=sys.stderr)
        sys.exit(2)
