#!/usr/bin/env python3
"""Experimental free GitHub Linux mmap load for official open-weight gpt-oss-120b.

Explicit --execute only. This is a HARDWARE FEASIBILITY PROBE, not a promise of
GPU acceleration, fast inference, proprietary GPT-6, or permanent model hosting.

Official OpenAI model -> trusted ggml-org public GGUF. Pinned entire bytes, HF
revision, SHA-256. No vendor inference API, no credentials, no paid cloud.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import os
import re
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

from cloud_probe import normalized, signed_record, linux_memory_gib

SCHEMA = "mj.oss120b-mmap-proof/1.0"
ID = "gpt-oss-120b-MXFP4"
REPO = "ggml-org/gpt-oss-120b-GGUF"
REV = "238abdd290bb874b90a5da1b4549881b7d05c091"
FILENAME = "gpt-oss-120b-MXFP4.gguf"
SIZE_BYTES = 63_387_346_208
SHA256 = "582bd40f6886200101f4c4ed9f25f3fe80cc14c86e9e2b37746cd8904a0c622d"
SOURCE = f"https://huggingface.co/{REPO}/resolve/{REV}/{FILENAME}"
MIN_DISK_FREE_GIB = SIZE_BYTES/(1024**3)+8
MIN_RAM_GIB = 12
PORT = 18895
LOAD_LIMIT_SECONDS = 480
MAX_DIAGNOSTIC_BYTES = 128 * 1024

def local_resources(folder:Path)->dict:
    disk = shutil.disk_usage(folder)
    ram = linux_memory_gib()
    record={"ram_gib":ram, "disk_free_gib":round(disk.free/2**30,2),
            "required_min_free_disk_gib":round(MIN_DISK_FREE_GIB,2),
            "disk_total_gib":round(disk.total/2**30,2)}
    if ram is None or ram < MIN_RAM_GIB:
        raise RuntimeError("RUNNER_RAM_BELOW_MINIMUM")
    if disk.free/2**30 < MIN_DISK_FREE_GIB:
        raise RuntimeError("RUNNER_DISK_BELOW_MODEL_PLUS_RESERVE")
    return record

def exact_digest(path:Path)->tuple[int,str]:
    h=hashlib.sha256()
    size=0
    with path.open("rb") as fp:
        while data:=fp.read(8*1024*1024):
            size+=len(data);h.update(data)
    return size,h.hexdigest()

def verify_weights(path:Path)->dict:
    size, sha=exact_digest(path)
    if size!=SIZE_BYTES or sha!=SHA256:
        raise ValueError("WEIGHTS_INTEGRITY_MISMATCH")
    return {"source":SOURCE,"revision":REV,"bytes":size,"sha256":sha}

def retrieve_public_weights(out:Path)->dict:
    if out.exists():
        return verify_weights(out)
    partial=out.with_suffix(".part")
    partial.unlink(missing_ok=True)
    try:
        subprocess.run([
            "curl","--fail","--location","--silent","--show-error",
            "--retry","1","--max-time","1200",
            "--output",str(partial),SOURCE
        ],check=True,timeout=1210)
        proof=verify_weights(partial)
        os.replace(partial,out)
        return proof
    finally:
        partial.unlink(missing_ok=True)

def client(key:str,path:str,body=None,timeout:int=10):
    if path not in ("/health","/v1/models","/v1/chat/completions"):
        raise ValueError("API_PATH_NOT_ALLOWED")
    c=http.client.HTTPConnection("127.0.0.1",PORT,timeout=timeout)
    headers={"Authorization":"Bearer "+key}
    if body is not None: headers["Content-Type"]="application/json"
    try:
        c.request("POST" if body is not None else "GET",path,
            body=json.dumps(body,ensure_ascii=False).encode("utf8") if body is not None else None,
            headers=headers)
        res=c.getresponse()
        text=res.read(500_000)
        return res.status,json.loads(text)
    finally:
        c.close()

def cgroup_memory_events(path:Path=Path("/sys/fs/cgroup/memory.events")) -> dict:
    """Safe numeric counters only; no host files/secrets in results."""
    try:
        data={}
        for line in path.read_text(encoding="utf8").splitlines():
            key,space,value=line.partition(" ")
            if space and key in ("oom","oom_kill","max","high"):
                data[key]=max(0,int(value.strip()))
        return data
    except (OSError,ValueError):
        return {}

def diagnose_runtime(logfile:Path)->dict:
    try:
        with logfile.open("rb") as f:
            f.seek(0,2)
            f.seek(max(0,f.tell()-MAX_DIAGNOSTIC_BYTES))
            text=f.read(MAX_DIAGNOSTIC_BYTES).decode("utf8","replace").lower()
        patterns=(("oom",r"out of memory|oom killer|cannot allocate memory"),
                  ("allocation",r"failed to allocate|allocat(e|ion) failed"),
                  ("mmap",r"mmap|load.mode"),
                  ("load_error",r"error loading model|failed to load model|load_model.+failed"),
                  ("unsupported",r"unsupported|not implemented"))
        return {key:bool(re.search(pattern,text)) for key,pattern in patterns}
    except OSError:
        return {"log_unavailable":True}

def infer_once(binary:Path,model:Path,out:Path)->int:
    if not binary.is_file():raise FileNotFoundError("LLAMA_SERVER_BINARY_MISSING")
    key=secrets.token_urlsafe(40)
    env=dict(os.environ)
    env["LLAMA_API_KEY"]=key
    output_log=out.parent/"runtime.txt"
    with output_log.open("wb") as log:
        start=time.monotonic()
        memory_before=cgroup_memory_events()
        proc=subprocess.Popen([
            str(binary),"-m",str(model),"-a",ID,
            "--host","127.0.0.1","--port",str(PORT),
            "--no-webui","--no-mmproj","--cache-ram","0",
            "--load-mode","mmap","--fit","off","--no-repack",
            "-c","512","-b","64","-ub","32","-t","4","-np","1","-ngl","0"
        ],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        result={"schema":SCHEMA,"model":ID,"source_repo":REPO,"revision":REV,
                "weights_sha256":SHA256,"paid_compute":False,
                "real_model_api_credits":0,"original_gpt6_or_anthropic":False,
                "runner_memory_gib":linux_memory_gib(),"execution":"attempted",
                "model_loaded":False,"response_observed":False}
        try:
            ready=False
            for _ in range(LOAD_LIMIT_SECONDS // 3):
                if proc.poll() is not None:break
                try:
                    status,_=client(key,"/health",timeout=3)
                    if status==200:
                        ready=True
                        break
                except (OSError,ValueError,TimeoutError):
                    pass
                time.sleep(3)
            result["start_seconds"]=round(time.monotonic()-start,2)
            if not ready:
                result["status"]="LOAD_UNVERIFIED"
                result["process_exit_code"]=proc.poll()
                return 3
            st, m=client(key,"/v1/models",timeout=10)
            result["model_loaded"]=bool(st==200 and ID in
                                    [x.get("id") for x in m.get("data",[])])
            if not result["model_loaded"]:
                result["status"]="MODEL_ID_MISMATCH"
                return 3
            prompt={"model":ID,"messages":[
              {"role":"user","content":"Calculate 19 times 23. Reply with just the number."}
             ],"max_tokens":128,"reasoning_effort":"low",
             "chat_template_kwargs":{"reasoning_effort":"low"},
             "temperature":1.0,"stream":False}
            sent=time.monotonic()
            try:
                code, body=client(key,"/v1/chat/completions",prompt,timeout=300)
                answer=body.get("choices",[{}])[0].get("message",{}).get("content","") if code==200 else ""
                if not isinstance(answer,str): answer=""
                result.update({"status":"INFERENCE_FINISHED" if answer else "NO_VISIBLE_ANSWER",
                    "http_status":code,"answer":answer[:160],
                    "response_observed":bool(answer.strip()),
                    "output_tokens":body.get("usage",{}).get("completion_tokens",0),
                    "response_seconds":round(time.monotonic()-sent,2),
                    "correct_math":answer.strip()=="437"})
                return 0 if result["response_observed"] else 3
            except (OSError,ValueError,TimeoutError,IndexError,KeyError):
                result["status"]="INFERENCE_TIMED_OUT_OR_FAILED"
                result["response_seconds"]=round(time.monotonic()-sent,2)
                return 3
        finally:
            result["total_seconds"]=round(time.monotonic()-start,2)
            result["process_exit_code"]=proc.poll()
            result["runtime_log_classification"]=diagnose_runtime(output_log)
            memory_after=cgroup_memory_events()
            result["cgroup_memory_event_delta"]={key:max(0,v-memory_before.get(key,0))
                for key,v in memory_after.items()}
            out.write_bytes(normalized(signed_record(result))+b"\n")
            proc.terminate()
            try:proc.wait(timeout=8)
            except subprocess.TimeoutExpired:proc.kill()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    if not args.execute:
        print(json.dumps({"status":"DRY_RUN","source":SOURCE,"model_bytes":SIZE_BYTES,
            "model_sha256":SHA256,"proprietary_api_credit_use":0,
            "model_inference":False},sort_keys=True))
        return 0
    args.weights.mkdir(parents=True,exist_ok=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"phase":"resource_guard","obs":local_resources(args.weights)}),flush=True)
    check=retrieve_public_weights(args.weights/FILENAME)
    print(json.dumps({"phase":"file_sha_verified","bytes":check["bytes"],"sha256":check["sha256"]}),flush=True)
    return infer_once(args.binary,args.weights/FILENAME,args.out)

if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError,TimeoutError) as e:
        print(json.dumps({"status":"FAILED","code":type(e).__name__,
                          "reason":str(e)[:130]}),file=sys.stderr)
        sys.exit(2)
