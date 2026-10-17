#!/usr/bin/env python3
"""Pinned 9B/27B/35B-A3B actual inference benchmark on free public GitHub Linux runners.

This uses open Apache-2.0 base-model weights, not Anthropic proprietary weights.
No vendor model API calls, secrets, paid compute, or cloud credential handling.
"""
from __future__ import annotations
import argparse
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
import urllib.parse
from cloud_probe import linux_memory_gib, normalized, signed_record

SCHEMA = "mj.cloud-frontier-benchmark/1.0.0"
MODEL_MAP = {
    "baseline-9b": {
        "repo": "gguf/Qwen3.5-9B-GGUF",
        "revision": "dc40af5d4666acd6ce1c1dd71ec61cee2f1c966e",
        "filename": "qwen3.5-9b-Q3_K_M.gguf",
        "bytes": 5_178_748_960,
        "sha256": "24fccd8416fcdca792f939de4a2240eb57795b0f7a9ba92e62398eab3ec5c11e",
        "model": "Qwen3.5-9B-Q3_K_M",
        "size_b": 9,
        "min_ram_gib": 12,
    },
    "dense-27b": {
        "repo": "bartowski/Qwen_Qwen3.5-27B-GGUF",
        "revision": "d7b113c40283f4d99f4eb0ec20d126ad653cc736",
        "filename": "Qwen_Qwen3.5-27B-IQ2_M.gguf",
        "bytes": 11_085_693_408,
        "sha256": "ae1c8d413f7fcca80b37f595fae0dec245d6d84485ae73687c51d93c4daea20f",
        "model": "Qwen3.5-27B-IQ2_M",
        "size_b": 27,
        "min_ram_gib": 15,
    },
    "moe-35b": {
        "repo": "bartowski/Qwen_Qwen3.5-35B-A3B-GGUF",
        "revision": "3d2a22d4eb631b9c8fd2be94599dc2fd84b4a595",
        "filename": "Qwen_Qwen3.5-35B-A3B-IQ2_M.gguf",
        "bytes": 12_964_403_008,
        "sha256": "bce81eb2a0915447f3594a5f37bfd2707ae33e734bc4e1f1bc56c6d1b3325cee",
        "model": "Qwen3.5-35B-A3B-IQ2_M",
        "size_b": 35,
        "min_ram_gib": 15,
    },
}
PORT = 18883
MAX_OUTPUT = 64
CHALLENGES = [
    {"id": "math_product", "prompt": "Compute 19 times 23. Give the number only.", "expected": "437", "grade": "exact"},
    {"id": "math_multiplication", "prompt": "Compute 127 times 38. Give the number only.", "expected": "4826", "grade": "exact"},
    {"id": "math_multistep", "prompt": "Compute ((72 / 8) + 17) times 6. Give the number only.", "expected": "156", "grade": "exact"},
    {"id": "linear_equation", "prompt": "Solve 7x + 5 = 96. Return x as a number only.", "expected": "13", "grade": "exact"},
    {"id": "word_reverse", "prompt": "Reverse the ASCII string abcXYZ exactly, preserving letter case. Only return the reversed string.", "expected": "ZYXcba", "grade": "exact"},
    {"id": "weekday", "prompt": "Today is Thursday. What day of the week will it be in 10 days? Reply with the English weekday only.", "expected": "Sunday", "grade": "casefold"},
    {"id": "logical_rule", "prompt": "All blue keys are metal. No metal keys are plastic. Is it possible for a blue key to be plastic? Answer only YES or NO.", "expected": "NO", "grade": "casefold"},
    {"id": "age_gap", "prompt": "Alice is exactly 3 years older than Ben. After five years, how many years older will Alice be? Number only.", "expected": "3", "grade": "exact"},
    {"id": "sum_squares", "prompt": "Sum the squares of integers 1 through 5 inclusive. Return the integer only.", "expected": "55", "grade": "exact"},
    {"id": "format", "prompt": "Return the exact uppercase word ONLINE. Do not add explanation.", "expected": "ONLINE", "grade": "exact"},
    {"id": "json_schema", "prompt": "Return only this minified JSON object with keys ok and count, where ok is boolean true and count is integer 3.", "expected": '{"ok":true,"count":3}', "grade": "json"},
    {"id": "japanese", "prompt": "Translate Good morning into Japanese. Only give the translation, no punctuation.", "expected": "おはよう", "grade": "casefold"},
]

def weight_url(cfg):
    return f"https://huggingface.co/{cfg['repo']}/resolve/{cfg['revision']}/{urllib.parse.quote(cfg['filename'])}"

def digest(path):
    h=hashlib.sha256()
    size=0
    with path.open("rb") as fp:
        while b:=fp.read(8*1024*1024):
            size+=len(b);h.update(b)
    return size,h.hexdigest()

def verify_weight(path, cfg):
    size,sha=digest(path)
    if size != cfg["bytes"] or sha != cfg["sha256"]:
        raise ValueError("WEIGHTS_SIZE_OR_SHA_MISMATCH")
    return {"bytes":size, "sha256":sha, "source":weight_url(cfg)}

def available(cfg, directory):
    mem=linux_memory_gib()
    disk=shutil.disk_usage(directory)
    disk_needed=cfg["bytes"]/2**30+3.5
    if mem is None or mem < cfg["min_ram_gib"]:
        raise RuntimeError("RUNNER_MEMORY_UNDER_PUBLISHED_THRESHOLD")
    if disk.free/2**30 < disk_needed:
        raise RuntimeError("RUNNER_DISK_UNDER_FILE_PLUS_RESERVE")
    return {"ram_gib":mem, "disk_free_gib":round(disk.free/2**30,2),
            "disk_needed_gib":round(disk_needed,2)}

def download(cfg, out):
    if out.exists():
        return verify_weight(out,cfg)
    tmp=out.with_name(out.name+".part")
    tmp.unlink(missing_ok=True)
    try:
        subprocess.run(["curl","--fail","--location","--silent","--show-error",
            "--retry","2","--retry-delay","4","--max-time","1100",
            "--output",str(tmp),weight_url(cfg)],
            check=True,timeout=1150)
        proof=verify_weight(tmp,cfg)
        os.replace(tmp,out)
        return proof
    finally:
        tmp.unlink(missing_ok=True)

def grade(answer, challenge):
    if not isinstance(answer,str):return False
    text=answer.strip()
    expected=challenge["expected"]
    if challenge["grade"] == "exact":
        return text==expected
    if challenge["grade"] == "casefold":
        return text.casefold() == expected.casefold()
    if challenge["grade"] == "json":
        try:
            val=json.loads(text)
            exp=json.loads(expected)
            return val==exp and type(val.get("ok")) is bool and type(val.get("count")) is int
        except (json.JSONDecodeError,AttributeError):
            return False
    raise ValueError("GRADER_UNKNOWN")

def request(path, key, body=None, timeout=10):
    connection=http.client.HTTPConnection("127.0.0.1",PORT,timeout=timeout)
    headers={"Authorization":"Bearer "+key}
    if body is not None:headers["Content-Type"]="application/json"
    try:
        connection.request("POST" if body is not None else "GET",path,
            body=json.dumps(body,ensure_ascii=False).encode("utf8") if body is not None else None,
            headers=headers)
        r=connection.getresponse()
        raw=r.read(1024*1024)
        return r.status,json.loads(raw)
    finally:
        connection.close()

def run_model(cfg, exe, weights, outfile, limit):
    if not exe.is_file():raise FileNotFoundError("LLAMA_SERVER_BINARY_MISSING")
    key=secrets.token_urlsafe(40)
    env=dict(os.environ)
    env["LLAMA_API_KEY"]=key
    logpath=outfile.parent/"runtime.log"
    model_id=cfg["model"]
    with logpath.open("wb") as log:
        proc=subprocess.Popen([str(exe),"-m",str(weights),"-a",model_id,"--host","127.0.0.1",
            "--port",str(PORT),"--no-webui","--no-mmproj","--cache-ram","0",
            "-c","1536","-t","4","-np","1","-ngl","0"],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            start=time.monotonic()
            while time.monotonic()-start<250:
                if proc.poll() is not None:raise RuntimeError("LLAMA_SERVER_FAILED_TO_START")
                try:
                    status,_=request("/health",key,timeout=2)
                    if status==200:break
                except (OSError,TimeoutError,ValueError):
                    pass
                time.sleep(2)
            else:raise TimeoutError("MODEL_START_TIMEOUT")
            state,models=request("/v1/models",key)
            if state!=200 or model_id not in [x.get("id") for x in models.get("data",[])]:
                raise RuntimeError("LOADED_MODEL_ID_NOT_VERIFIED")
            results=[]
            for case in CHALLENGES[:limit]:
                body={
                    "model":model_id,"max_tokens":MAX_OUTPUT,"temperature":0,"stream":False,
                    "chat_template_kwargs":{"enable_thinking":False},
                    "messages":[{"role":"system","content":"You are a precise assistant. Return only the requested answer, no reasoning."},
                                {"role":"user","content":case["prompt"]}],
                }
                start=time.monotonic()
                status,payload=request("/v1/chat/completions",key,body,timeout=160)
                seconds=round(time.monotonic()-start,2)
                answer=""
                if status==200:
                    answer=payload.get("choices",[{}])[0].get("message",{}).get("content","")
                if not isinstance(answer,str):answer=""
                passed=grade(answer,case)
                results.append({"id":case["id"],"status":status,
                                "passed":passed,"answer":answer[:200],
                                "seconds":seconds,
                                "completion_tokens":payload.get("usage",{}).get("completion_tokens",0)})
                print(json.dumps({"tested":case["id"],"passed":passed,"seconds":seconds},sort_keys=True),flush=True)
            receipt=signed_record({
                "schema":SCHEMA, "profile":next(k for k,v in MODEL_MAP.items() if v==cfg),
                "model_id":model_id, "model_repo":cfg["repo"], "model_revision":cfg["revision"],
                "quantization_sha256":cfg["sha256"],
                "inference_confirmed":any(x["status"]==200 and bool(x["answer"]) for x in results),
                "anthropic_original_model":False,
                "new_spend_usd":0,"external_model_credits":0,"cloud_runner":"github_hosted_public_linux_cpu",
                "score":sum(x["passed"] for x in results),"total":len(results),"results":results,
                "runtime_seconds":round(sum(x["seconds"] for x in results),2)})
            outfile.write_bytes(normalized(receipt)+b"\n")
            print(json.dumps({"result":"BENCHMARK_COMPLETED","profile":receipt["profile"],
                "score":receipt["score"],"total":receipt["total"],
                "sha256":receipt["sha256"]},sort_keys=True),flush=True)
            return 0 if receipt["inference_confirmed"] else 3
        finally:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill()

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--profile",choices=tuple(MODEL_MAP),required=True)
    a.add_argument("--binary",type=Path,required=True)
    a.add_argument("--weights",type=Path,required=True)
    a.add_argument("--receipt",type=Path,required=True)
    a.add_argument("--execute",action="store_true")
    a.add_argument("--tasks",type=int,default=12)
    args=a.parse_args()
    if not 1<=args.tasks<=len(CHALLENGES):
        raise ValueError("TASK_COUNT_INVALID")
    cfg=MODEL_MAP[args.profile]
    if not args.execute:
        print(json.dumps({"mode":"DRY_RUN","profile":args.profile,"model":cfg["model"],
            "expected_bytes":cfg["bytes"],"expected_sha256":cfg["sha256"],"new_spend_usd":0}))
        return 0
    args.weights.mkdir(parents=True,exist_ok=True)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"phase":"resource_gate",**available(cfg,args.weights)}),flush=True)
    proof=download(cfg,args.weights/cfg["filename"])
    print(json.dumps({"phase":"verified_open_weights",**proof}),flush=True)
    return run_model(cfg,args.binary,args.weights/cfg["filename"],args.receipt,args.tasks)

if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,TimeoutError,subprocess.CalledProcessError) as e:
        print(json.dumps({"state":"FAILED","code":type(e).__name__,"reason":str(e)[:180]}),file=sys.stderr)
        sys.exit(2)
