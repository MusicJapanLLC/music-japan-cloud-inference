#!/usr/bin/env python3
"""Actual >120B public open-weight inference on GitHub's free hosted Linux CPU.

Pin *every shard* by publisher revision, size and SHA-256. Use mmap, no fit,
and tiny context. Record real output or reproducible failure evidence.
No OpenAI/Anthropic API, customer data, login theft, paid GPU or paid job.

A model having more parameters, or an IQ1 GGUF loading, DOES NOT prove it is
more accurate than gpt-oss-120b. Evidence and quality are separate gates.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
from urllib.parse import quote

from cloud_probe import linux_memory_gib, normalized, signed_record
from objective_verifier import verify_answer

SCHEMA = "mj.beyond120.real-inference/1.0.0"
PORT = 18957
CONTEXT = 768
PROMPTS = (
    "Compute ((72 / 8) + 17) times 6. Reply only with the integer.",
    "Today is Thursday. What day of the week will it be in 10 days? Reply with only the English weekday.",
)
MODELS = {
    "thinking-235b-iq2m": {
        "repo": "bartowski/Qwen_Qwen3-235B-A22B-Thinking-2507-GGUF",
        "base": "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "revision": "49849ae07d639e1568228f7803f990806d439e2c",
        "params_b": 235,
        "active_b": 22,
        "quant": "IQ2_M",
        "license_source": "Qwen official Apache-2.0 model; downstream quant card independently review",
        "model_alias": "qwen3-235b-a22b-thinking-iq2m",
        "min_free_reserve_gib": 3.5,
        "max_output_tokens": 320,
        "enable_thinking": True,
        "parts": (
            ("Qwen_Qwen3-235B-A22B-Thinking-2507-IQ2_M/Qwen_Qwen3-235B-A22B-Thinking-2507-IQ2_M-00001-of-00002.gguf", 39_896_524_640, "93f6f218a8d85b2caddc593bb490bb09c3fa653c0920018ef6ec0f43a657cfad"),
            ("Qwen_Qwen3-235B-A22B-Thinking-2507-IQ2_M/Qwen_Qwen3-235B-A22B-Thinking-2507-IQ2_M-00002-of-00002.gguf", 38_657_360_256, "9df3196aca9dd43df41ffdce4e296322426109396832fc8afc4fb6683b9e2279"),
        ),
    },
    "frontier-397b-iq1s": {
        "repo": "bartowski/Qwen_Qwen3.5-397B-A17B-GGUF",
        "base": "Qwen/Qwen3.5-397B-A17B",
        "revision": "69ed2f05fdcea0e9a880dd4c518be64c01663157",
        "params_b": 397,
        "active_b": 17,
        "quant": "IQ1_S",
        "license_source": "Apache-2.0 as stated in publisher model metadata",
        "model_alias": "qwen3.5-397b-a17b-iq1s",
        "min_free_reserve_gib": 1.5,
        "max_output_tokens": 128,
        "enable_thinking": False,
        "parts": (
            ("Qwen_Qwen3.5-397B-A17B-IQ1_S/Qwen_Qwen3.5-397B-A17B-IQ1_S-00001-of-00003.gguf", 39_623_329_728, "ce8f640d990bfb89354f60d2bbcd780c8af5b0ba3c25b78028929c63b0d1dfd5"),
            ("Qwen_Qwen3.5-397B-A17B-IQ1_S/Qwen_Qwen3.5-397B-A17B-IQ1_S-00002-of-00003.gguf", 39_830_756_192, "2c1b1fbdfb0c6c02b837baf5975bf9a4c01c221feeef12f3ca84936945bd3d01"),
            ("Qwen_Qwen3.5-397B-A17B-IQ1_S/Qwen_Qwen3.5-397B-A17B-IQ1_S-00003-of-00003.gguf", 9_277_145_344, "a1ef1259f9f446d67df65f25fff8275ecb02f14972ad36baa16508d23813f036"),
        ),
    },
}

def check_manifest(profile: str) -> dict:
    if profile not in MODELS:
        raise ValueError("MODEL_PROFILE_NOT_ALLOWLISTED")
    m=MODELS[profile]
    parts=m["parts"]
    if m["params_b"] <= 120 or len(parts) < 2:
        raise ValueError("HIGH_CAPACITY_PROFILE_OR_SHARDS_MISSING")
    if len({p[0] for p in parts}) != len(parts):
        raise ValueError("DUPLICATE_SHARD")
    for name,size,sha in parts:
        if (name.startswith("/") or ".." in name or "\\" in name or
            not name.endswith(".gguf") or len(sha)!=64 or
            not re.fullmatch(r"[0-9a-f]{64}",sha) or size<=1000):
            raise ValueError("BAD_SHARD_MANIFEST")
    return m

def model_size_gib(profile: str) -> float:
    return sum(x[1] for x in check_manifest(profile)["parts"])/2**30

def resource_gate(profile: str, folder: Path) -> dict:
    cfg=check_manifest(profile)
    disk=shutil.disk_usage(folder)
    ram=linux_memory_gib()
    min_required=model_size_gib(profile)+cfg["min_free_reserve_gib"]
    obs={
        "free_gib":round(disk.free/2**30,2),
        "total_gib":round(disk.total/2**30,2),
        "ram_gib":ram,
        "weight_gib":round(model_size_gib(profile),2),
        "required_free_gib":round(min_required,2),
        "dedicated_gpu_allocated":False,
    }
    if ram is None or ram<12:
        raise RuntimeError("RUNNER_RAM_TOO_LOW")
    if disk.free/2**30<min_required:
        raise RuntimeError("RUNNER_FREE_DISK_TOO_LOW_FOR_ALL_SHARDS")
    return obs

def pinned_url(profile: str, part: tuple) -> str:
    cfg=check_manifest(profile)
    if part not in cfg["parts"]:
        raise ValueError("SHARD_NOT_IN_PINNED_MANIFEST")
    return ("https://huggingface.co/"+cfg["repo"]+"/resolve/"+cfg["revision"]+"/"+
            quote(part[0],safe="/"))

def sha256_file(path:Path) -> tuple[int,str]:
    sha=hashlib.sha256()
    total=0
    with path.open("rb") as f:
        while buf:=f.read(8*1024*1024):
            total+=len(buf)
            sha.update(buf)
    return total,sha.hexdigest()

def verify_part(file:Path, part:tuple) -> dict:
    count,sha=sha256_file(file)
    if count!=part[1] or sha!=part[2]:
        raise ValueError("SHARD_SIZE_OR_SHA256_INVALID")
    return {"shard":part[0],"bytes":count,"sha256":sha}

def download(profile:str,part:tuple,dir:Path)->dict:
    target=dir/part[0]
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file():
        return verify_part(target,part)
    partial=target.with_suffix(".partial")
    partial.unlink(missing_ok=True)
    try:
        subprocess.run([
            "curl","--fail","--location","--silent","--show-error",
            "--retry","2","--retry-delay","4","--max-time","1800",
            "--output",str(partial),pinned_url(profile,part),
        ],check=True,timeout=1820)
        proof=verify_part(partial,part)
        os.replace(partial,target)
        return proof
    finally:
        partial.unlink(missing_ok=True)

def read_cgroup(path:Path=Path("/sys/fs/cgroup/memory.events"))->dict:
    try:
        out={}
        for line in path.read_text("utf8").splitlines():
            key,sep,value=line.partition(" ")
            if sep and key in ("oom","oom_kill","max","high"):
                out[key]=int(value.strip())
        return out
    except (ValueError,OSError):
        return {}

def local_http(key:str,path:str,body=None,timeout:int=15)->tuple[int,dict]:
    if path not in ("/health","/v1/models","/v1/chat/completions"):
        raise ValueError("MODEL_API_ROUTE_NOT_ALLOWLISTED")
    conn=http.client.HTTPConnection("127.0.0.1",PORT,timeout=timeout)
    headers={"Authorization":"Bearer "+key}
    if body is not None:headers["Content-Type"]="application/json"
    try:
        conn.request("POST" if body is not None else "GET",path,
                     body=json.dumps(body,ensure_ascii=False).encode("utf8") if body else None,
                     headers=headers)
        res=conn.getresponse()
        return res.status,json.loads(res.read(800_000))
    finally:
        conn.close()

def validate_public_question(question:str)->str:
    if not isinstance(question,str):
        raise ValueError("PUBLIC_QUESTION_NOT_TEXT")
    prompt=question.strip()
    if not 5<=len(prompt)<=1200 or "\\0" in prompt:
        raise ValueError("PUBLIC_QUESTION_LENGTH_OR_NUL_INVALID")
    return prompt

def actually_infer(profile:str,bin:Path,root:Path,out:Path,tasks:int=1,public_question:str|None=None)->int:
    cfg=check_manifest(profile)
    if not bin.is_file() or not (root/cfg["parts"][0][0]).is_file():
        raise FileNotFoundError("LLAMA_RUNTIME_OR_VERIFIED_WEIGHTS_MISSING")
    if not 1<=tasks<=len(PROMPTS):
        raise ValueError("INVALID_TASKS")
    selected_prompts = (validate_public_question(public_question),) if public_question is not None else PROMPTS[:tasks]
    if public_question is not None and tasks!=1:
        raise ValueError("PUBLIC_CHAT_REQUIRES_SINGLE_REQUEST")
    token=secrets.token_urlsafe(48)
    env=dict(os.environ)
    env["LLAMA_API_KEY"]=token
    started=time.monotonic()
    evidence=signed_record({
        "schema":SCHEMA,"profile":profile,"model_id":cfg["model_alias"],
        "parameters_billions":cfg["params_b"],"active_billions":cfg["active_b"],
        "model_repo":cfg["repo"],"revision":cfg["revision"],"quantization":cfg["quant"],
        "model_shards":[{"path":p[0],"size":p[1],"sha256":p[2]} for p in cfg["parts"]],
        "question_sha256":hashlib.sha256(selected_prompts[0].encode("utf8")).hexdigest(),
        "question_text_in_receipt":False,
        "public_question_override":public_question is not None,
        "model_loaded":False,"real_inference":False,"results":[],
        "model_provider_credits":0,"paid_cloud_allocations":0,
        "public_github_linux_cpu_only":True,"not_a_proprietary_model":True,
        "status":"ATTEMPTED",
    })
    before=read_cgroup()
    out.parent.mkdir(parents=True,exist_ok=True)
    logpath=out.parent/"runtime-diagnostics.txt"
    with logpath.open("wb") as log:
        process=subprocess.Popen([
            str(bin),"-m",str(root/cfg["parts"][0][0]),"-a",cfg["model_alias"],
            "--host","127.0.0.1","--port",str(PORT),
            "--no-webui","--no-mmproj","--load-mode","mmap","--fit","off",
            "--no-repack","--cache-ram","0",
            "-c",str(CONTEXT),"-t","4","-b","32","-ub","16","-np","1","-ngl","0",
            "--jinja",
        ],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:
            limit=900 if profile=="thinking-235b-iq2m" else 780
            healthy=False
            while time.monotonic()-started<limit:
                if process.poll() is not None:
                    evidence["status"]="LOAD_PROCESS_EXITED"
                    break
                try:
                    st,_=local_http(token,"/health",timeout=3)
                    if st==200:
                        healthy=True
                        break
                except (OSError,TimeoutError,ValueError,json.JSONDecodeError):
                    pass
                time.sleep(3)
            evidence["load_seconds"]=round(time.monotonic()-started,2)
            if not healthy:
                evidence["status"]="LOAD_TIMEOUT_OR_EXIT"
                return 3
            st,info=local_http(token,"/v1/models")
            evidence["model_loaded"]=bool(st==200 and cfg["model_alias"] in
                                           [row.get("id") for row in info.get("data",[])])
            if not evidence["model_loaded"]:
                evidence["status"]="MODEL_ID_NOT_VERIFIED"
                return 3
            for i,question in enumerate(selected_prompts):
                payload={
                    "model":cfg["model_alias"],
                    "messages":[{"role":"system","content":"Solve carefully and independently. Finish with only the requested answer."},
                                {"role":"user","content":question}],
                    "temperature":0.6 if cfg["enable_thinking"] else 0,
                    "max_tokens":cfg["max_output_tokens"],
                    "stream":False,
                    "chat_template_kwargs":{"enable_thinking":cfg["enable_thinking"]},
                }
                start=time.monotonic()
                try:
                    code,resp=local_http(token,"/v1/chat/completions",payload,timeout=900)
                    choices=resp.get("choices") or []
                    msg=choices[0].get("message",{}) if choices else {}
                    ans=msg.get("content") if isinstance(msg,dict) else ""
                    if not isinstance(ans,str):ans=""
                    used=(resp.get("usage") or {}).get("completion_tokens",0)
                except (OSError,TimeoutError,ValueError,KeyError,IndexError):
                    code,ans,used=0,"",0
                verify=verify_answer(question,ans)
                evidence["results"].append({
                    "id":i+1,"http_status":code,"visible_answer":ans[:300],
                    "actual_model_generated_text":bool(ans.strip()),
                    "independent_verification":verify.get("status"),
                    "model_output_correct":verify.get("status")=="VERIFIED",
                    "completion_tokens":used,
                    "inference_seconds":round(time.monotonic()-start,2),
                })
                print(json.dumps({"case":i+1,"status":code,
                                  "generated_text":bool(ans.strip()),
                                  "seconds":evidence["results"][-1]["inference_seconds"]}),flush=True)
            evidence["real_inference"]=any(x["http_status"]==200 and x["actual_model_generated_text"]
                                            for x in evidence["results"])
            evidence["status"]="ACTUAL_INFERENCE" if evidence["real_inference"] else "NO_VISIBLE_RESPONSE"
            return 0 if evidence["real_inference"] else 3
        finally:
            evidence["elapsed_seconds"]=round(time.monotonic()-started,2)
            evidence["process_exit_code_before_stop"]=process.poll()
            after=read_cgroup()
            evidence["cgroup_memory_events_delta"]={k:max(0,v-before.get(k,0)) for k,v in after.items()}
            out.write_bytes(normalized(signed_record({k:v for k,v in evidence.items() if k!="sha256"}))+b"\n")
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",required=True,choices=tuple(MODELS))
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--tasks",type=int,default=1)
    parser.add_argument("--public-question",type=str,default=None,
                        help="PUBLIC non-confidential prompt; this becomes the single inference question")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    cfg=check_manifest(args.profile)
    if not args.execute:
        print(json.dumps({"status":"DRY_RUN","profile":args.profile,
                          "total_file_bytes":sum(x[1] for x in cfg["parts"]),
                          "weight_gib":round(model_size_gib(args.profile),2),
                          "external_inference_credits_used":0}))
        return 0
    args.data.mkdir(parents=True,exist_ok=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"phase":"HARDWARE_GATED","profile":args.profile,
                      "real_hardware":resource_gate(args.profile,args.data)}),flush=True)
    for part in cfg["parts"]:
        proof=download(args.profile,part,args.data)
        print(json.dumps({"phase":"PINNED_SHARD_VERIFIED","shard":part[0],
                          "bytes":proof["bytes"],"sha256":proof["sha256"]}),flush=True)
    return actually_infer(args.profile,args.binary,args.data,args.out,args.tasks,args.public_question)

if __name__=="__main__":
    try:sys.exit(main())
    except (RuntimeError,ValueError,OSError,subprocess.CalledProcessError,TimeoutError) as e:
        print(json.dumps({"state":"FAILED","error_code":type(e).__name__,
                          "detail":str(e)[:120]}),file=sys.stderr)
        sys.exit(2)
