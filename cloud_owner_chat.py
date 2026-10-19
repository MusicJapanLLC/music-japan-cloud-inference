#!/usr/bin/env python3
"""Music Japan high-effort public cloud LLM chat — REAL offline-model inference.

The user supplies a PUBLIC, non-confidential prompt. Actual pinned OpenAI
gpt-oss-20b-MXFP4 weights run via llama.cpp on free GitHub-hosted Ubuntu CPUs.
No OpenAI/Anthropic paid inference, no account/API credential collection.

One invocation: actual model response; optional independent challenge and
synthesis in SAME model runtime; limited question-grounded deterministic tool.
No model or quality equivalence with GPT-6/Astra is claimed.
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
from cloud_probe import normalized,signed_record
from cloud_infer_frontier import MODEL_MAP,available,download
from objective_verifier import apply_verifier

SCHEMA="mj.cloud-owner-chat/1.0"
PROFILE="openai-20b"
CFG=MODEL_MAP[PROFILE]
PORT=18931
MAX_QUESTION_CHARS=3200
MAX_TOTAL_TOKENS=9600
MAX_WALL_SECONDS=1500
MAX_ROUNDS=3

def validate_prompt(prompt:object) -> str:
    if not isinstance(prompt,str):
        raise ValueError("QUESTION_NOT_TEXT")
    prompt=prompt.strip()
    if not 4<=len(prompt)<=MAX_QUESTION_CHARS:
        raise ValueError("QUESTION_LENGTH_OUT_OF_RANGE")
    if "\0" in prompt:
        raise ValueError("NUL_NOT_ACCEPTED")
    return prompt

def safe_budgets(rounds:int, tokens:int, max_seconds:int)->None:
    if isinstance(rounds,bool) or not isinstance(rounds,int) or not 1<=rounds<=MAX_ROUNDS:
        raise ValueError("ROUNDS_INVALID")
    if isinstance(tokens,bool) or not isinstance(tokens,int) or not 256<=tokens<=3200:
        raise ValueError("TOKENS_PER_ROUND_INVALID")
    if rounds*tokens>MAX_TOTAL_TOKENS:
        raise ValueError("TOTAL_OUTPUT_BUDGET_INVALID")
    if not 120<=max_seconds<=MAX_WALL_SECONDS:
        raise ValueError("TIME_BUDGET_INVALID")

def stage_prompt(question:str, answers:list[str])->str:
    if not answers:
        return ("Solve the user's question as carefully as possible. "
                "Use deep reasoning for ambiguous or difficult tasks. "
                "Check calculations and contradicting assumptions. "
                "Respond with the final useful answer, not private reasoning notes.\n\n"
                "USER QUESTION:\n"+question)
    if len(answers)==1:
        return ("Independently re-solve this question. The earlier answer may be wrong: "
                "look for alternative interpretations, calculation errors and weak evidence. "
                "Give a corrected, complete final answer.\n\n"
                "QUESTION:\n"+question+"\n\nEARLIER ANSWER:\n"+answers[0][:4500])
    return ("Choose the strongest supported answer to the question. "
            "Two attempts may disagree. Recompute if possible and correct mistakes. "
            "Give the best final answer, distinguishing unverified assertions.\n\n"
            "QUESTION:\n"+question+"\n\nFIRST ATTEMPT:\n"+answers[0][:4000]+
            "\n\nSECOND ATTEMPT:\n"+answers[1][:4000])

def request_local(key:str, path:str, body:dict|None=None,timeout:int=15):
    if path not in ("/health","/v1/models","/v1/chat/completions"):
        raise ValueError("NOT_ALLOWED_LOCAL_ROUTE")
    conn=http.client.HTTPConnection("127.0.0.1",PORT,timeout=timeout)
    headers={"Authorization":"Bearer "+key}
    if body is not None:
        headers["Content-Type"]="application/json"
    try:
        conn.request("POST" if body is not None else "GET",path,
              body=json.dumps(body,ensure_ascii=False).encode("utf8") if body is not None else None,
              headers=headers)
        resp=conn.getresponse()
        content=resp.read(2_000_000)
        if len(content)>=2_000_000:raise ValueError("MODEL_RESPONSE_TOO_LARGE")
        return resp.status,json.loads(content)
    finally:conn.close()

def open_model(binary:Path,weights:Path,key:str,log):
    if not binary.is_file() or not weights.is_file():
        raise FileNotFoundError("PINNED_WEIGHTS_OR_LLAMA_CPP_NOT_FOUND")
    env=dict(os.environ)
    env["LLAMA_API_KEY"]=key
    args=[
        str(binary),"-m",str(weights),"-a",CFG["model"],
        "--host","127.0.0.1","--port",str(PORT),
        "--no-webui","--no-mmproj","--cache-ram","0",
        "-c","4096","-t","4","-ngl","0","-np","1",
    ]
    return subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)

def model_message(prompt:str,max_tokens:int,effort:str)->dict:
    if effort not in ("low","medium","high"):
        raise ValueError("INVALID_REASONING_EFFORT")
    return {
        "model":CFG["model"],
        "messages":[
            {"role":"system","content":"You are Music Japan's research LLM. Provide a high-quality final answer, avoid unsupported certainty."},
            {"role":"user","content":prompt},
        ],
        "reasoning_effort":effort,
        "chat_template_kwargs":{"reasoning_effort":effort},
        "max_tokens":max_tokens,
        "temperature":1.0,
        "stream":False,
    }

def extract_reply(status:int,payload:dict)->tuple[str,int,int]:
    if status!=200 or not isinstance(payload,dict):
        raise RuntimeError("MODEL_HTTP_NOT_OK")
    rows=payload.get("choices")
    if not isinstance(rows,list) or not rows or not isinstance(rows[0],dict):
        raise RuntimeError("MODEL_RESPONSE_INVALID")
    msg=rows[0].get("message",{})
    text=msg.get("content") if isinstance(msg,dict) else None
    if not isinstance(text,str) or not text.strip():
        raise RuntimeError("MODEL_FINAL_TEXT_EMPTY")
    usage=payload.get("usage") or {}
    tokens=int(usage.get("completion_tokens") or 0)
    details=usage.get("completion_tokens_details") or {}
    reasoning=int(details.get("reasoning_tokens") or 0)
    return text.strip(),tokens,reasoning

def select_final(question:str,candidates:list[str])->dict:
    if not candidates:raise ValueError("NO_CANDIDATES")
    # Question-derived math/date checking independently corrects checkable
    # tasks. Model majority or thinking duration alone is NOT factual proof.
    result=apply_verifier(question,candidates[-1])
    return {
        "final_answer":result["final_answer"],
        "model_answer":candidates[-1],
        "answer_source":result["source"],
        "tool_corrected":result["source"]=="deterministic_tool_correction",
        "verification":result["verification"],
        "rounds_agreed":len(set(candidates))==1,
    }

def run_chat(question:str,binary:Path,weights:Path,output:Path,
             rounds:int=3,tokens:int=1280,effort:str="high",max_seconds:int=MAX_WALL_SECONDS)->dict:
    validate_prompt(question)
    safe_budgets(rounds,tokens,max_seconds)
    output.parent.mkdir(parents=True,exist_ok=True)
    token=secrets.token_urlsafe(48)
    begun=time.monotonic()
    with (output.parent/"llama-cloud-chat-runtime.log").open("wb") as fp:
        process=open_model(binary,weights,token,fp)
        try:
            for _ in range(135):
                if process.poll() is not None:
                    raise RuntimeError("MODEL_RUNTIME_FAILED")
                try:
                    status,_=request_local(token,"/health",timeout=3)
                    if status==200:break
                except (OSError,ValueError,TimeoutError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("MODEL_STARTUP_TIMEOUT")
            st, models=request_local(token,"/v1/models")
            if st!=200 or CFG["model"] not in [m.get("id") for m in models.get("data",[]) if isinstance(m,dict)]:
                raise RuntimeError("PINNED_MODEL_NOT_LOADED")
            candidates=[]
            history=[]
            for i in range(rounds):
                elapsed=time.monotonic()-begun
                if elapsed >= max_seconds-30:
                    raise TimeoutError("INFERENCE_COMPUTE_TIME_LIMIT")
                stage=stage_prompt(question,candidates)
                body=model_message(stage,tokens,effort)
                step_start=time.monotonic()
                st, response=request_local(token,"/v1/chat/completions",body,
                             timeout=min(500,max(30,int(max_seconds-elapsed))))
                answer,out_tokens,reasoning_tokens=extract_reply(st,response)
                candidates.append(answer)
                history.append({
                    "round":i+1,"answer":answer[:8000],
                    "elapsed_seconds":round(time.monotonic()-step_start,2),
                    "output_tokens":out_tokens,
                    "reported_reasoning_tokens":reasoning_tokens,
                    "official_model":CFG["model"],
                })
                # Only metadata in stdout, no public prompts or private thoughts.
                print(json.dumps({"round":i+1,"http":st,"visible_answer":True,
                    "seconds":history[-1]["elapsed_seconds"],
                    "tokens":out_tokens},sort_keys=True),flush=True)
            selected=select_final(question,candidates)
            result=signed_record({
                "schema":SCHEMA,
                "observed_utc":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "prompt_sha256":hashlib.sha256(question.encode("utf8")).hexdigest(),
                "prompt_in_output":False,
                "model_profile":PROFILE,"model_id":CFG["model"],
                "model_repo":CFG["repo"],"model_revision":CFG["revision"],
                "model_weight_sha256":CFG["sha256"],
                "real_local_linux_model_inference":True,
                "external_model_api_calls":0,"paid_compute_created":False,
                "quality_equivalent_to_proprietary_gpt6":False,
                "reasoning_effort":effort,"rounds":rounds,
                "token_budget_per_round":tokens,
                "elapsed_total_seconds":round(time.monotonic()-begun,2),
                "rounds_evidence":history,
                "selection":selected,
            })
            output.write_bytes(normalized(result)+b"\n")
            print(json.dumps({"status":"REAL_MODEL_CHAT_COMPLETED",
                "result_sha256":result["sha256"],
                "rounds":rounds,"model_id":CFG["model"],
                "output_present":bool(selected["final_answer"])},sort_keys=True),flush=True)
            return result
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--binary",type=Path,required=True)
    parser.add_argument("--weights-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--rounds",type=int,choices=(1,2,3),default=2)
    parser.add_argument("--tokens",type=int,default=1280)
    parser.add_argument("--effort",choices=("low","medium","high"),default="high")
    parser.add_argument("--max-seconds",type=int,default=1200)
    args=parser.parse_args(argv)
    safe_budgets(args.rounds,args.tokens,args.max_seconds)
    question=os.environ.get("MJ_PUBLIC_RESEARCH_QUESTION","")
    if not args.execute:
        print(json.dumps({"status":"DRY_RUN","model":CFG["model"],
            "reasoning_effort":args.effort,"rounds":args.rounds,
            "model_provider_credits":0,"runtime_started":False}))
        return 0
    question=validate_prompt(question)
    args.weights_dir.mkdir(parents=True,exist_ok=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    hardware=available(CFG,args.weights_dir)
    print(json.dumps({"phase":"runner_verified","ram_gib":hardware["ram_gib"],
         "free_disk_gib":hardware["disk_free_gib"]}),flush=True)
    weights=args.weights_dir/CFG["filename"]
    proof=download(CFG,weights)
    print(json.dumps({"phase":"weights_verified","size":proof["bytes"],
                      "sha256":proof["sha256"]}),flush=True)
    run_chat(question,args.binary,weights,args.out,args.rounds,
             args.tokens,args.effort,args.max_seconds)
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,TimeoutError,subprocess.CalledProcessError) as exc:
        print(json.dumps({"status":"FAILED","reason_code":str(exc)[:100],
                          "error_type":type(exc).__name__}),file=sys.stderr)
        sys.exit(2)
