#!/usr/bin/env python3
"""Music Japan Reasoning Mesh — real, bounded test-time compute across Linux VMs.

Three-stage research pipeline:
  PROPOSE:  OpenAI released gpt-oss-20b (effort=high, public MXFP4)
  CHALLENGE: Community public Fable-inspired Qwable 9B
  SYNTHESIZE: gpt-oss-20b again, high effort, with two independent answers
  EVALUATE: Offline independent grading. Never expose benchmark answers to models.

All model computation is local to an ephemeral owner-authorized public Github
Actions Linux CPU runner. No paid API, proprietary-model credits, private
prompts, customer content or third-party secrets. Not a hosted 24/7 service.
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
from cloud_probe import normalized, signed_record, verify
from cloud_infer_frontier import CHALLENGES, MODEL_MAP, available, download, grade

SCHEMA = "mj.reasoning-mesh-stage/1.0"
REPORT_SCHEMA = "mj.reasoning-mesh-comparison/1.0"
PORT = 18894
STAGES = {"propose": "openai-20b", "challenge": "fable-reconstruction-9b",
          "synthesize": "openai-20b"}
CASE_IDS = [c["id"] for c in CHALLENGES]
DEFAULT_LIMIT = len(CASE_IDS)
MAX_INPUT_CHARS = 10000

def receipt_ok(record: dict, expected_stage: str | None = None) -> bool:
    if not verify(record) or record.get("schema") != SCHEMA:
        return False
    stage = record.get("stage")
    if stage not in STAGES or (expected_stage is not None and stage != expected_stage):
        return False
    if record.get("model_profile") != STAGES[stage]:
        return False
    if record.get("new_spend_usd") != 0 or record.get("proprietary_model_credits") != 0:
        return False
    if record.get("original_claude_or_gpt6") is not False:
        return False
    rows = record.get("items")
    if not isinstance(rows,list) or not 1 <= len(rows) <= DEFAULT_LIMIT:
        return False
    if [x.get("id") for x in rows] != CASE_IDS[:len(rows)]:
        return False
    return all(isinstance(x.get("answer"),str) and
               isinstance(x.get("latency_seconds"),(float,int)) and
               x.get("http_status") == 200
               for x in rows)

def read_verified(path: Path, expected_stage: str) -> dict:
    record = json.loads(path.read_text(encoding="utf8"))
    if not receipt_ok(record, expected_stage):
        raise ValueError("UPSTREAM_RECEIPT_NOT_VERIFIED")
    return record

def stage_instruction(stage: str, case: dict, proposal: str = "",
                      challenge: str = "") -> str:
    question=case["prompt"]
    if stage == "propose":
        return (
          "Solve this carefully. Work through different possibilities internally, "
          "double-check arithmetic and formatting, then output ONLY the final requested answer. "
          "Spend more reasoning on ambiguous or multi-step questions.\n\n"
          "QUESTION:\n"+question)
    if stage == "challenge":
        return (
          "You are an INDEPENDENT reviewer. The proposed answer below might be wrong. "
          "Solve the question again from first principles, look for mistakes, and output "
          "ONLY your own final answer. Do not automatically agree. "
          "If the proposal is correct, repeating it is fine.\n\n"
          "QUESTION:\n"+question+"\n\nPROPOSED ANSWER:\n"+proposal[:1200])
    if stage == "synthesize":
        return (
          "You have two independent proposals for the same question. "
          "Evaluate their correctness rather than voting by popularity. "
          "Take as much reasoning as necessary for accuracy, test the arithmetic, "
          "notice discrepancies, and RETURN ONLY the correct final answer in the requested format. "
          "Either previous answer may be wrong.\n\nQUESTION:\n"+question+
          "\n\nFIRST ANSWER:\n"+proposal[:1200]+
          "\n\nINDEPENDENT CHALLENGER:\n"+challenge[:1200])
    raise ValueError("UNKNOWN_STAGE")

def infer_http(key: str, path: str, body: dict | None=None, timeout=220) -> tuple[int,dict]:
    if path not in ("/v1/models", "/health", "/v1/chat/completions"):
        raise ValueError("UNKNOWN_LOCAL_API_PATH")
    connection = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    headers = {"Authorization":"Bearer "+key}
    if body is not None:
        headers["Content-Type"]="application/json"
    try:
        connection.request("POST" if body is not None else "GET",path,
             body=json.dumps(body,ensure_ascii=False).encode("utf8") if body is not None else None,
             headers=headers)
        response=connection.getresponse()
        data=response.read(2*1024*1024)
        if len(data)>=2*1024*1024:
            raise ValueError("LOCAL_RESPONSE_TOO_LARGE")
        return response.status,json.loads(data)
    finally:
        connection.close()

def parse_answer(response:dict) -> str:
    try:
        message=response["choices"][0]["message"]
        output=message.get("content") or ""
        return output if isinstance(output,str) else ""
    except (IndexError,KeyError,TypeError):
        return ""

def build_request(profile:str, stage:str, prompt:str,
                  effort:str="high", token_budget:int=1024) -> dict:
    if stage not in STAGES or profile!=STAGES[stage]:
        raise ValueError("MODEL_OR_STAGE_MISMATCH")
    if effort not in ("low","medium","high") or not 128<=token_budget<=2048:
        raise ValueError("INVALID_REASONING_BUDGET")
    if not isinstance(prompt,str) or len(prompt)>MAX_INPUT_CHARS:
        raise ValueError("PROMPT_TOO_LARGE")
    model=MODEL_MAP[profile]["model"]
    gptoss=profile=="openai-20b"
    user=[
        {"role":"system","content":"You are a careful text-only reasoning model. The user wants only the final answer, without extra words."},
        {"role":"user","content":prompt}
    ]
    payload={
        "model":model, "messages":user, "stream":False,
        "max_tokens":token_budget if gptoss else min(token_budget,384),
        "temperature":1.0 if gptoss else 0.2,
        "chat_template_kwargs":({"reasoning_effort":effort} if gptoss else {"enable_thinking":False}),
    }
    if gptoss:
        payload["reasoning_effort"]=effort
    return payload

def run_stage(stage: str, model_file:Path, binary:Path,
              output:Path, upstream:Path|None, challenger:Path|None,
              limit:int, effort:str, token_budget:int, max_seconds:int) -> dict:
    if stage not in STAGES:
        raise ValueError("INVALID_STAGE")
    if not 1<=limit<=DEFAULT_LIMIT:
        raise ValueError("TASK_LIMIT_OUT_OF_RANGE")
    if not binary.is_file() or not model_file.is_file():
        raise FileNotFoundError("MODEL_OR_INFERENCE_RUNTIME_NOT_FOUND")
    previous=read_verified(upstream, "propose") if stage in ("challenge","synthesize") and upstream else None
    other=read_verified(challenger,"challenge") if stage=="synthesize" and challenger else None
    if stage in ("challenge","synthesize") and previous is None:
        raise ValueError("PREVIOUS_PROPOSAL_REQUIRED")
    if stage=="synthesize" and other is None:
        raise ValueError("CHALLENGE_PROOF_REQUIRED")
    if previous is not None and len(previous["items"])!=limit:
        raise ValueError("PROPOSAL_TASK_MISMATCH")
    if other is not None and len(other["items"])!=limit:
        raise ValueError("CHALLENGER_TASK_MISMATCH")
    key=secrets.token_urlsafe(48)
    cfg=MODEL_MAP[STAGES[stage]]
    output.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    env["LLAMA_API_KEY"]=key
    start=time.monotonic()
    with (output.parent / ("runtime-"+stage+".log")).open("wb") as logs:
        process=subprocess.Popen([
            str(binary), "-m",str(model_file),"-a",cfg["model"],
            "--host","127.0.0.1","--port",str(PORT),
            "--no-webui","--no-mmproj","--cache-ram","0",
            "-c","4096","-t","4","-np","1","-ngl","0"
        ],env=env,stdin=subprocess.DEVNULL,stdout=logs,stderr=subprocess.STDOUT)
        try:
            for _ in range(150):
                if process.poll() is not None:
                    raise RuntimeError("MODEL_RUNTIME_EXITED")
                try:
                    code,_=infer_http(key,"/health",timeout=2)
                    if code==200:
                        break
                except (OSError,TimeoutError,ValueError):
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("MODEL_START_TIMEOUT")
            code,models=infer_http(key,"/v1/models")
            if code!=200 or cfg["model"] not in [x.get("id") for x in models.get("data",[])]:
                raise RuntimeError("REAL_MODEL_ID_NOT_VERIFIED")
            rows=[]
            for i,case in enumerate(CHALLENGES[:limit]):
                if time.monotonic()-start>max_seconds:
                    raise TimeoutError("STAGE_CPU_TIME_BUDGET_EXCEEDED")
                prompt=stage_instruction(stage,case,
                    previous["items"][i]["answer"] if previous else "",
                    other["items"][i]["answer"] if other else "")
                body=build_request(STAGES[stage],stage,prompt,effort,token_budget)
                sent=time.monotonic()
                code,result=infer_http(key,"/v1/chat/completions",body,timeout=220)
                duration=round(time.monotonic()-sent,2)
                answer=parse_answer(result).strip()
                usage=result.get("usage",{}) if isinstance(result,dict) else {}
                item={
                    "id":case["id"],
                    "answer":answer[:1000],
                    "http_status":code,
                    "latency_seconds":duration,
                    "output_tokens":int(usage.get("completion_tokens") or 0),
                    "request_budget_tokens":body["max_tokens"],
                    "observed_reasoning_tokens":int(
                        (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
                    "generated_visible_text":bool(answer),
                }
                rows.append(item)
                print(json.dumps({"stage":stage,"case":case["id"],"status":code,
                    "visible":bool(answer),"output_tokens":item["output_tokens"],
                    "seconds":duration}),flush=True)
                # Errors are NOT silently passed as successful model conclusions.
                if code!=200 or not answer:
                    raise RuntimeError("MODEL_NO_VALID_FINAL_TEXT_"+case["id"])
            result=signed_record({
                "schema":SCHEMA,"stage":stage,"model_profile":STAGES[stage],
                "model_id":cfg["model"],"model_revision":cfg["revision"],
                "weights_sha256":cfg["sha256"],
                "reasoning_effort":effort if STAGES[stage]=="openai-20b" else "not_supported_by_this_checkpoint",
                "requested_output_tokens_per_task":token_budget,
                "upstream_sha256":previous["sha256"] if previous else None,
                "challenge_sha256":other["sha256"] if other else None,
                "new_spend_usd":0,"proprietary_model_credits":0,
                "original_claude_or_gpt6":False,
                "public_runner_only":True,
                "items":rows,
                "total_elapsed_seconds":round(time.monotonic()-start,2),
                "observed_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            })
            output.write_bytes(normalized(result)+b"\n")
            print(json.dumps({"stage":stage,"complete":True,
                "evidence_sha256":result["sha256"],"cases":len(rows)}),flush=True)
            return result
        finally:
            process.terminate()
            try:process.wait(timeout=12)
            except subprocess.TimeoutExpired:process.kill()

def evaluate(proposal:dict,challenge:dict,synthesis:dict)->dict:
    for stage,rec in (("propose",proposal),("challenge",challenge),("synthesize",synthesis)):
        if not receipt_ok(rec,stage):
            raise ValueError("MISSING_OR_TAMPERED_STAGE_"+stage)
    ids=[row["id"] for row in proposal["items"]]
    if ids != [row["id"] for row in challenge["items"]] or ids != [row["id"] for row in synthesis["items"]]:
        raise ValueError("CROSS_STAGE_TASK_ID_MISMATCH")
    if challenge["upstream_sha256"]!=proposal["sha256"]:
        raise ValueError("CHALLENGER_PARENT_MISMATCH")
    if synthesis["upstream_sha256"]!=proposal["sha256"] or synthesis["challenge_sha256"]!=challenge["sha256"]:
        raise ValueError("SYNTHESIS_PARENT_MISMATCH")
    rows=[]
    for i,case in enumerate(CHALLENGES[:len(ids)]):
        grades={}
        for stage,rec in (("propose",proposal),("challenge",challenge),("synthesize",synthesis)):
            text=rec["items"][i]["answer"]
            grades[stage]={"passed":grade(text,case),"answer":text[:200],
                           "seconds":rec["items"][i]["latency_seconds"]}
        rows.append({"id":case["id"],"scores":grades,
                     "synthesis_changed_proposal":(
                       proposal["items"][i]["answer"]!=synthesis["items"][i]["answer"])})
    counts={stage:sum(row["scores"][stage]["passed"] for row in rows)
            for stage in STAGES}
    report=signed_record({
        "schema":REPORT_SCHEMA,
        "evaluation_only_not_an_oracle_router":True,
        "profiles":STAGES,
        "real_model_inference_confirmed":True,
        "proprietary_model_credits_used":0,
        "paid_cloud_created":False,
        "tasks":len(rows),
        "scores":counts,
        "change_from_proposal":counts["synthesize"]-counts["propose"],
        "total_inference_latency_seconds":round(sum(
            item["latency_seconds"]
            for record in (proposal,challenge,synthesis) for item in record["items"]),2),
        "source_receipts":{n:r["sha256"] for n,r in [
            ("propose",proposal),("challenge",challenge),("synthesize",synthesis)]},
        "results":rows,
        "recommendation":"MORE_REASONING_HELPED_THIS_EVAL" if counts["synthesize"]>counts["propose"]
            else "MORE_REASONING_DID_NOT_IMPROVE_THIS_EVAL",
    })
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage",choices=[*STAGES,"evaluate"],required=True)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--binary",type=Path)
    ap.add_argument("--weights-dir",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--proposal",type=Path)
    ap.add_argument("--challenge",type=Path)
    ap.add_argument("--synthesis",type=Path)
    ap.add_argument("--limit",type=int,default=DEFAULT_LIMIT)
    ap.add_argument("--effort",choices=("low","medium","high"),default="high")
    ap.add_argument("--tokens",type=int,default=1152)
    ap.add_argument("--max-seconds",type=int,default=2100)
    args=ap.parse_args()
    if args.stage=="evaluate":
        if not all((args.proposal,args.challenge,args.synthesis,args.output)):
            raise ValueError("EVALUATION_REQUIRES_ALL_RECEIPTS")
        result=evaluate(read_verified(args.proposal,"propose"),
                        read_verified(args.challenge,"challenge"),
                        read_verified(args.synthesis,"synthesize"))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_bytes(normalized(result)+b"\n")
        print(json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
        return 0
    if not args.execute:
        print(json.dumps({"status":"DRY_RUN", "stage":args.stage,
            "model_profile":STAGES[args.stage],
            "model_credit_use":0,"paid_jobs":0,"inference":False}),flush=True)
        return 0
    if not args.binary or not args.weights_dir or not args.output:
        raise ValueError("MODEL_AND_OUTPUT_REQUIRED")
    if not 600 <= args.max_seconds <= 2700:
        raise ValueError("TIME_BUDGET_OUT_OF_RANGE")
    cfg=MODEL_MAP[STAGES[args.stage]]
    args.weights_dir.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"stage":args.stage,"hardware":available(cfg,args.weights_dir)}),flush=True)
    model_path=args.weights_dir/cfg["filename"]
    verified=download(cfg,model_path)
    print(json.dumps({"stage":args.stage,"verified_model_sha256":verified["sha256"],
                      "model_revision":cfg["revision"]}),flush=True)
    run_stage(args.stage,model_path,args.binary,args.output,args.proposal,args.challenge,
              args.limit,args.effort,args.tokens,args.max_seconds)
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,TimeoutError,subprocess.CalledProcessError) as err:
        print(json.dumps({"state":"FAILED","reason":str(err)[:200],
                          "error_type":type(err).__name__}),file=sys.stderr)
        sys.exit(2)
