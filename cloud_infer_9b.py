#!/usr/bin/env python3
"""One-shot genuine 9B CPU inference on a public GitHub-hosted runner, no model API credits."""
from __future__ import annotations
import argparse, hashlib, http.client, json, os, secrets, shutil, subprocess, sys, time
from pathlib import Path
from cloud_probe import linux_memory_gib, normalized, signed_record

HF_REPO="gguf/Qwen3.5-9B-GGUF"
HF_COMMIT="dc40af5d4666acd6ce1c1dd71ec61cee2f1c966e"
FILE="qwen3.5-9b-Q3_K_M.gguf"
SIZE=5_178_748_960
SHA256="24fccd8416fcdca792f939de4a2240eb57795b0f7a9ba92e62398eab3ec5c11e"
MODEL="Qwen3.5-9B-Q3_K_M"
URL=f"https://huggingface.co/{HF_REPO}/resolve/{HF_COMMIT}/{FILE}"
PORT=18881
PROMPTS=[
 {"id":"math","ask":"Calculate 19 times 23. Answer only the integer.","needle":"437"},
 {"id":"format","ask":"Answer with the exact word ONLINE and nothing else.","needle":"ONLINE"},
 {"id":"japanese","ask":"Translate Good morning into Japanese. Only the translation.","needle":"おはよう"}]
def sha_file(path):
    d=hashlib.sha256();n=0
    with open(path,"rb") as inp:
        while x:=inp.read(8*1024*1024):
            n+=len(x);d.update(x)
    return n,d.hexdigest()
def verify(path):
    size,digest=sha_file(path)
    if size!=SIZE or digest!=SHA256:
        raise ValueError("UNTRUSTED_MODEL_BYTES")
    return {"size":size,"sha256":digest}
def resources(folder):
    ram=linux_memory_gib()
    free=shutil.disk_usage(folder).free/2**30
    if ram is None or ram<12:
        raise RuntimeError("CLOUD_RUNNER_RAM_UNDER_12_GIB")
    if free<6.8:
        raise RuntimeError("CLOUD_RUNNER_DISK_UNDER_6_8_GIB")
    return {"ram_gib":ram,"free_disk_gib":round(free,2)}
def download_model(dest):
    if dest.exists():
        return verify(dest)
    part=dest.with_name(dest.name+".part")
    part.unlink(missing_ok=True)
    try:
        subprocess.run(["curl","--fail","--location","--silent","--show-error",
            "--retry","2","--retry-delay","4","--max-time","780",
            "--output",str(part),URL],timeout=810,check=True)
        proof=verify(part)
        os.replace(part,dest)
        return proof
    finally:
        part.unlink(missing_ok=True)
def http_request(method,path,key,body=None,timeout=8):
    c=http.client.HTTPConnection("127.0.0.1",PORT,timeout=timeout)
    headers={"Authorization":"Bearer "+key}
    if body is not None: headers["Content-Type"]="application/json"
    try:
        c.request(method,path,body=json.dumps(body).encode("utf8") if body is not None else None,headers=headers)
        response=c.getresponse()
        return response.status,json.loads(response.read(1024*1024))
    finally:c.close()
def infer(binary,weights,outfile):
    if not binary.is_file(): raise FileNotFoundError("LLAMA_SERVER_MISSING")
    token=secrets.token_urlsafe(32)
    env=dict(os.environ)
    env["LLAMA_API_KEY"]=token
    logfile=outfile.parent/"llama-diagnostics.log"
    with logfile.open("wb") as log:
        proc=subprocess.Popen([str(binary),"-m",str(weights),"-a",MODEL,
            "--host","127.0.0.1","--port",str(PORT),"--no-webui",
            "--no-mmproj","-c","2048","-t","4","-ngl","0",
            "-np","1","--cache-ram","0"],
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        try:
            start=time.monotonic()
            while time.monotonic()-start<210:
                if proc.poll() is not None:raise RuntimeError("MODEL_START_FAILED")
                try:
                    status,_=http_request("GET","/health",token,timeout=3)
                    if status==200:break
                except (OSError,ValueError,TimeoutError):
                    pass
                time.sleep(2)
            else:raise TimeoutError("MODEL_LOADING_TIMEOUT")
            status,models=http_request("GET","/v1/models",token)
            if status!=200 or MODEL not in [x.get("id") for x in models.get("data",[])]:
                raise RuntimeError("MODEL_ID_MISMATCH")
            results=[]
            for t in PROMPTS:
                payload={"model":MODEL,"max_tokens":40,"temperature":0,
                    "messages":[{"role":"system","content":"Reply concisely. No reasoning or explanations."},
                        {"role":"user","content":t["ask"]}],
                    "stream":False,"chat_template_kwargs":{"enable_thinking":False}}
                started=time.monotonic()
                code,ans=http_request("POST","/v1/chat/completions",token,payload,timeout=150)
                output=ans.get("choices",[{}])[0].get("message",{}).get("content","") if code==200 else ""
                if not isinstance(output,str):output=""
                results.append({"id":t["id"],"http_status":code,"output":output[:400],
                    "correct":t["needle"] in output,"has_text":bool(output.strip()),
                    "seconds":round(time.monotonic()-started,2),
                    "output_tokens":ans.get("usage",{}).get("completion_tokens",0)})
            row=signed_record({"schema":"mj.cloud-9b-inference/1.0.0",
                "model":MODEL,"model_repo":HF_REPO,"model_revision":HF_COMMIT,
                "model_sha256":SHA256,"genuine_open_weight_inference":True,
                "fable_or_opus_model":False,"credits_used":0,"paid_cloud_used":False,
                "results":results,"quality_passes":sum(x["correct"] for x in results),
                "quality_total":len(results)})
            outfile.write_bytes(normalized(row)+b"\n")
            print(json.dumps(row,sort_keys=True,ensure_ascii=False),flush=True)
            return 0 if sum(x["has_text"] for x in results)>=2 else 3
        finally:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if not a.execute:
        print(json.dumps({"mode":"DRY_RUN","model":MODEL,"source":URL,"credits":0}))
        return 0
    a.data.mkdir(parents=True,exist_ok=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"phase":"resource_check",**resources(a.data)}),flush=True)
    proof=download_model(a.data/FILE)
    print(json.dumps({"phase":"model_verified",**proof}),flush=True)
    return infer(a.binary,a.data/FILE,a.out)
if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError,TimeoutError,subprocess.CalledProcessError) as e:
        print(json.dumps({"status":"FAILED","type":type(e).__name__,"reason":str(e)[:120]}),file=sys.stderr)
        sys.exit(2)
