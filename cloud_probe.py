#!/usr/bin/env python3
"""Bounded, read-only inventory for GitHub-hosted Ubuntu compute.

Never reads credentials or account secrets. Only measures actual worker hardware,
not vendor claims; produces a SHA256 integrity receipt. Stdlib-only.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

SCHEMA="mj.cloud-runner-proof/1.0.0"
MEM_LIMIT_GIB=64
DISK_LIMIT_GIB=2048

def normalized(x):
    return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf8")

def signed_record(row):
    content={k:v for k,v in row.items() if k!="sha256"}
    return {**content,"sha256":hashlib.sha256(normalized(content)).hexdigest()}

def verify(row):
    return isinstance(row,dict) and isinstance(row.get("sha256"),str) and signed_record(row)["sha256"]==row["sha256"]

def linux_memory_gib(path:Path=Path("/proc/meminfo")):
    try:
        match=re.search(r"^MemTotal:\s+(\d+)\s+kB$",path.read_text(),re.MULTILINE)
        return round(int(match.group(1))*1024/2**30,2) if match else None
    except (OSError,ValueError):
        return None

def nvidia_names(run=subprocess.run):
    try:
        r=run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
              capture_output=True,text=True,timeout=5,check=False)
        return [x.strip()[:96] for x in r.stdout.splitlines() if x.strip()][:8] if r.returncode==0 else []
    except (OSError,subprocess.TimeoutExpired):
        return []

def collect(disk_root=None):
    root=Path(disk_root) if disk_root else Path.cwd()
    disk=shutil.disk_usage(root)
    mem=linux_memory_gib() if platform.system()=="Linux" else None
    gpus=nvidia_names()
    data={
        "schema":SCHEMA,
        "observed_utc":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "platform":platform.system(),
        "runner_environment":"GitHub Actions" if os.environ.get("GITHUB_ACTIONS")=="true" else "other",
        "cpu_logical":os.cpu_count(),
        "ram_gib":mem,
        "disk_total_gib":round(disk.total/2**30,2),
        "disk_free_gib":round(disk.free/2**30,2),
        "nvidia_gpu_names":gpus,
        "minimum_requirements":{
            "ram_64gib":mem is not None and mem>=MEM_LIMIT_GIB,
            "disk_2tib":disk.total/2**30>=DISK_LIMIT_GIB,
            "nvidia_gpu_observed":bool(gpus)},
        "model":None,
        "model_inference_executed":False,
        "paid_jobs_or_credits_used_by_script":0,
        "secrets_or_messages_collected":False,
    }
    return signed_record(data)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path)
    p.add_argument("--verify",type=Path)
    args=p.parse_args()
    if args.verify:
        x=json.loads(args.verify.read_text(encoding="utf8"))
        good=verify(x) and x.get("schema")==SCHEMA
        print(json.dumps({"verified":good}))
        return 0 if good else 2
    row=collect()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(normalized(row)+b"\n")
    print(json.dumps(row,ensure_ascii=False,sort_keys=True))
    return 0
if __name__=="__main__":
    raise SystemExit(main())
