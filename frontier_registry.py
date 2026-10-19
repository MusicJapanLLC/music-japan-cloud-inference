#!/usr/bin/env python3
"""Music Japan Frontier Registry: official *public* open-weight provenance.

Researches model repository metadata only. No weights, hosted inference,
account access, paid jobs, API credits, tokens, cookie jars, or private data.

Proves WHAT is published, not that hardware has been allocated or that the
model is equivalent to a proprietary frontier model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

SCHEMA = "mj.frontier-catalog/1.0.0"
MAX_META_BYTES = 2_000_000
MODEL_TARGETS = {
    "openai/gpt-oss-120b": {"publisher": "openai", "category": "open_reasoning", "expected_license": "apache-2.0"},
    "Qwen/Qwen3.5-397B-A17B": {"publisher": "Qwen", "category": "frontier_moe_vision", "expected_license": "apache-2.0"},
    "zai-org/GLM-5.2": {"publisher": "zai-org", "category": "frontier_coding_reasoning", "expected_license": "mit"},
    "MiniMaxAI/MiniMax-M3": {"publisher": "MiniMaxAI", "category": "frontier_multimodal", "expected_license": "minimax-community"},
    "Qwen/Qwen3.5-35B-A3B": {"publisher": "Qwen", "category": "efficient_reasoning", "expected_license": "apache-2.0"},
}

def canonical(record: dict) -> bytes:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")

def receipt(record:dict)->dict:
    data = {k:v for k,v in record.items() if k != "receipt_sha256"}
    return {**data,"receipt_sha256":hashlib.sha256(canonical(data)).hexdigest()}

def verify(record:dict)->bool:
    return (isinstance(record,dict) and record.get("schema")==SCHEMA and
            isinstance(record.get("receipt_sha256"),str) and
            receipt(record)["receipt_sha256"]==record["receipt_sha256"])

def validate_registry_url(model:str)->str:
    if model not in MODEL_TARGETS:
        raise ValueError("MODEL_NOT_ALLOWLISTED")
    return "https://huggingface.co/api/models/"+model+"?blobs=true"

def fetch_public_meta(model:str, opener=None)->dict:
    url=validate_registry_url(model)
    req=urllib.request.Request(url,method="GET",
         headers={"Accept":"application/json","User-Agent":"MusicJapanFrontierResearch/1.0"})
    opener=opener or urllib.request.urlopen
    with opener(req,timeout=12) as res:
        if getattr(res,"status",200)!=200:
            raise ValueError("NON_200_PUBLIC_REGISTRY")
        if hasattr(res,"geturl") and res.geturl()!=url:
            raise ValueError("REGISTRY_REDIRECT_NOT_PINNED")
        payload=res.read(MAX_META_BYTES+1)
        if len(payload)>MAX_META_BYTES:
            raise ValueError("REGISTRY_METADATA_TOO_LARGE")
    obj=json.loads(payload)
    if not isinstance(obj,dict) or obj.get("id")!=model:
        raise ValueError("REGISTRY_IDENTITY_MISMATCH")
    return obj

def summarize(model:str,raw:dict)->dict:
    spec=MODEL_TARGETS.get(model)
    if not spec or not isinstance(raw,dict) or raw.get("id")!=model:
        raise ValueError("SOURCE_IDENTITY_UNVERIFIED")
    owner=model.split("/",1)[0]
    if owner!=spec["publisher"]:
        raise ValueError("PUBLISHER_MISMATCH")
    model_sha=raw.get("sha")
    if not isinstance(model_sha,str) or len(model_sha)!=40 or any(c not in "0123456789abcdef" for c in model_sha):
        raise ValueError("MODEL_REVISION_NOT_SHA1")
    license_str=(raw.get("cardData") or {}).get("license")
    siblings=raw.get("siblings")
    if not isinstance(siblings,list):
        raise ValueError("NO_FILE_INVENTORY")
    count=0
    total=0
    official_file_hashes=0
    for s in siblings:
        if not isinstance(s,dict):
            continue
        name=s.get("rfilename","")
        if not (isinstance(name,str) and name.endswith((".safetensors",".gguf"))):
            continue
        size=s.get("size")
        if isinstance(size,int) and not isinstance(size,bool) and size>0:
            total+=size
            count+=1
        sha=(s.get("lfs") or {}).get("sha256")
        if isinstance(sha,str) and len(sha)==64:
            official_file_hashes+=1
    license_matches=license_str==spec["expected_license"]
    gated=raw.get("gated",None)
    private=raw.get("private",None)
    public=(private is False and gated in (False,None))
    return {
        "model":model, "category":spec["category"],
        "official_publisher":spec["publisher"], "revision":model_sha,
        "declared_license":license_str,
        "license_matches_expected":license_matches,
        "commercial_review_required":license_str not in ("apache-2.0","mit"),
        "public_metadata_accessible":public,
        "weight_file_count":count,
        "weight_bytes_with_reported_size":total,
        "weight_gib_with_reported_size":round(total/2**30,2),
        "weight_size_estimate_incomplete":count<sum(1 for s in siblings if isinstance(s,dict) and
                                        str(s.get("rfilename","")).endswith((".safetensors",".gguf"))),
        "sha256_weight_file_metadata_count":official_file_hashes,
        "public_weights_access_claim":bool(public and count and license_matches),
        "weights_downloaded":False,
        "runtime_tested":False,
        "external_credits_used":0,
    }

def run(online=False, fetch=fetch_public_meta, time=None)->dict:
    entries=[]
    for model in MODEL_TARGETS:
        if not online:
            entries.append({"model":model,"status":"NOT_QUERIED","weights_downloaded":False})
            continue
        try:
            item=summarize(model,fetch(model))
            item["status"]="PUBLIC_METADATA_OBSERVED"
            entries.append(item)
        except (OSError,ValueError,urllib.error.URLError,json.JSONDecodeError) as ex:
            entries.append({"model":model,"status":"NOT_VERIFIED","error_code":type(ex).__name__,
                            "weights_downloaded":False})
    return receipt({
        "schema":SCHEMA,
        "observed_utc": (time or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds"),
        "scope":"anonymous official publisher repo metadata only",
        "entries":entries, "model_inference_calls":0,"paid_compute_allocations":0,
        "customer_data_accessed":False,
        "verified_hosted_model_inference":False,
    })

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--online",action="store_true",help="GET only public official HF repo metadata")
    p.add_argument("--out",type=Path)
    p.add_argument("--verify",type=Path)
    args=p.parse_args(argv)
    if args.verify:
        j=json.loads(args.verify.read_text("utf8"))
        passed=verify(j)
        print(json.dumps({"verified":passed,"observed_models":len(j.get("entries",[]))}))
        return 0 if passed else 2
    info=run(online=args.online)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(canonical(info)+b"\n")
    print(json.dumps(info,sort_keys=True,ensure_ascii=False))
    return 0

if __name__=="__main__":
    sys.exit(main())
