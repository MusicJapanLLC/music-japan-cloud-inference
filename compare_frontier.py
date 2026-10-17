#!/usr/bin/env python3
"""Evidence-gated high-quality model selection.

Only genuine signed model receipts from the same 12-prompt benchmark are compared.
Greater parameter count or a Claude-like name never implies superiority.
No model is downloaded or invoked by this analyzer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from cloud_probe import verify

SCHEMA="mj.cloud-frontier-benchmark/1.0.0"
ALLOWLIST={"baseline-9b","dense-27b","moe-35b","openai-20b","fable-reconstruction-9b","fable-reconstruction-27b","fable-opus-merge-9b"}
CASE_COUNT=12

def load_receipt(path:Path):
    r=json.loads(Path(path).read_text(encoding="utf8"))
    if not verify(r) or r.get("schema")!=SCHEMA:
        raise ValueError("UNVERIFIED_RECEIPT")
    if r.get("profile") not in ALLOWLIST:
        raise ValueError("UNKNOWN_MODEL_PROFILE")
    if r.get("anthropic_original_model") is not False:
        raise ValueError("PROPRIETARY_MODEL_PROVENANCE_UNVERIFIED")
    if r.get("external_model_credits")!=0 or r.get("new_spend_usd")!=0:
        raise ValueError("CREDIT_OR_PAID_RUN")
    if r.get("cloud_runner")!="github_hosted_public_linux_cpu":
        raise ValueError("UNVERIFIED_EXECUTION_CLASS")
    if not r.get("inference_confirmed"):
        raise ValueError("INFERENCE_NOT_CONFIRMED")
    if r.get("total")!=CASE_COUNT or len(r.get("results",[]))!=CASE_COUNT:
        raise ValueError("INCOMPLETE_BENCHMARK")
    ids=[v.get("id") for v in r["results"]]
    if len(ids)!=len(set(ids)) or any(v.get("status")!=200 for v in r["results"]):
        raise ValueError("MISSING_OR_NON_200_RESULT")
    if sum(v.get("passed") is True for v in r["results"])!=r["score"]:
        raise ValueError("QUALITY_SCORE_MISMATCH")
    return r

def rank(receipts:list[dict])->dict:
    if not receipts:
        return {"status":"INSUFFICIENT_EVIDENCE","leader":None}
    if len({x["profile"] for x in receipts})!=len(receipts):
        raise ValueError("DUPLICATE_PROFILE")
    ids=[tuple(x["id"] for x in r["results"]) for r in receipts]
    if len(set(ids))!=1:
        raise ValueError("NON_COMPARABLE_BENCHMARK")
    def metrics(r):
        times=[x["seconds"] for x in r["results"] if isinstance(x.get("seconds"),(int,float))]
        if len(times)!=CASE_COUNT or not all(x>0 for x in times):
            raise ValueError("INVALID_TIMINGS")
        return {
            "model":r["model_id"],"profile":r["profile"],"quality":r["score"],
            "total":r["total"],"median_seconds":round(median(times),2),
            "total_seconds":round(sum(times),2), "receipt_sha256":r["sha256"],
        }
    rows=[metrics(r) for r in receipts]
    rows.sort(key=lambda x:(-x["quality"],x["median_seconds"]))
    baseline=next((x for x in rows if x["profile"]=="baseline-9b"),None)
    winner=rows[0]
    ready=bool(winner["quality"]>=9 and (baseline is None or winner["quality"]>baseline["quality"]))
    # Do not claim an upgrade if no measured improvement over baseline.
    return {
        "status":"COMPARED" if len(rows)>=2 else "ONLY_ONE_BENCHMARK",
        "recommendation":"RESEARCH_UPGRADE_CANDIDATE" if ready else "DO_NOT_AUTO_UPGRADE",
        "leader":winner, "ranked":rows,
        "notes":[
            "12 synthetic prompts do not establish frontier-level or proprietary-model equivalence",
            "Model comparisons use quantized weights; 2-bit accuracy may degrade",
            "GitHub runner is ephemeral, not an always-on API endpoint",
            "No silent routing to any substitute model",
        ]
    }
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("receipts",nargs="+",type=Path)
    a=ap.parse_args()
    results=rank([load_receipt(p) for p in a.receipts])
    print(json.dumps(results,indent=2,ensure_ascii=False,sort_keys=True))
if __name__=="__main__":
    main()
