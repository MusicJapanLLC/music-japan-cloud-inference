#!/usr/bin/env python3
"""Independent retrospective ensemble of real signed public model receipts.

NOT a live inference run. Compares simple majority voting, independent tools,
and the unattainable oracle ceiling separately without feeding correct labels
into the ensemble selector. No provider credits or network calls.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from cloud_probe import signed_record, normalized
from cloud_infer_frontier import CHALLENGES,grade
from compare_frontier import load_receipt
from objective_verifier import apply_verifier

SCHEMA="mj.retro-ensemble-analysis/1.0"

def canonical_vote(answer:str,case:dict)->str:
    if not isinstance(answer,str):return ""
    s=answer.strip()
    if case["grade"]=="casefold":
        return s.casefold()
    if case["grade"]=="json":
        try:
            return json.dumps(json.loads(s),sort_keys=True,separators=(",",":"))
        except (ValueError,TypeError):
            return s
    return s

def ensemble_decision(receipts:list[dict],index:int)->dict:
    if not receipts:
        raise ValueError("NO_MODEL_RECEIPTS")
    case=CHALLENGES[index]
    votes=Counter(canonical_vote(r["results"][index].get("answer",""),case)
                  for r in receipts if r["results"][index].get("status")==200)
    votes.pop("",None)
    if not votes:
        return {"status":"ABSTAIN","answer":"","votes":0,"agreement":0}
    top=votes.most_common()
    high=top[0][1]
    leaders=sorted(k for k,v in votes.items() if v==high)
    if len(leaders)>1:
        return {"status":"TIED_ABSTAIN","answer":"","votes":len(votes),"agreement":high,
                "candidates":len(leaders)}
    chosen=leaders[0]
    # Choose a real answer with the majority normalized form without rewriting.
    original=next(r["results"][index]["answer"] for r in receipts
                  if canonical_vote(r["results"][index]["answer"],case)==chosen)
    return {"status":"MAJORITY" if high>=2 else "SINGLE_MODEL_UNCONFIRMED",
            "answer":original,"votes":sum(votes.values()),"agreement":high}

def analyze(receipts:list[dict])->dict:
    if len(receipts)<2:
        raise ValueError("NEED_AT_LEAST_TWO_INDEPENDENT_MODELS")
    names=[r["profile"] for r in receipts]
    if len(set(names))!=len(names):
        raise ValueError("DUPLICATE_PROFILE")
    expected_ids=[x["id"] for x in CHALLENGES]
    if any([x["id"] for x in r["results"]]!=expected_ids for r in receipts):
        raise ValueError("NON_COMPARABLE_EVAL_TASKS")
    rows=[]
    for index,case in enumerate(CHALLENGES):
        selected=ensemble_decision(receipts,index)
        original=selected["answer"]
        check=apply_verifier(case["prompt"],original) if original else {
            "final_answer":"","source":"abstain","verification":{"status":"ABSTAIN"}}
        upper=any(grade(r["results"][index]["answer"],case) for r in receipts)
        rows.append({"id":case["id"],"decision":selected["status"],
            "independent_agreement":selected.get("agreement",0),
            "vote_count":selected.get("votes",0),
            "voted_answer":original[:200],
            "majority_passed":grade(original,case) if original else False,
            "tool_assisted_answer":check["final_answer"][:200],
            "tool_source":check["source"],
            "tool_assisted_passed":grade(check["final_answer"],case),
            "oracle_any_model_correct":upper})
    totals={
        "best_single_model_correct":max(r["score"] for r in receipts),
        "majority_ensemble_correct":sum(x["majority_passed"] for x in rows),
        "majority_plus_tools_correct":sum(x["tool_assisted_passed"] for x in rows),
        "oracle_upper_bound_correct":sum(x["oracle_any_model_correct"] for x in rows),
        "abstentions":sum(x["decision"].endswith("ABSTAIN") for x in rows),
        "tasks":len(rows)
    }
    return signed_record({
        "schema":SCHEMA,"profiles":names,
        "source_receipt_sha256":[x["sha256"] for x in receipts],
        "real_model_outputs_reused":True,
        "new_model_calls":0,"external_model_credits_used":0,
        "oracle_used_in_selection":False,
        "retrospective_only_not_live_service":True,
        "totals":totals,"cases":rows,
        "explanation":"Multiple models and tool verification were compared on actual previous model answers; the oracle is a nondeployable upper bound.",
    })

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipts",type=Path,nargs="+")
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    result=analyze([load_receipt(f) for f in args.receipts])
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes(normalized(result)+b"\n")
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))

if __name__=="__main__":
    main()
