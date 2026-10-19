#!/usr/bin/env python3
"""Evidence-gated public GitHub Summary for ORIGINAL open gpt-oss-120b answers.

Only verifies and displays a signed real-inference receipt. Never reads
customer prompts, secret keys, model weights, or host credential files.
"""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path
from cloud_probe import verify
from official120b_chat import SCHEMA
import experimental_gptoss120b as original
from reasoning_profiles import profile as reasoning_profile

def read_verified(path:Path)->dict:
    record=json.loads(path.read_text(encoding="utf8"))
    if not verify(record) or record.get("schema")!=SCHEMA:
        raise ValueError("RECEIPT_SHA_OR_SCHEMA_FAILED")
    if record.get("model_id")!=original.ID or record.get("official_weight_sha256")!=original.SHA256:
        raise ValueError("ORIGINAL_MODEL_IDENTITY_UNVERIFIED")
    if record.get("model_original_openai_gpt_oss_120b") is not True or record.get("actual_local_120b_inference") is not True:
        raise ValueError("REAL_120B_INFERENCE_REQUIRED")
    if record.get("model_provider_credits_used")!=0 or record.get("paid_compute_allocated") is not False:
        raise ValueError("NO_SPEND_CONDITION_VIOLATED")
    mode=record.get("research_mode")
    if mode in ("max","ultra"):
        preset=reasoning_profile(mode)
        digest=record.get("draft_proposal_receipt_sha256")
        if (not isinstance(digest,str) or len(digest)!=64 or
            any(c not in "0123456789abcdef" for c in digest) or
            record.get("external_open_model_proposal_applied") is not True):
            raise ValueError("MULTI_MODEL_PROVENANCE_REQUIRED")
        if (record.get("rounds")!=preset["final_rounds"] or
            record.get("requested_tokens_each_round")!=preset["final_tokens_per_round"] or
            record.get("reasoning_effort")!=preset["final_effort"]):
            raise ValueError("MAX_ULTRA_COMPUTE_POLICY_NOT_SATISFIED")
    elif mode not in (None,"direct_120b"):
        raise ValueError("UNRECOGNIZED_REASONING_PRESET")
    selected=record.get("selection") or {}
    if not isinstance(selected.get("final_answer"),str) or not selected["final_answer"].strip():
        raise ValueError("VISIBLE_FINAL_REPLY_REQUIRED")
    return record

def render(record:dict)->str:
    answer=record["selection"]["final_answer"]
    if len(answer)>40000:raise ValueError("ANSWER_TOO_LONG")
    source=record["selection"]["source"]
    evidence=record["sha256"]
    return (
      "## Music Japan — OpenAI Original gpt-oss-120b Answer\n\n"
      "**Model:** "+html.escape(original.ID)+"\n\n"
      "**Publisher's original open model:** yes (GGUF reproduction of public checkpoint)\n\n"
      "**Thinking level:** "+html.escape(record["reasoning_effort"])+"\n\n"
      "**Music Japan orchestrated mode:** "+html.escape(str(record.get("research_mode") or "direct_120b"))+"\n\n"
      "**Cross-model 20B proposal verified:** "+str(bool(record.get("external_open_model_proposal_applied")))+"\n\n"
      "**Model reasoning rounds:** "+str(record["rounds"])+"\n\n"
      "**Generation and startup time:** "+str(record["elapsed_total_seconds"])+" seconds\n\n"
      "**Independent check:** "+html.escape(str(record["selection"]["verification"].get("status","ABSTAIN")))+"\n\n"
      "**Selection source:** "+html.escape(source)+"\n\n"
      "**Real model API credits:** 0\n\n"
      "**Model checksum:** "+html.escape(record["official_weight_sha256"])+"\n\n"
      "**Proof checksum:** "+html.escape(evidence)+"\n\n"
      "### Model's final answer\n\n<pre>"+html.escape(answer)+"</pre>\n\n"
      "_Public research input only. The GitHub Actions run is ephemeral; avoid confidential prompts._\n"
    )

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    record=read_verified(args.receipt)
    rendered=render(record)
    if args.out:
        args.out.write_text(rendered,encoding="utf8")
    else:
        path=os.environ.get("GITHUB_STEP_SUMMARY")
        if not path:raise ValueError("SUMMARY_OUTPUT_NOT_AUTHORIZED")
        with Path(path).open("a",encoding="utf8") as fp:
            fp.write(rendered)
    print(json.dumps({"summary_ok":True,"model":record["model_id"],
        "rounds":record["rounds"],"receipt":record["sha256"]},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
