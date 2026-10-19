#!/usr/bin/env python3
"""Music Japan real test-time-compute profiles for owner-run public GGUF inference.

MAX and ULTRA are Music Japan orchestration profiles, NOT names for a proprietary
model, and do not imply native API effort beyond gpt-oss reasoning_effort=high.
Each preset uses real pinned gpt-oss-20b preliminary inference followed by
original public gpt-oss-120b question-grounded final inference.
"""
from __future__ import annotations

import json

SCHEMA = "mj.test-time-compute-profiles/1.0"
PRESETS = {
    "max": {
        "label": "MAX",
        "draft_model": "openai-20b",
        "draft_rounds": 2,
        "draft_tokens_per_round": 1280,
        "draft_effort": "high",
        "final_model": "openai-gpt-oss-120b",
        "final_rounds": 1,
        "final_tokens_per_round": 384,
        "final_effort": "high",
        "independent_tool_verifier": True,
        "real_model_api_credits": 0,
        "new_spend_usd": 0,
    },
    "ultra": {
        "label": "ULTRA",
        "draft_model": "openai-20b",
        "draft_rounds": 3,
        "draft_tokens_per_round": 2048,
        "draft_effort": "high",
        "final_model": "openai-gpt-oss-120b",
        "final_rounds": 2,
        "final_tokens_per_round": 512,
        "final_effort": "high",
        "independent_tool_verifier": True,
        "real_model_api_credits": 0,
        "new_spend_usd": 0,
    },
}

def profile(name:str)->dict:
    if name not in PRESETS:
        raise ValueError("UNKNOWN_REASONING_PRESET")
    result={"schema":SCHEMA,"preset":name,**PRESETS[name]}
    if result["draft_model"]!="openai-20b" or result["final_model"]!="openai-gpt-oss-120b":
        raise ValueError("UNVERIFIED_MODEL_ROUTE")
    if (result["real_model_api_credits"]!=0 or result["new_spend_usd"]!=0
        or result["draft_effort"]!="high" or result["final_effort"]!="high"):
        raise ValueError("MODEL_CREDIT_OR_THINKING_POLICY_FAILED")
    return result

def validate_proposal(receipt:dict, original_question:str, expected_mode:str)->str:
    """Validate gpt-oss-20b proposal, question binding and actual local inference."""
    from cloud_probe import verify
    from cloud_owner_chat import SCHEMA as CHAT_SCHEMA, CFG
    import hashlib
    mode=profile(expected_mode)
    if not isinstance(receipt,dict) or not verify(receipt) or receipt.get("schema")!=CHAT_SCHEMA:
        raise ValueError("UNVERIFIED_DRAFT_RECEIPT")
    if receipt.get("model_id")!=CFG["model"] or receipt.get("model_weight_sha256")!=CFG["sha256"]:
        raise ValueError("WRONG_MODEL_DRAFT")
    if receipt.get("model_revision")!=CFG["revision"] or not receipt.get("real_local_linux_model_inference"):
        raise ValueError("DRAFT_INFERENCE_NOT_PROVEN")
    if (receipt.get("external_model_api_calls")!=0 or
        receipt.get("paid_compute_created") is not False or
        receipt.get("rounds")!=mode["draft_rounds"] or
        receipt.get("reasoning_effort")!=mode["draft_effort"]):
        raise ValueError("DRAFT_POLICY_MISMATCH")
    if receipt.get("token_budget_per_round")!=mode["draft_tokens_per_round"]:
        raise ValueError("DRAFT_THINKING_BUDGET_MISMATCH")
    if receipt.get("prompt_sha256")!=hashlib.sha256(original_question.encode("utf8")).hexdigest():
        raise ValueError("DRAFT_QUESTION_IDENTITY_MISMATCH")
    result=(receipt.get("selection") or {}).get("final_answer")
    if not isinstance(result,str) or not result.strip():
        raise ValueError("DRAFT_NO_FINAL_TEXT")
    if len(result)>30000:
        raise ValueError("DRAFT_ANSWER_TOO_LONG")
    return result.strip()[:1600]
