#!/usr/bin/env python3
"""Prepare a public Fable-inspired GGUF for offline local/cloud serving.

No proprietary API, secrets or billable service. Requires explicit --download
for network access. All weights pinned by Hugging Face commit + byte SHA256.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cloud_infer_frontier import MODEL_MAP, download, verify_weight, weight_url

PROFILES = ("fable-reconstruction-9b", "fable-reconstruction-27b", "fable-opus-merge-9b")

def prepare(profile: str, directory: Path, fetch: bool=False) -> dict:
    if profile not in PROFILES:
        raise ValueError("PUBLIC_RECONSTRUCTION_PROFILE_REQUIRED")
    cfg=MODEL_MAP[profile]
    if cfg.get("reconstruction") is not True or cfg.get("public_license")!="apache-2.0":
        raise ValueError("CHECKPOINT_PROVENANCE_NOT_ALLOWLISTED")
    dest=directory / cfg["filename"]
    result={
        "profile":profile,"model_id":cfg["model"],"base_model":cfg["base_model"],
        "model_repo":cfg["repo"],"model_revision":cfg["revision"],
        "license_declared_by_publisher":cfg["public_license"],
        "original_anthropic_fable_or_opus":False,
        "source":weight_url(cfg),"expected_bytes":cfg["bytes"],
        "expected_sha256":cfg["sha256"],"target_filename":cfg["filename"],
        "proprietary_llm_api_calls":0,"paid_cloud_jobs_created":0
    }
    if fetch:
        directory.mkdir(parents=True,exist_ok=True)
        result.update(download(cfg,dest))
        result["status"]="VERIFIED_WEIGHTS_READY"
    elif dest.is_file():
        result.update(verify_weight(dest,cfg))
        result["status"]="VERIFIED_WEIGHTS_READY"
    else:
        result["status"]="DRY_RUN_NOT_DOWNLOADED"
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile",choices=PROFILES,default=PROFILES[0])
    p.add_argument("--directory",type=Path,default=Path(__file__).resolve().parent.parent/"models")
    p.add_argument("--download",action="store_true",
                   help="explicit public file download (not proprietary API)")
    args=p.parse_args()
    print(json.dumps(prepare(args.profile,args.directory,args.download),indent=2,ensure_ascii=False,sort_keys=True))
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,ValueError,RuntimeError) as e:
        print(json.dumps({"status":"ERROR","reason":str(e)[:150]}),file=sys.stderr)
        sys.exit(2)
