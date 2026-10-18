#!/usr/bin/env python3
"""Pinned, offline, owner-operated DeepSeek V4 Pro 0813 runtime preparation.

The checkpoint is the publisher's 1.6T model plus its DSpark draft, not weights
trained by Music Japan. Planning is NOT GPU allocation or inference evidence.
No cloud provisioning, paid model APIs, implicit downloads or model fallback.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys

MODEL='deepseek-ai/DeepSeek-V4-Pro-0813'
REVISION='72e1d3230f6c080a530b0a1d46f8eb4602340597'
SERVED_MODEL='mj-deepseek-v4-pro-0813'
WEIGHT_BYTES=892744322880
REPORTED_PARAMETERS=1650497936906
ENGINE_VERSION='0.25.0'
LOCK_SHA256='2ce8f718065196b23a0d444d085133577c0ce3f4e4d852df5ebdd0b72f7294c9'
DEFAULT_PROFILE='cuda-8'
# Closed, explicit launch profiles. These are configuration contracts, not GPU
# benchmarks or permission to allocate hardware. Keep the existing default.
PROFILES={
    'cuda-8':{'gpu_count':8,'tensor_parallel_size':8,'data_parallel_size':1,
              'minimum_compute_capability':9.0,'maximum_compute_capability':None,
              'gpu_memory_utilization':'0.85','max_num_seqs':4,
              'draft_sample_method':'probabilistic'},
    # The fixed publisher README's single-node 4xGB300 recipe uses DP4 + EP,
    # not TP4: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/
    # 72e1d3230f6c080a530b0a1d46f8eb4602340597/README.md
    'gb300-4':{'gpu_count':4,'tensor_parallel_size':1,'data_parallel_size':4,
               'minimum_compute_capability':10.0,'maximum_compute_capability':11.0,
               'gpu_memory_utilization':'0.90','max_num_seqs':1,
               'draft_sample_method':'greedy'},
}
MAX_LOCK_BYTES=500_000
CONTEXTS=(8192,16384,32768)


class ContractError(ValueError):
    """Application-authored error codes only; no credential-bearing errors."""


def launch_profile(name):
    if not isinstance(name,str) or name not in PROFILES:
        raise ContractError('UNSUPPORTED_LAUNCH_PROFILE')
    return PROFILES[name]


def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,
                      separators=(',',':'),allow_nan=False).encode('utf8')


def loads(text):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ContractError('DUPLICATE_JSON_KEY')
            result[k]=v
        return result
    try:
        return json.loads(text,object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ContractError('NONFINITE_JSON')))
    except (UnicodeError,json.JSONDecodeError) as exc:
        raise ContractError('INVALID_JSON') from exc


def _sha(value):
    return isinstance(value,str) and re.fullmatch('[a-f0-9]{64}',value) is not None


def safe_path(value):
    if (not isinstance(value,str) or not 1<=len(value)<=180
        or not re.fullmatch('[A-Za-z0-9_./-]+',value)
        or any(x in ('','.','..') for x in value.split('/'))
        or PurePosixPath(value).is_absolute()):
        raise ContractError('UNSAFE_WEIGHT_PATH')
    return value


def validate_lock(data):
    fields={'schema','repo','revision','license','reported_parameters','config',
            'config_sha256','license_sha256','weights','total_weight_bytes','auxiliary_files'}
    if not isinstance(data,dict) or set(data)!=fields:
        raise ContractError('INVALID_LOCK_FIELDS')
    if (data['schema']!='mj.trillion-model-lock/1.0' or data['repo']!=MODEL
        or data['revision']!=REVISION or data['license']!='mit'
        or type(data['reported_parameters']) is not int
        or data['reported_parameters']!=REPORTED_PARAMETERS
        or type(data['total_weight_bytes']) is not int
        or data['total_weight_bytes']!=WEIGHT_BYTES):
        raise ContractError('WRONG_OR_UNPINNED_1_6T_CHECKPOINT')
    config=data['config']
    if (not isinstance(config,dict) or config.get('model_type')!='deepseek_v4'
        or type(config.get('dspark_block_size')) is not int
        or config['dspark_block_size']<=0
        or not all(k in config for k in ('dspark_noise_token_id','dspark_target_layer_ids','dspark_markov_rank'))):
        raise ContractError('NATIVE_DSPARK_CHECKPOINT_REQUIRED')
    if not _sha(data['config_sha256']) or not _sha(data['license_sha256']):
        raise ContractError('UNPINNED_AUXILIARY_FILES')
    auxiliary=data['auxiliary_files']
    required={'LICENSE','config.json','generation_config.json','model.safetensors.index.json',
              'tokenizer.json','tokenizer_config.json'}
    if not isinstance(auxiliary,list) or len(auxiliary)!=len(required):
        raise ContractError('PINNED_TOKENIZER_ASSETS_REQUIRED')
    names=set()
    for item in auxiliary:
        if (not isinstance(item,dict) or set(item)!={'path','bytes','sha256'}
            or item.get('path') not in required or item['path'] in names
            or type(item.get('bytes')) is not int or not 0<item['bytes']<=32_000_000
            or not _sha(item.get('sha256'))):
            raise ContractError('INVALID_AUXILIARY_FILE_LOCK')
        names.add(item['path'])
        if item['path']=='config.json' and item['sha256']!=data['config_sha256']:
            raise ContractError('CONFIG_HASH_DISAGREES')
        if item['path']=='LICENSE' and item['sha256']!=data['license_sha256']:
            raise ContractError('LICENSE_HASH_DISAGREES')
    entries=data['weights']
    if not isinstance(entries,list) or len(entries)!=66:
        raise ContractError('INCOMPLETE_66_SHARD_LOCK')
    paths=set();total=0
    for item in entries:
        if not isinstance(item,dict) or set(item)!={'path','bytes','sha256'}:
            raise ContractError('INVALID_SHARD_ENTRY')
        name=safe_path(item['path'])
        if (not name.endswith('.safetensors') or name in paths
            or type(item['bytes']) is not int or item['bytes']<=0
            or not _sha(item['sha256'])):
            raise ContractError('INVALID_OR_DUPLICATE_SHARD')
        paths.add(name);total+=item['bytes']
    if total!=WEIGHT_BYTES:raise ContractError('SHARD_TOTAL_MISMATCH')
    return copy.deepcopy(data)


def read_lock(path):
    path=Path(path)
    if path.stat().st_size>MAX_LOCK_BYTES:raise ContractError('LOCK_SIZE_LIMIT')
    data=loads(path.read_bytes())
    if hashlib.sha256(canonical(data)).hexdigest()!=LOCK_SHA256:
        raise ContractError('SOURCE_LOCK_DIGEST_MISMATCH')
    return validate_lock(data)


def hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def verify_files(root,entries):
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ContractError('INVALID_OWNED_SNAPSHOT')
    base=root.resolve(strict=True);total=0
    for item in entries:
        name=safe_path(item['path']);path=base/name
        try:
            current=base
            for part in PurePosixPath(name).parts:
                current=current/part
                if current.is_symlink():raise ContractError('SYMLINK_IN_SNAPSHOT')
            info=path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size!=item['bytes']:
                raise ContractError('WEIGHT_SIZE_OR_TYPE_MISMATCH')
            if hash_file(path)!=item['sha256']:raise ContractError('WEIGHT_SHA256_MISMATCH')
            total+=info.st_size
        except OSError as exc:raise ContractError('WEIGHT_FILE_UNAVAILABLE') from exc
    return {'verified_files':len(entries),'verified_bytes':total}


def verify_snapshot(lock,root):
    data=validate_lock(lock);root=Path(root)
    assets=verify_files(root,data['auxiliary_files'])
    index=loads((root/'model.safetensors.index.json').read_bytes())
    mapping=index.get('weight_map') if isinstance(index,dict) else None
    names={item['path'] for item in data['weights']}
    if not isinstance(mapping,dict) or not mapping or not set(mapping.values())<=names:
        raise ContractError('TENSOR_INDEX_SHARD_MISMATCH')
    result=verify_files(root,data['weights'])
    return {**result,'verified_auxiliary_files':assets['verified_files']}


def parse_gpu_csv(text):
    cards=[]
    try:
        for row in csv.reader(text.splitlines()):
            if not row:continue
            if len(row)!=5:raise ContractError('INVALID_GPU_FIELDS')
            ident,name,total,free,cap=(x.strip() for x in row)
            total_bytes=int(total)*2**20;free_bytes=int(free)*2**20;cc=float(cap)
            if (not ident or not name or total_bytes<=0 or not 0<=free_bytes<=total_bytes
                or not math.isfinite(cc) or not 0<cc<100):raise ContractError('INVALID_GPU_VALUES')
            cards.append({'uuid':ident,'name':name[:120],'total_bytes':total_bytes,
                          'free_bytes':free_bytes,'compute_capability':cc})
    except (ValueError,TypeError) as exc:raise ContractError('INVALID_GPU_REPORT') from exc
    return cards


def collect_gpus():
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.free,compute_cap',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,
                         timeout=8,check=False)
    except (OSError,subprocess.TimeoutExpired):return []
    return parse_gpu_csv(p.stdout) if p.returncode==0 else []


def preflight(lock,cards,engine_version,profile=DEFAULT_PROFILE):
    validate_lock(lock)
    selected=launch_profile(profile)
    if not isinstance(cards,list):raise ContractError('INVALID_GPU_INVENTORY')
    uuids=set()
    for card in cards:
        if (not isinstance(card,dict) or not isinstance(card.get('uuid'),str)
            or not card['uuid'] or card['uuid'] in uuids):raise ContractError('DUPLICATE_GPU_IDENTITY')
        uuids.add(card['uuid'])
        for name in ('total_bytes','free_bytes'):
            if type(card.get(name)) is not int or card[name]<0:raise ContractError('INVALID_GPU_MEMORY')
        if card['free_bytes']>card['total_bytes']:raise ContractError('INVALID_GPU_MEMORY')
        cc=card.get('compute_capability')
        if type(cc) not in (float,int) or not math.isfinite(cc):raise ContractError('INVALID_GPU_ARCH')
    gpu_count=selected['gpu_count']
    # Preserve the initial 20% planning margin, calculated for the selected
    # sharding degree. Real rank replication, topology and KV fit need a launch.
    per_card=(WEIGHT_BYTES*6+5*gpu_count-1)//(5*gpu_count)+2**30
    reasons=[]
    if len(cards)!=gpu_count:reasons.append('GPU_COUNT_DOES_NOT_MATCH_PROFILE')
    if any(c['free_bytes']<per_card for c in cards):reasons.append('FREE_VRAM_BELOW_PLANNING_RESERVE')
    if any(c['compute_capability']<selected['minimum_compute_capability']
           or (selected['maximum_compute_capability'] is not None
               and c['compute_capability']>=selected['maximum_compute_capability']) for c in cards):
        reasons.append('GPU_ARCH_UNSUPPORTED_FOR_PROFILE')
    if engine_version!=ENGINE_VERSION:reasons.append('VLLM_VERSION_PIN_NOT_SATISFIED')
    return {'model':MODEL,'revision':REVISION,'eligible_for_launch':not reasons,'reasons':reasons,
            'profile':profile,'required_gpus':gpu_count,
            'tensor_parallel_size':selected['tensor_parallel_size'],
            'data_parallel_size':selected['data_parallel_size'],
            'topology_verified':False,
            'visible_gpus':len(cards),'minimum_free_bytes_per_gpu':per_card,
            'weight_bytes':WEIGHT_BYTES,'memory_requirement_is_estimate':True,
            'weights_verified':False,'model_inference_verified':False,
            'new_gpu_allocations':0,'model_provider_api_calls':0}


def build_command(root,context=32768,speculation=True,profile=DEFAULT_PROFILE):
    if type(context) is not int or context not in CONTEXTS or type(speculation) is not bool:
        raise ContractError('UNSUPPORTED_CONTEXT_OR_SPECULATION')
    selected=launch_profile(profile)
    args=['vllm','serve',str(Path(root).resolve()),'--served-model-name',SERVED_MODEL,
          '--host','127.0.0.1','--port','18189',
          '--tensor-parallel-size',str(selected['tensor_parallel_size']),
          '--distributed-executor-backend','mp','--enable-expert-parallel',
          '--tokenizer-mode','deepseek_v4','--reasoning-parser','deepseek_v4',
          '--tool-call-parser','deepseek_v4','--enable-auto-tool-choice',
          '--max-model-len',str(context),'--max-num-seqs',str(selected['max_num_seqs']),
          '--max-num-batched-tokens','8192',
          '--gpu-memory-utilization',selected['gpu_memory_utilization'],
          '--enable-prefix-caching','--enable-chunked-prefill']
    if profile=='gb300-4':
        args+=['--data-parallel-size','4','--data-parallel-size-local','4',
               '--data-parallel-backend','mp','--kv-cache-dtype','fp8','--block-size','256',
               '--moe-backend','deep_gemm_mega_moe',
               '--attention-config','{"use_fp4_indexer_cache":true}']
    if speculation:
        args+=['--speculative-config',json.dumps({'method':'dspark','num_speculative_tokens':7,
                                                 'draft_sample_method':selected['draft_sample_method']},
                                                separators=(',',':'))]
    return args


def runtime_environment(inherited,own_api_key):
    if (not isinstance(own_api_key,str) or not 32<=len(own_api_key)<=512
        or any(ord(c)<33 or ord(c)>126 for c in own_api_key)):
        raise ContractError('VALID_OWNER_API_KEY_REQUIRED')
    allowed={'PATH','HOME','TMP','TEMP','USERPROFILE','SYSTEMROOT','WINDIR',
             'LD_LIBRARY_PATH','CUDA_HOME','CUDA_VISIBLE_DEVICES'}
    result={k:v for k,v in inherited.items() if k in allowed}
    result.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                  HF_HUB_DISABLE_TELEMETRY='1',VLLM_API_KEY=own_api_key)
    return result


def main(argv=None):
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--lock',type=Path,default=Path(__file__).with_name('model.lock.json'))
    cli.add_argument('--weights-dir',type=Path,required=True)
    cli.add_argument('--profile',choices=tuple(PROFILES),default=DEFAULT_PROFILE,
                     help='Explicit single-node hardware/parallelism profile; never allocates GPUs')
    cli.add_argument('--context',type=int,choices=CONTEXTS,default=32768)
    cli.add_argument('--baseline',action='store_true',help='Disable DSpark for a matched benchmark arm')
    cli.add_argument('--execute',action='store_true')
    cli.add_argument('--api-key-file',type=Path)
    args=cli.parse_args(argv)
    try:
        data=read_lock(args.lock)
        try:version=importlib.metadata.version('vllm')
        except importlib.metadata.PackageNotFoundError:version='not_installed'
        report=preflight(data,collect_gpus(),version,args.profile)
        command=build_command(args.weights_dir,args.context,not args.baseline,args.profile)
        print(json.dumps({**report,'mode':'execute_requested' if args.execute else 'plan',
                          'command':command},ensure_ascii=False),flush=True)
        if not args.execute:return 0
        if not report['eligible_for_launch']:raise ContractError('HARDWARE_OR_ENGINE_PREFLIGHT_FAILED')
        if args.api_key_file is None:raise ContractError('OWNER_AUTH_FILE_REQUIRED')
        key=args.api_key_file.read_text('utf-8-sig').strip()
        env=runtime_environment(os.environ,key)
        proof=verify_snapshot(data,args.weights_dir)
        print(json.dumps({'weights_verified':True,**proof,'model_inference_verified':False}),flush=True)
        # Explicit owner invocation only. No automatic GPU rental or model download.
        return subprocess.run(command,env=env,shell=False,check=False).returncode
    except (ContractError,OSError) as exc:
        code=str(exc) if isinstance(exc,ContractError) else 'LOCAL_INPUT_UNAVAILABLE'
        print(json.dumps({'status':'NOT_LAUNCHED','code':code}),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
