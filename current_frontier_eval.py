#!/usr/bin/env python3
"""Bounded public synthetic own-weight development evaluation; dry run by default."""
from __future__ import annotations
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import quote

LLAMA_COMMIT = '848bbda9041decd87904f7eb5a2990863ecce03b'
PROFILES = {
 'qwen38-27b-iq3s': dict(repo='unsloth/Qwen3.8-27B-GGUF', revision='4ca720788d1e01f1bff70c033e0d0028fd02e502',
   files=[('Qwen3.8-27B-UD-IQ3_S.gguf',12040883104,'d847e2c1e4aa276e4b7b8e9ad7628050e61e165d49ab995407bc36677a6f3864')],
   template=dict(enable_thinking=True, preserve_thinking=True, reasoning_effort='low'), effort='low'),
 'nemotron35-120b-iq3xxs': dict(repo='unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF', revision='75a5829055f69a0359085ae6bbd0a81b60bc05c6',
   files=[('UD-IQ3_XXS/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-IQ3_XXS-00001-of-00003.gguf',7874568,'01150188b68c1ebf4528d095eab5fa12b7ae18acfe7cb9fcc5563c0772badd94'),
          ('UD-IQ3_XXS/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-IQ3_XXS-00002-of-00003.gguf',49997462976,'fbf889020bf9effe03da67985d24a1b5d58ad382fcdd02fa390c09687abc0957'),
          ('UD-IQ3_XXS/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-IQ3_XXS-00003-of-00003.gguf',9758873024,'e69358032031aa6dac6838dffb03ad2d54e510bad2fea10513db7dd27dfbd70b')],
   template=dict(enable_thinking=True, medium_effort=True), effort=None),
}
# A separate explicit comparison: reuse exact reviewed artifacts, never fall
# back automatically from the original thinking-enabled profile.
_nemo = PROFILES['nemotron35-120b-iq3xxs']
PROFILES['nemotron35-120b-iq3xxs-instruct'] = dict(
    repo=_nemo['repo'], revision=_nemo['revision'], files=list(_nemo['files']),
    template=dict(enable_thinking=False), effort=None, context=1024, max_tokens=256)

PROMPT = 'Synthetic public evaluation. Return ONLY JSON with keys arithmetic, reversed, earliest_finish, decision. arithmetic: Compute 19*23, then subtract 37, then divide by 8. reversed: reverse AbcXYZ, then lowercase it. earliest_finish: tasks A=3 days; B=2 days after A; C=5 days after A; D=2 days after B and C; E=1 day after D. Unlimited workers. All start at time 0 when possible. Return total days until E ends. decision: A transcript says "10:00 M: We could charge 30000 next month. 10:01 N: That is only a proposal; price not decided. 10:02 M: Agreed, undecided." Return the status only: proposed, approved, or completed.'
EXPECTED = dict(arithmetic=50, reversed='zyxcba', earliest_finish=11, decision='proposed')
MAX_RESPONSE = 2 * 1024 * 1024
TOTAL_SECONDS = 2100


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf8')


def verify_file(path, item):
    _, size, expected = item
    digest = hashlib.sha256()
    observed = 0
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            observed += len(block)
            digest.update(block)
    if observed != size or digest.hexdigest() != expected:
        raise ValueError('WEIGHT_INTEGRITY_MISMATCH')


def resources(folder, profile):
    ram_kib = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal:'))
    ram = ram_kib / 1024**2
    required = sum(item[1] for item in profile['files']) + 8 * 1024**3
    free = shutil.disk_usage(folder).free
    if ram < 15 or free < required:
        raise ValueError('RESOURCE_GATE_FAILED')
    return dict(ram_gib=round(ram,2), disk_free_gib=round(free/1024**3,2), required_disk_gib=round(required/1024**3,2), cpu_logical=os.cpu_count())


def time_left(deadline, cap):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError('TOTAL_DEVELOPMENT_BUDGET_EXCEEDED')
    return min(cap, left)


def download_weights(folder, profile, deadline):
    paths = []
    for item in profile['files']:
        name, _, _ = item
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.part')
            partial.unlink(missing_ok=True)
            url = f"https://huggingface.co/{profile['repo']}/resolve/{profile['revision']}/{quote(name, safe='/')}"
            try:
                limit = time_left(deadline, 1100)
                subprocess.run(['curl','-q','--fail','--location','--silent','--show-error','--retry','0',
                    '--max-time',str(max(1,int(limit))), '--output',str(partial),url],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=True, timeout=limit,
                    env={key:os.environ[key] for key in ('PATH','LANG','LC_ALL') if key in os.environ})
                verify_file(partial, item)
                os.replace(partial,path)
            finally:
                partial.unlink(missing_ok=True)
        else:
            verify_file(path,item)
        time_left(deadline, 1)
        paths.append(path)
    return paths


def request(port, token, route, timeout, body=None):
    connection = http.client.HTTPConnection('127.0.0.1',port,timeout=timeout)
    headers = {'Authorization':'Bearer '+token}
    if body is not None:
        headers['Content-Type']='application/json'
    try:
        connection.request('GET' if body is None else 'POST',route,
                           None if body is None else canonical(body),headers)
        response = connection.getresponse()
        raw = response.read(MAX_RESPONSE+1)
        if len(raw)>MAX_RESPONSE:
            raise ValueError('RESPONSE_TOO_LARGE')
        value = json.loads(raw)
        if response.status != 200 or not isinstance(value,dict):
            raise ValueError('LOCAL_RUNTIME_HTTP_FAILURE')
        return value
    finally:
        connection.close()


def grade_text(text):
    """Require strict JSON and exact fields/values; equal int/float numbers pass, bool never does."""
    def matches(value):
        return (type(value) is dict and set(value)==set(EXPECTED)
                and all((type(value[key]) in (int,float) if type(wanted) is int else type(value[key]) is str)
                        and value[key]==wanted for key,wanted in EXPECTED.items()))
    try:
        value=json.loads(text)
        strict=True
    except (ValueError,TypeError):
        value=None
        strict=False
    normalized=False
    if not strict and isinstance(text,str):
        stripped=text.strip()
        if stripped.startswith('```json\n') and stripped.endswith('\n```'):
            try:
                value=json.loads(stripped[len('```json\n'):-len('\n```')])
                normalized=True
            except ValueError:
                pass
    return dict(strict_json=strict, strict_contract_passed=bool(strict and matches(value)),
                visible_json_fence_parsed=normalized,
                fence_contract_passed=bool(normalized and matches(value)))


def runtime_limits(profile):
    context=profile.get('context',2048)
    maximum=profile.get('max_tokens',1024)
    if (type(context) is not int or type(maximum) is not int
            or not 256 <= context <= 2048 or not 1 <= maximum <= min(context,1024)):
        raise ValueError('INVALID_PROFILE_LIMITS')
    return context,maximum


def model_body(profile_name):
    profile=PROFILES[profile_name]
    body=dict(model=profile_name,messages=[dict(role='user',content=PROMPT)],
              max_tokens=runtime_limits(profile)[1],temperature=1.0,stream=False,chat_template_kwargs=dict(profile['template']))
    if profile['effort']:
        body['reasoning_effort']=profile['effort']
    return body


def run(profile_name, binary, weights_dir, output):
    profile=PROFILES[profile_name]
    context,maximum=runtime_limits(profile)
    started=time.monotonic()
    deadline=started+TOTAL_SECONDS
    record=dict(schema='mj.current-ownweight-development-eval/1', profile=profile_name,
        repo=profile['repo'],revision=profile['revision'],weight_pins=[dict(filename=n,bytes=b,sha256=h) for n,b,h in profile['files']],
        llama_cpp_commit=LLAMA_COMMIT, prompt=PROMPT,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
        input_class='synthetic_public',paid_compute_allocated=False,model_api_credits_used=0,
        reasoning_settings_requested=profile['template'],reasoning_effort_attested=False,
        generation_attempts=0,inference_confirmed=False,context=context,max_output_tokens=maximum,
        runtime='cpu_mmap_fit_off_no_repack',vision_tested=False)
    child=None
    phase='resource_guard'
    generation_start=None
    try:
        if not binary.is_file():
            raise ValueError('BINARY_MISSING')
        weights_dir.mkdir(parents=True,exist_ok=True)
        output.parent.mkdir(parents=True,exist_ok=True)
        record['resources']=resources(weights_dir,profile)
        phase='weight_verification'
        paths=download_weights(weights_dir,profile,deadline)
        record['all_weight_bytes_sha_verified']=True
        # Verify runtime version includes the exact build commit prefix.
        phase='runtime_revision_check'
        version=subprocess.run([str(binary),'--version'],capture_output=True,text=True,timeout=10,check=True)
        if LLAMA_COMMIT[:7] not in version.stdout+version.stderr:
            raise ValueError('RUNTIME_REVISION_MISMATCH')
        phase='runtime_start'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        token=secrets.token_urlsafe(48)
        # The child receives only local runtime environment essentials, never
        # runner credentials, repository tokens, or ambient provider keys.
        env={key:os.environ[key] for key in ('PATH','LD_LIBRARY_PATH','LANG','LC_ALL','TMPDIR') if key in os.environ}
        env['LLAMA_API_KEY']=token
        with (output.parent/(profile_name+'-runtime.log')).open('wb') as log:
            child=subprocess.Popen([str(binary),'-m',str(paths[0]),'-a',profile_name,'--host','127.0.0.1','--port',str(port),
                '--no-webui','--no-mmproj','--cache-ram','0','--load-mode','mmap','--fit','off','--no-repack',
                '-c',str(context),'-b','64','-ub','32','-t','4','-np','1','-ngl','0'],
                env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            phase='readiness'
            load_start=time.monotonic()
            ready_until=min(deadline,load_start+420)
            while time.monotonic()<ready_until:
                if child.poll() is not None:
                    raise RuntimeError('MODEL_START_FAILED')
                try:
                    if request(port,token,'/health',min(2,time_left(ready_until,2))).get('status')=='ok':
                        break
                except (OSError,ValueError,http.client.HTTPException):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('MODEL_READY_TIMEOUT')
            record['load_seconds']=round(time.monotonic()-load_start,3)
            phase='model_identity'
            catalog=request(port,token,'/v1/models',time_left(deadline,10))
            if profile_name not in [item.get('id') for item in catalog.get('data',[]) if isinstance(item,dict)]:
                raise ValueError('LOADED_MODEL_ID_MISMATCH')
            phase='generation'
            record['generation_attempts']=1
            generation_start=time.monotonic()
            response=request(port,token,'/v1/chat/completions',time_left(deadline,900),model_body(profile_name))
            record['generation_seconds']=round(time.monotonic()-generation_start,3)
            if response.get('model')!=profile_name:
                raise ValueError('RESPONSE_MODEL_ID_MISMATCH')
            phase='response_validation'
            choice=response['choices'][0]
            message=choice['message']
            content=message.get('content')
            if not isinstance(content,str):
                content=''
            record.update(final_content=content,finish_reason=choice.get('finish_reason'),model=response.get('model'),
                          usage=response.get('usage'),raw_response=response,
                          inference_confirmed=bool(content.strip()),
                          complete_final_answer=bool(content.strip()) and choice.get('finish_reason')=='stop',
                          output_truncated=choice.get('finish_reason')=='length',
                          grading=grade_text(content),status='completed')
    except Exception as exc:
        # Arbitrary runtime exceptions may contain private local configuration.
        safe_codes={'BINARY_MISSING','RESOURCE_GATE_FAILED','WEIGHT_INTEGRITY_MISMATCH',
                    'TOTAL_DEVELOPMENT_BUDGET_EXCEEDED','RUNTIME_REVISION_MISMATCH',
                    'MODEL_START_FAILED','MODEL_READY_TIMEOUT','LOADED_MODEL_ID_MISMATCH',
                    'RESPONSE_MODEL_ID_MISMATCH','RESPONSE_TOO_LARGE','LOCAL_RUNTIME_HTTP_FAILURE'}
        message=str(exc)
        code=message if message in safe_codes else ('OPERATION_TIMEOUT' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'OPERATION_FAILED')
        record.update(status='failed',error_type=type(exc).__name__,error_code=code,phase=phase,
                      complete_final_answer=False)
        if generation_start is not None:
            record['generation_seconds']=round(time.monotonic()-generation_start,3)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill();child.wait(timeout=5)
        record['elapsed_seconds']=round(time.monotonic()-started,3)
        record['receipt_sha256']=hashlib.sha256(canonical(record)).hexdigest()
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_bytes(canonical(record)+b'\n')
        print('MJ_EVAL_RECEIPT '+canonical(record).decode('utf8'),flush=True)
    return 0 if record['status']=='completed' and record['complete_final_answer'] else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=PROFILES,required=True)
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--weights-dir',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if not args.execute:
        profile=PROFILES[args.profile]
        context,maximum=runtime_limits(profile)
        print(json.dumps(dict(dry_run=True,profile=args.profile,model_inference=False,
                             weights_downloaded=False,llama_cpp_commit=LLAMA_COMMIT,
                             context=context,max_output_tokens=maximum,
                             reasoning_settings_requested=dict(profile['template']))))
        return 0
    if any(value is None for value in (args.binary,args.weights_dir,args.output)):
        parser.error('--execute requires --binary, --weights-dir and --output')
    return run(args.profile,args.binary.resolve(),args.weights_dir.resolve(),args.output.resolve())


if __name__=='__main__':
    raise SystemExit(main())
