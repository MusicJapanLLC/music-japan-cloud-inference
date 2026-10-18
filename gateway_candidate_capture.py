#!/usr/bin/env python3
"""Capture public candidates through the existing owner gateway, without running code.

The private gateway engine is supplied by the operator and verified before import.
No provider URLs, account keys, paid fallback, or generated-code execution are added.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types

import verified_dev_agent as v

ENGINE_SHA256 = '8bc82f9d94908d7e17de20cbf7348ebb59efaba676eab8afda6234eb63fa05f5'
SCHEMA = 'mj.candidate-captures/1.0'
ALLOWED_ROUTES = ('qwen27', 'flashnext', 'deepseek')


def load_engine(path: Path):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ENGINE_SHA256:
        raise ValueError('EXISTING_ENGINE_SHA256_MISMATCH')
    module = types.ModuleType('_mj_pinned_candidate_gateway')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def public_response(data: dict, expected_model: str, engine) -> dict:
    route = next((r for r in engine.ROUTES.values() if r.model == expected_model), None)
    if route is None:
        raise ValueError('UNAPPROVED_MODEL')
    engine.validate_answer(data, route)
    choice = data['choices'][0]
    message = choice['message']
    if message.get('tool_calls') or message.get('refusal'):
        raise ValueError('NON_FINAL_RESPONSE')
    result = {'model': data['model'], 'choices': [{'index': 0,
        'finish_reason': choice['finish_reason'],
        'message': {'role': 'assistant', 'content': message['content']}}]}
    if data.get('usage') is not None:
        result['usage'] = data['usage']
    # Keep only typed identity/cost fields, never arbitrary metadata or reasoning.
    extra = data.get('mj', {})
    if isinstance(extra, dict):
        safe = {}
        for key in ('source_model', 'upstream_model', 'identity_basis'):
            value = extra.get(key)
            if isinstance(value, str) and len(value) <= 2048:
                safe[key] = value
        for key in ('paid_fallback', 'weight_bytes_verified', 'company_credentials_sent'):
            if type(extra.get(key)) is bool:
                safe[key] = extra[key]
        if safe:
            result['mj'] = safe
    return result


def capture(task, engine, caller, routes, *, max_tokens=4096, timeout_seconds=240):
    v.audit_workspace(task.workspace)
    before = v.workspace_manifest(task.workspace)
    with tempfile.TemporaryDirectory(prefix='mj-public-generation-') as td:
        frozen = Path(td) / 'workspace'
        shutil.copytree(task.workspace, frozen)
        if v.workspace_manifest(frozen) != before:
            raise ValueError('TASK_CHANGED_DURING_FREEZE')
        record = _capture_frozen(replace(task, workspace=frozen), engine, caller, routes,
                                 max_tokens=max_tokens, timeout_seconds=timeout_seconds)
        if v.workspace_manifest(frozen) != before:
            raise ValueError('FROZEN_TASK_CHANGED_DURING_CAPTURE')
        return record


def _capture_frozen(task, engine, caller, routes, *, max_tokens=4096, timeout_seconds=240):
    if (not routes or len(routes) > 3 or len(set(routes)) != len(routes)
            or any(name not in ALLOWED_ROUTES for name in routes)
            or type(max_tokens) is not int or not 64 <= max_tokens <= 4096
            or type(timeout_seconds) is not int or not 30 <= timeout_seconds <= 300):
        raise ValueError('CAPTURE_BUDGET_OR_ROUTES_INVALID')
    prompt = v.prompt_for(task)
    manifest_sha = v.sha(v.workspace_manifest(task.workspace))
    prompt_sha = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    start = time.monotonic()

    def one(name):
        route = engine.ROUTES[name]
        request = engine.Task(task_id='candidate-' + name, route=name, prompt=prompt,
                              data_class='public', max_tokens=max_tokens, effort='low')
        launched = time.monotonic()
        row = {'profile': name, 'expected_model': route.model, 'family': route.family}
        try:
            raw = caller.generate(request, route)
            response = public_response(raw, route.model, engine)
            row['responses'] = [response]
            row['capture_status'] = 'complete'
        except Exception as exc:
            # Never serialize exception text from a credential-bearing transport.
            code = exc.code if isinstance(exc, engine.GatewayError) else 'CAPTURE_FAILED'
            row['responses'] = [{'error': {'code': code}}]
            row['capture_status'] = 'error'
        row['generation_seconds'] = round(time.monotonic() - launched, 4)
        return row

    families = [engine.ROUTES[name].family for name in routes]
    if len(set(families)) != len(families):
        raise ValueError('ONE_CANDIDATE_PER_PROVIDER_FAMILY_REQUIRED')
    with ThreadPoolExecutor(max_workers=len(routes)) as pool:
        candidates = list(pool.map(one, routes))
    return {'schema': SCHEMA, 'prompt_sha256': prompt_sha, 'candidates': candidates,
            'task_id': task.task_id, 'task_manifest_sha256': manifest_sha,
            'observed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
            'max_tokens_per_candidate': max_tokens, 'timeout_seconds_per_candidate': timeout_seconds,
            'wall_seconds': round(time.monotonic() - start, 4),
            'data_class': 'public_synthetic', 'fresh_model_calls': len(routes),
            'model_identity_basis': 'existing_gateway_report_only_not_weight_attestation',
            'generated_source_executed_on_controller': False,
            'transport': 'existing_pinned_owner_gateway', 'engine_sha256': ENGINE_SHA256}


def split_capture(record):
    """Strict replay contract plus separate generation timing/provenance."""
    fields = ('profile', 'expected_model', 'expected_upstream_model', 'family', 'responses')
    replay = {'schema': SCHEMA, 'prompt_sha256': record['prompt_sha256'],
              'task_manifest_sha256': record['task_manifest_sha256'],
              'candidates': [{key: row[key] for key in fields if key in row}
                             for row in record['candidates']]}
    metadata = {key: value for key, value in record.items() if key != 'candidates'}
    metadata['schema'] = 'mj.candidate-capture-provenance/1.0'
    metadata['capture_payload_sha256'] = hashlib.sha256(v.canonical(replay)).hexdigest()
    metadata['candidates'] = [{key: value for key, value in row.items() if key != 'responses'}
                               for row in record['candidates']]
    return replay, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--token-file', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--routes', nargs='+', choices=ALLOWED_ROUTES, default=list(ALLOWED_ROUTES))
    parser.add_argument('--output', type=Path, default=Path('captures.json'))
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    task = v.load_task(args.task)
    if not args.execute:
        print(json.dumps({'mode': 'dry_run', 'task_id': task.task_id,
                          'prompt_sha256': hashlib.sha256(v.prompt_for(task).encode()).hexdigest(),
                          'calls': 0, 'generated_source_executed': False}))
        return 0
    if not args.public or args.engine is None or args.token_file is None:
        raise ValueError('EXPLICIT_PUBLIC_TASK_ENGINE_AND_LOCAL_AUTH_REQUIRED')
    metadata_path = args.output.with_suffix('.metadata.json')
    if args.output.exists() or metadata_path.exists():
        raise ValueError('CAPTURE_OUTPUT_EXISTS')
    engine = load_engine(args.engine)
    caller = engine.OwnerGateway(args.host, args.token_file, timeout=args.timeout)
    result = capture(task, engine, caller, args.routes,
                     max_tokens=args.max_tokens, timeout_seconds=args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    replay, metadata = split_capture(result)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(replay, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    with metadata_path.open('x', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'output': str(args.output), 'wall_seconds': result['wall_seconds'],
                      'candidates': [{'profile': r['profile'], 'status': r['capture_status'],
                                      'generation_seconds': r['generation_seconds']} for r in result['candidates']]}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
